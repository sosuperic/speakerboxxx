require 'optim'
require 'socket'
require 'pl'
require 'csvigo'
require 'utils.lua_utils'

local Network = {}

function Network:init(opt)
    self.tnt = require 'torchnet'
    self:setup_model(opt)
    self:setup_engine(opt)
    self:setup_gpu(opt)
end


function Network:setup_model(opt)
    local method = {
        sgd = optim.sgd,
        adam = optim.adam,
        adagrad = optim.adagrad,
        adadelta = optim.adadelta,
        rmsprop = optim.rmsprop
    }
    self.optim_method = method[opt.method]

    local split = ternary_op(opt.train_on_valid, 'valid', 'train')

    if opt.model == 'duration' then
        local net, criterion = unpack(require 'duration_model')
        self.net = net
        self.criterion = criterion
        self.model = 'duration'
        self.train_iterator = self:get_iterator(opt.batchsize, 'duration', split)
        self.valid_iterator = self:get_iterator(opt.batchsize, 'duration', 'valid')
    elseif opt.model == 'acoustic' then
        local net, criterion = unpack(require 'acoustic_model')
        self.net = net
        self.criterion = criterion
        self.model = 'acoustic'
        self.train_iterator = self:get_iterator(opt.batchsize, 'acoustic', split)
        self.valid_iterator = self:get_iterator(opt.batchsize, 'acoustic', 'valid')
    else
        print('model must be duration or acoustic')
        os.exit()
    end
end

function Network:setup_engine(opt)
    -- Set up engines and define hooks
    self.engine = self.tnt.OptimEngine()
    self.train_meter  = self.tnt.AverageValueMeter()
    self.valid_engine = self.tnt.SGDEngine()
    self.valid_meter = self.tnt.AverageValueMeter()
    self.timer = self.tnt.TimeMeter{unit=true}

    -- Hooks for main (training) engine
    self.engine.hooks.onStartEpoch = function(state)
        self.train_meter:reset()
    end
    self.engine.hooks.onForwardCriterion = function(state)
        self.train_meter:add(state.criterion.output)
        print(string.format('Epoch: %d; avg. loss: %2.4f',
            state.epoch, self.train_meter:value()))
    end
    self.engine.hooks.onEndEpoch = function(state)
        -- Create directory to save models, etc. at end of first epoch
        -- Do it now to minimize chance of error and useless folder being created 
        if state.epoch == 1 and (not opt.dont_save) then
            self:make_save_directory(opt)
            self:save_opt(opt)
            self:setup_logger(opt)
        end

        -- Get loss on validation
        print('Getting validation loss')
        self.valid_engine:test{
            network   = self.net,
            iterator  = self.valid_iterator,
            criterion = self.criterion
        }
        local train_loss = self.train_meter:value()
        local valid_loss = self.valid_meter:value()
        if not opt.dont_save then
            self.logger:add{train_loss, valid_loss, self.timer:value()}
        end

        -- Timer
        self.timer:incUnit()
        print(string.format('Avg time for one epoch: %.4f',
            self.timer:value()))

        -- Save model and loss
        if (state.epoch % opt.save_model_every_epoch == 0) and (not opt.dont_save) then
            local fn = string.format('net_e%d.t7', state.epoch)
            self:save_network(fn)
        end
    end
    self.engine.hooks.onEnd = function(state)
        if not opt.dont_save then
            local fn = string.format('net_e%d.t7', state.epoch)
            self:save_network(fn)
        end
    end

    -- Hooks for validation engine
    self.valid_engine.hooks.onStartEpoch = function(state)
        self.valid_meter:reset()
    end
    self.valid_engine.hooks.onForwardCriterion = function(state)
        self.valid_meter:add(state.criterion.output)
    end
    self.valid_engine.hooks.onEnd = function(state)
        print(string.format('Validation avg. loss: %2.4f',
            self.valid_meter:value()))
    end
end

function Network:make_save_directory(opt)
    -- Create directory (if necessary) to save models to using current time 
    local cur_dt = os.date('*t', socket.gettime())
    local save_dirname = string.format('%d_%d_%d___%d_%d_%d',
        cur_dt.year, cur_dt.month, cur_dt.day,
        cur_dt.hour, cur_dt.min, cur_dt.sec)
    save_path = path.join(opt.models_dir, self.model, save_dirname)
    make_dir_if_not_exists(save_path)
    self.save_path = save_path
end

function Network:save_opt(opt)
    local fp = path.join(self.save_path, 'cmd')
    torch.save(fp .. '.t7', opt)
    csvigo.save{path=fp .. '.csv', data=convert_table_for_csvigo(opt)}
end

function Network:save_network(fn)
    local fp = path.join(self.save_path, fn)
    print(string.format('Saving model to: %s', fp))
    torch.save(fp, self.net)
end


function Network:setup_logger(opt)
    local fp = path.join(self.save_path, 'stats.log')
    self.logger = optim.Logger(fp)
    self.logger:setNames{'Train loss', 'Valid loss', 'Avg. epoch time'}
end

function Network:setup_gpu(opt)
    if opt.gpuid >= 0 then
        require 'cunn'
        require 'cutorch'
        -- require 'cudnn'
        print(string.format('Using GPU %d', opt.gpuid))
        cutorch.setDevice((3 - opt.gpuid) + 1)
        cutorch.manualSeed(123)

        self.net = self.net:cuda()
        self.criterion = self.criterion:cuda()

        -- Copy sample to GPU buffer
        -- alternatively, this logic can be implemented via a TransformDataset
        local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
        self.engine.hooks.onSample = function(state)
            igpu:resize(state.sample.input:size() ):copy(state.sample.input)
            tgpu:resize(state.sample.target:size()):copy(state.sample.target)
            state.sample.input  = igpu
            state.sample.target = tgpu
        end
        self.valid_engine.hooks.onSample = function(state)
            igpu:resize(state.sample.input:size() ):copy(state.sample.input)
            tgpu:resize(state.sample.target:size()):copy(state.sample.target)
            state.sample.input  = igpu
            state.sample.target = tgpu
        end
    end
end

function Network:get_iterator(batchsize, model, split)
    return self.tnt.ParallelDatasetIterator{
        nthread = 1,
        closure = function()
        -- Closure's in separate threads, hence why we need to require torchnet
        -- Also reason (besides modularity) for putting dataset into separate file.
        -- Requiring packages at the start of this file won't be visible to this thread
            local tnt = require 'torchnet'
            require 'dataset'
            local dataset 
            if model == 'duration' then
                dataset = tnt.DurationDataset(split)
            elseif model == 'acoustic' then
                dataset = tnt.AcousticDataset(split)
            end

            return tnt.BatchDataset{
                batchsize = batchsize,
                dataset = dataset,
            }
        end,
        transform = function(sample)
            local max_seq_len = 0
            for i=1,#sample.input do
                if sample.input[i]:size(1) > max_seq_len then
                    max_seq_len = sample.input[i]:size(1)
                end
            end
            local input_padded = torch.zeros(#sample.input, max_seq_len, sample.input[1]:size(2))
            local target_padded = torch.zeros(#sample.target, max_seq_len, sample.target[1]:size(2))
            for i=1,#sample.input do
                input_padded[{{i}, {1,sample.input[i]:size(1)}, {1,sample.input[i]:size(2)}}] = sample.input[i]
                target_padded[{{i}, {1,sample.target[i]:size(1)}, {1,sample.target[i]:size(2)}}] = sample.target[i]
            end
            sample.input = input_padded:transpose(1,2)     -- Switch batch and seq length
            sample.target = target_padded:transpose(1,2)
            return sample
        end,
    }
end

function Network:train(opt)
    self.engine:train{
        network   = self.net,
        iterator  = self.train_iterator,
        criterion = self.criterion,
        optimMethod = self.optim_method,
        config = {
            learningRate = opt.lr,
            learningRateDecay = opt.lr_decay,
            momentum = opt.mom,
            dampening = opt.damp,
            nesterov = opt.nesterov,
        },
        maxepoch  = opt.maxepochs,
    }
end

return Network
