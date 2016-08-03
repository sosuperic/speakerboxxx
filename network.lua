-- Create a network. Setup, move to GPU if needed, train and test

require 'optim'
require 'socket'
require 'pl'
require 'csvigo'
require 'utils.lua_utils'
require 'hdf5'

local Network = {}

function Network:init(opt)
    self.tnt = require 'torchnet'
    self:setup_gpu(opt)
    if opt.mode == 'train' then
        self:setup_model(opt)
        self:setup_train_engine(opt)
    elseif opt.mode == 'test' then
        self:setup_test(opt)
    end
    self:move_to_gpu(opt)
end

------------------------------------------------------------------------------------------------
-- TRAINING
------------------------------------------------------------------------------------------------
function Network:setup_model(opt)
    local method = {
        sgd = optim.sgd,
        adam = optim.adam,
        adagrad = optim.adagrad,
        adadelta = optim.adadelta,
        rmsprop = optim.rmsprop,
        adamax = optim.adamax
    }
    self.optim_method = method[opt.method]

    local split = ternary_op(opt.train_on_valid, 'valid', 'train')

    if opt.model == 'duration' then
        local net, criterion = unpack(require 'duration_model')
        self.nets = {net}
        self.criterions = {criterion}
        self.model = 'duration'
        self.train_iterator = self:get_iterator(opt.batchsize, 'duration', split)
        self.valid_iterator = self:get_iterator(opt.batchsize, 'duration', 'valid')
    elseif opt.model == 'acoustic' then
        local net, criterion = unpack(require 'acoustic_model')
        self.nets = {net}
        -- torch.save('models/acoustic/2016_7_20___14_30_32/net_e1.t7', net)
        -- os.exit()
        self.criterions = {criterion}
        self.model = 'acoustic'
        self.train_iterator = self:get_iterator(opt.batchsize, 'acoustic', split)
        self.valid_iterator = self:get_iterator(opt.batchsize, 'acoustic', 'valid')

        -- for sample in self.train_iterator() do
        --     print(sample)
        --     os.exit()
        -- end
    else
        print('model must be duration or acoustic')
        os.exit()
    end
end

function Network:setup_train_engine(opt)
    -- Set up engines and define hooks
    self.engine = self.tnt.OptimEngine()
    self.train_meter  = self.tnt.AverageValueMeter()
    self.valid_engine = self.tnt.SGDEngine()
    self.valid_meter = self.tnt.AverageValueMeter()
    self.engines = {self.engine, self.valid_engine}
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
            network   = self.nets[1],
            iterator  = self.valid_iterator,
            criterion = self.criterions[1]
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
    torch.save(fp, self.nets[1])
end


function Network:setup_logger(opt)
    local fp = path.join(self.save_path, 'stats.log')
    self.logger = optim.Logger(fp)
    self.logger:setNames{'Train loss', 'Valid loss', 'Avg. epoch time'}
end

function Network:train(opt)
    self.engine:train{
        network   = self.nets[1],
        iterator  = self.train_iterator,
        criterion = self.criterions[1],
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

------------------------------------------------------------------------------------------------
-- TESTING
------------------------------------------------------------------------------------------------
function Network:setup_test(opt)
    _, self.duration_criterion = unpack(require 'duration_model')
    _, self.acoustic_criterion = unpack(require 'acoustic_model')
    self.criterions = {self.duration_criterion, self.acoustic_criterion}

    print(string.format('Loading duration model from: %s', opt.load_duration_model_path))
    self.duration_model = torch.load(opt.load_duration_model_path)
    print(string.format('Loading acoustic model from: %s', opt.load_acoustic_model_path))
    self.acoustic_model = torch.load(opt.load_acoustic_model_path)
    self.nets = {self.duration_model, self.acoustic_model}

    self.duration_iterator = self:get_iterator(opt.batchsize, 'duration', 'test')
    self.acoustic_iterator = self:get_iterator(opt.batchsize, 'acoustic', 'test')

    self.engine = self.tnt.SGDEngine()
    self.engines = {self.engine}
    self.meter = self.tnt.AverageValueMeter()

    -- Hooks
    self.engine.hooks.onStartEpoch = function(state)
        self.meter:reset()
    end
    self.engine.hooks.onForwardCriterion = function(state)
        self.meter:add(state.criterion.output)
    end
    self.engine.hooks.onEnd = function(state)
        print(string.format('Loss: %2.4f', self.meter:value()))
    end

end

function Network:test_duration_loss(opt)
    print('Testing duration model')
    self.engine:test{
        network = self.duration_model,
        iterator = self.duration_iterator,
        criterion = self.duration_criterion
    }
end

function Network:test_acoustic_loss(opt)
    print('Testing acoustic model')
    self.engine:test{
        network = self.acoustic_model,
        iterator = self.acoustic_iterator,
        criterion = self.acoustic_criterion
    }
end

function Network:test_acoustic_params(opt)
    print('Generating spectral params using only acoustic model')
    for sample in self.acoustic_iterator() do
        if opt.gpuid >= 0 then
            sample.input = sample.input:cuda()
            torch.setdefaulttensortype('torch.CudaTensor')
        end
        local output = self.acoustic_model:forward(sample.input)
        for i, rec in ipairs(sample.rec) do
            local item_output = torch.squeeze(output[{{1,sample.input:size(1)}, {i}, {}}])  -- Remove batch dim
            local f = hdf5.open(path.join(opt.save_test_dir, 'acoustic_only', rec .. '.h5'), 'w')
            f:write('data', item_output:double())
            f:close()
        end
    end
end

function Network:test_full_pipeline(opt)
    print('Testing full pipeline - duration plus acoustic')
    -- Use outputs of duration model to create inputs for acoustic model
    -- Save outputs of acoustic model in order to generate wav files

    for sample in self.duration_iterator() do
        -- Get phoneme durations for each item in sample (sample is a minibatch)
        if opt.gpuid >= 0 then
            sample.input = sample.input:cuda()
            torch.setdefaulttensortype('torch.CudaTensor')
        end
        local output = self.duration_model:forward(sample.input)
        local durations = {}
        for i=1,output:size(2) do               -- batchsize
            local item = output[{{},{i},{1}}]
            local item_durations = {}           -- duration for each phoneme in item (i.e. sequence)
            for j=1,item:size(1) do
                table.insert(item_durations, item[j][1][1])
            end
            table.insert(durations, item_durations)
        end

        -- Get size of input feature tensor for acoustic model
        -- First calculate the number of frames for each item
        -- Sequence dimension is then the max number of frames
        local function calc_num_frames_for_one_item(item_durations)
            -- Round number of frames to closest integer
            local phone_nframes = {}               -- number of frames per phoneme 
            local total_nframes = 0
            for i, duration in ipairs(item_durations) do
                local n = round(duration / 0.005)
                table.insert(phone_nframes, n)
                total_nframes = total_nframes + n
            end
            return {phone_nframes, total_nframes}
        end
        local items_nframes = {}                    -- table of ints
        local items_phone_nframes = {}              -- table of table of ints
        local max_nframes = 0
        for i,item_durations in ipairs(durations) do
            local phone_nframes, total_nframes = unpack(calc_num_frames_for_one_item(item_durations))
            table.insert(items_nframes, total_nframes)
            table.insert(items_phone_nframes, phone_nframes)
            if total_nframes > max_nframes then max_nframes = total_nframes end
        end

        -- Create input feature by stacking linguistic frames
        local acoustic_input_features = torch.zeros(max_nframes, sample.input:size(2), sample.input:size(3))
        for i=1,output:size(2) do                           -- batchsize
            local item_frames = torch.zeros(max_nframes, 1, sample.input:size(3))
            local item_phone_nframes = items_phone_nframes[i]
            local cur_frame_idx = 1
            for j, nframes in ipairs(item_phone_nframes) do
                for k=1,nframes do                          -- stack
                    local frame = sample.input[{{j},{i},{}}]
                    frame[1][1][98] = k                     -- Set position of current frame
                    if cur_frame_idx > item_frames:size(1) then
                        break
                    end
                    item_frames[cur_frame_idx] = frame
                    cur_frame_idx = cur_frame_idx + 1
                end
            end
            acoustic_input_features[{{},{i},{}}] = item_frames
        end

        -- Pass features into acoustic_model and save
        local output = self.acoustic_model:forward(acoustic_input_features)
        for i, rec in ipairs(sample.rec) do
            print('Saving ' .. rec)
            local item_output = torch.squeeze(output[{{1,items_nframes[i]}, {i}, {}}])  -- Remove batch dim
            local f = hdf5.open(path.join(opt.save_test_dir, 'full_pipeline', rec .. '.h5'), 'w')
            f:write('data', item_output:double())
            f:close()
        end
    end
end
------------------------------------------------------------------------------------------------
-- USED BY BOTH TRAINING AND TESTING
------------------------------------------------------------------------------------------------
function Network:setup_gpu(opt)
    if opt.gpuid >= 0 then
        require 'cunn'
        require 'cutorch'
        -- require 'cudnn'
        print(string.format('Using GPU %d', opt.gpuid))
        cutorch.setDevice((3 - opt.gpuid) + 1)
        cutorch.manualSeed(123)
    end
end

function Network:move_to_gpu(opt)
    if opt.gpuid >= 0 then
        for i,net in ipairs(self.nets) do
            net = net:cuda()
        end

        for i,criterion in ipairs(self.criterions) do
            criterion = criterion:cuda()
        end
        -- self.net = self.net:cuda()
        -- self.criterion = self.criterion:cuda()

        -- Copy sample to GPU buffer
        -- alternatively, this logic can be implemented via a TransformDataset
        local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
        for i,engine in ipairs(self.engines) do
            engine.hooks.onSample = function(state)
                igpu:resize(state.sample.input:size() ):copy(state.sample.input)
                tgpu:resize(state.sample.target:size()):copy(state.sample.target)
                state.sample.input  = igpu
                state.sample.target = tgpu
            end
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

return Network
