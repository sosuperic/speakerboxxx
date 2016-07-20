require 'optim'
require 'rnn'
require 'nn'

local Network = {}
-- local Network = torch.class('Network')

function Network:init(opt)
    self.tnt = require 'torchnet'
    self:setup_model(opt)
    self:setup_engine(opt)
    self:setup_gpu(opt)
    self:make_save_directory(opt)    -- TODO: move this to training so we don't make directories for errrorss
end


function Network:setup_model(opt)
    if opt.model == 'duration' then
        local net, criterion = unpack(require 'duration_model')
        self.net = net
        self.criterion = criterion
        self.model = 'duration'
        self.iterator = self:get_iterator('duration', 'train')
    elseif opt.model == 'acoustic' then
        local net, criterion = unpack(require 'acoustic_model')
        self.net = net
        self.criterion = criterion
        self.model = 'acoustic'
        self.iterator = self:get_iterator('acoustic', 'train')
    else
        print('model must be duration or acoustic')
        os.exit()
    end
end

function Network:setup_engine(opt)
    -- Set up engine and define hooks
    self.engine = self.tnt.OptimEngine()
    -- local val_engine = 
    local train_meter  = self.tnt.AverageValueMeter()
    local val_meter = self.tnt.AverageValueMeter()
    local timer = self.tnt.TimeMeter{unit=true}
    self.engine.hooks.onStart = function(state)
        -- state.epoch
        -- state.maxepoch
        -- state.config
    end
    self.engine.hooks.onStartEpoch = function(state)
        train_meter:reset()
    end
    self.engine.hooks.onForwardCriterion = function(state)
        train_meter:add(state.criterion.output)
        if state.training then
            print(string.format('Epoch: %d; avg. loss: %2.4f',
                state.epoch, train_meter:value()))
        end
    end
    self.engine.hooks.onEndEpoch = function(state)
        timer:incUnit()
        print(string.format('Avg time for one epoch: %.4f',
            timer:value()))

        -- measure test loss and error:
        -- val_meter:reset()
        -- engine:test{
        --    network   = net,
        --    iterator  = getIterator('valid'),
        --    criterion = criterion,
        -- }
        -- print(state.network.output)
        -- print(string.format('Epoch: %d; test loss: %2.4f',
        --    val_meter:value()))
    end
    self.engine.hooks.onEnd = function(state)
        -- torch.save('net.t7', net)
    end
end

function Network:setup_gpu(opt)
    if opt.gpuid >= 0 then
        local id = (3 - opt.gpuid) + 1
        print(string.format('Using GPU %d', id))

        require 'cunn'
        require 'cutorch'
        require 'cudnn'
        cutorch.setDevice(id)
        cutorch.manualSeed(123)

        self.net = net:cuda()
        self.criterion = criterion:cuda()

        -- Copy sample to GPU buffer
        -- alternatively, this logic can be implemented via a TransformDataset
        local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
        engine.hooks.onSample = function(state)
            igpu:resize(state.sample.input:size() ):copy(state.sample.input)
            tgpu:resize(state.sample.target:size()):copy(state.sample.target)
            state.sample.input  = igpu
            state.sample.target = tgpu
        end
    end
end

function Network:make_save_directory(opt)
end

function Network:get_iterator(model, split)
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
                batchsize = 32,
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

function Network:train()
    self.engine:train{
        network   = self.net,
        iterator  = self:get_iterator(self.model, 'train'),
        criterion = self.criterion,
        optimMethod = optim.sgd,
        config = {
            learningRate = 1e-2,
            momentum = 0.9,
        },
        maxepoch  = 100,
    }
end

return Network
