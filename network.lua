local tnt = require 'torchnet'

require 'dataset'
local dataset = tnt.ArcticDataset()
print(dataset)
print(dataset:size())
-- os.exit()

-- use GPU or not:
local cmd = torch.CmdLine()
cmd:option('-usegpu', false, 'use gpu for training')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

-- function that sets of dataset iterator:
local function getIterator()
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      -- init    = function() require 'torchnet' end,
      closure = function()
      	-- Closure's in separate threads, hence why we need to require torchnet
      	-- Also reason (besides modularity) for putting dataset into separate file.
      	-- Requiring packages at the start of this file won't be visible to this thread
      	local tnt = require 'torchnet'
        local dataset = require 'dataset'
		return tnt.BatchDataset{
			batchsize = 32,
			dataset = tnt.ArcticDataset()
		}
      end,
   }
end


-- set up logistic regressor:
local net = nn.Sequential():add(nn.Linear(784,10))
local criterion = nn.CrossEntropyCriterion()

-- set up training engine:
local engine = tnt.SGDEngine()
local meter  = tnt.AverageValueMeter()
local clerr  = tnt.ClassErrorMeter{topk = {1}}
engine.hooks.onStartEpoch = function(state)
   meter:reset()
   clerr:reset()
end
engine.hooks.onForwardCriterion = function(state)
   meter:add(state.criterion.output)
   clerr:add(state.network.output, state.sample.target)
   if state.training then
      print(string.format('avg. loss: %2.4f; avg. error: %2.4f',
         meter:value(), clerr:value{k = 1}))
   end
end

-- set up GPU training:
if config.usegpu then

   -- copy model to GPU:
   require 'cunn'
   net       = net:cuda()
   criterion = criterion:cuda()

   -- copy sample to GPU buffer:
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   engine.hooks.onSample = function(state)
      igpu:resize(state.sample.input:size() ):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      state.sample.input  = igpu
      state.sample.target = tgpu
   end  -- alternatively, this logic can be implemented via a TransformDataset
end

-- train the model:
engine:train{
   network   = net,
   iterator  = getIterator(),
   criterion = criterion,
   lr        = 0.2,
   maxepoch  = 5,
}

-- measure test loss and error:
meter:reset()
clerr:reset()
engine:test{
   network   = net,
   iterator  = getIterator('test'),
   criterion = criterion,
}
print(string.format('test loss: %2.4f; test error: %2.4f',
   meter:value(), clerr:value{k = 1}))

------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------

-- function network:init(linguistic_inputs_path, acoustic_targets_path, duration_targets_path)
-- 	self.batcher = Batcher()
-- 	self.duration_model = require 'duration_model'
-- 	self.acoustic_model = require 'acoustic_model'
-- end

-- function network:setup()
-- end


-- function network:train()
-- end

-- function network:feval()
-- end