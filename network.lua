require 'optim'

local tnt = require 'torchnet'

-- use GPU or not:
local cmd = torch.CmdLine()
cmd:option('-usegpu', false, 'use gpu for training')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

-- function that sets of dataset iterator:
local function getIterator(split)
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      closure = function()
      	-- Closure's in separate threads, hence why we need to require torchnet
      	-- Also reason (besides modularity) for putting dataset into separate file.
      	-- Requiring packages at the start of this file won't be visible to this thread
      	local tnt = require 'torchnet'
        local dataset = require 'dataset'
		return tnt.BatchDataset{
			batchsize = 32,
			-- dataset = tnt.DurationDataset(split)
			dataset = tnt.AcousticDataset(split)
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
      	-- print(input_padded)
      	for i=1,#sample.input do
      		input_padded[{{i}, {1,sample.input[i]:size(1)}, {1,sample.input[i]:size(2)}}] = sample.input[i]
      		target_padded[{{i}, {1,sample.target[i]:size(1)}, {1,sample.target[i]:size(2)}}] = sample.target[i]
      	end
      	sample.input = input_padded:transpose(1,2) 	-- Switch batch and seq length
      	sample.target = target_padded:transpose(1,2)
      	return sample
  	  end,
   }
end

-- local net, criterion = unpack(require 'duration_model')
local net, criterion = unpack(require 'acoustic_model')

-- local it = getIterator('train_acoustic')
-- for sample in it() do
-- 	local output = net:forward(sample.input)
-- 	local loss = criterion:forward(output, sample.target)
-- 	print(output:size())
-- 	print(sample.target:size())
-- 	print(loss)
-- 	os.exit()

-- 	-- print(sample.input:size())
-- 	-- print(sample.target:size())
-- 	-- os.exit()
-- end
-- os.exit()

-- Set up engine and define hooks
local engine = tnt.OptimEngine()
-- local val_engine = 
local train_meter  = tnt.AverageValueMeter()
local val_meter = tnt.AverageValueMeter()
local timer = tnt.TimeMeter{unit=true}
engine.hooks.onStart = function(state)
	-- state.epoch
	-- state.maxepoch
	-- state.config
end
engine.hooks.onStartEpoch = function(state)
   train_meter:reset()
end
engine.hooks.onForwardCriterion = function(state)
   train_meter:add(state.criterion.output)
   if state.training then
      print(string.format('Epoch: %d; avg. loss: %2.4f',
   		state.epoch, train_meter:value()))
   end
end
engine.hooks.onEndEpoch = function(state)
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
engine.hooks.onEnd = function(state)
	-- torch.save('net.t7', net)
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
   -- iterator  = getIterator('train_duration'),
   iterator  = getIterator('train_acoustic'),
   criterion = criterion,
   optimMethod = optim.sgd,
   config = {
    learningRate = 1e-2,
    momentum = 0.9,
 	},
   maxepoch  = 100,
}
