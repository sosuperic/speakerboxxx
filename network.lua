local tnt = require 'torchnet'

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

require 'rnn'
require 'nn'
local feedforward = nn.Sequential()
	:add(nn.Linear(98, 128))
	:add(nn.ReLU())
	:add(nn.Dropout(0.5))

local seq_lstm = nn.SeqLSTM(128, 128)
local rnn = nn.Sequential()
	:add(seq_lstm)

local post_rnn = nn.Sequential()
	:add(nn.Linear(128, 1))

local net = nn.Sequential()
	:add(nn.MaskZero(nn.Sequencer(feedforward), 2))
	:add(nn.MaskZero(rnn, 2))
	:add(nn.MaskZero(nn.Sequencer(post_rnn), 2))

local criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.MSECriterion(), 1))

local it = getIterator()
-- for sample in it() do
-- 	-- print(sample.input)
-- 	local act = net:forward(sample.input)
-- 	local err = criterion:forward(act, sample.target)
-- 	print(err)
-- 	-- print(act:size())
-- 	-- print(sample.target:size())
-- 	-- print nn.JoinTable()
-- 	os.exit()
-- end

-- set up training engine:
local engine = tnt.SGDEngine()
local meter  = tnt.AverageValueMeter()
engine.hooks.onStartEpoch = function(state)
   meter:reset()
end
engine.hooks.onForwardCriterion = function(state)
   meter:add(state.criterion.output)
   if state.training then
      print(string.format('avg. loss: %2.4f',
   meter:value()))
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
   lr        = 0.001,
   maxepoch  = 50,
}

-- measure test loss and error:
meter:reset()
engine:test{
   network   = net,
   iterator  = getIterator('test'),
   criterion = criterion,
}
print(string.format('test loss: %2.4f',
   meter:value()))
