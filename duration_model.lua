-- Duration model

require 'rnn'
require 'nn'

-- Pre-reurrent layer
local feedforward = nn.Sequential()
	:add(nn.Linear(254, 256))
	:add(nn.ReLU())
	:add(nn.Dropout(0.5))
    :add(nn.Linear(256, 128))
    :add(nn.ReLU())
    :add(nn.Dropout(0.5))

-- Recurrent layer
local seq_lstm = nn.SeqLSTM(128, 128)
seq_lstm.maskzero=true
local rnn = nn.Sequential()
	:add(seq_lstm)

-- Post-recurrent layer
local post_rnn = nn.Sequential()
	:add(nn.Linear(128, 1))

-- Glue it togther
local net = nn.Sequential()
	:add(nn.MaskZero(nn.Sequencer(feedforward), 2))
	:add(rnn)
	:add(nn.MaskZero(nn.Sequencer(post_rnn), 2))

-- Criterion
local criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.MSECriterion(), 1))

return {net, criterion}