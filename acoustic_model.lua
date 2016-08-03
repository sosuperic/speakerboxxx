-- Acoustic model

require 'rnn'
require 'nn'

-- Pre-reurrent layer
local feedforward = nn.Sequential()
    :add(nn.Linear(98, 256))
    :add(nn.ReLU())
    :add(nn.Dropout(0.5))
    :add(nn.Linear(256, 256))
    :add(nn.ReLU())
    :add(nn.Dropout(0.5))
    :add(nn.Linear(256, 256))
    :add(nn.ReLU())
    :add(nn.Dropout(0.5))

-- Recurrent layer
local seq_lstm = nn.SeqLSTM(256, 256)
seq_lstm.maskzero = true
-- local seq_lstm2 = nn.SeqLSTM(256, 256)
-- seq_lstm2.maskzero = true
local rnn = nn.Sequential()
    :add(seq_lstm)
    -- :add(seq_lstm2)
    -- :add(seq_lstm:clone())

-- Post-recurrent layer
local post_rnn = nn.Sequential()
    :add(nn.Linear(256, 84))

-- Glue it togther
local net = nn.Sequential()
    :add(nn.MaskZero(nn.Sequencer(feedforward), 2))
    :add(rnn)
    :add(nn.MaskZero(nn.Sequencer(post_rnn), 2))

-- Criterion
local criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.MSECriterion(), 1))

return {net, criterion}