-- Main file to train network

---------------------------------------------------------------------------
-- Params
---------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Example:')
cmd:text('$ th g2p.lua -use_google_model -lr 0.001 -lr_decay 1e-9 -mom 0.9 -damp 0.9 -nesterov true')
cmd:text('$ th g2p.lua -use_google_model -load_model -load_model_dir 2016_6_23___12_14_32 -load_model_fn e4_2.7609.t7')
cmd:text('Options:')
-- Training
cmd:option('-model', '', 'duration or acoustic')
cmd:option('-train_on_valid', false, 'Train on valid instead of training. Used to debug because it is faster')
cmd:option('-batchsize', 32, 'number of examples in minibatch')
cmd:option('-epochs', 100, 'max number of epochs to train for')
-- Load model
cmd:option('-load_model', false, 'start training from existing model')
cmd:option('-load_model_dir', '', 'directory to load model and params from')
cmd:option('-load_model_fn', '', 'fn of model to load')
-- Optimization
cmd:option('-use_rmsprop', false, 'Use RMSprop to optimize')
cmd:option('-use_adam', false, 'Use Adam to optimize')
cmd:option('-lr', 0.1, 'learning rate')
cmd:option('-lr_decay', 0, 'learning rate decay')
cmd:option('-weight_decay', 0, 'weight decay')
cmd:option('-mom', 0, 'momentum')
cmd:option('-damp', 0, 'dampening')
cmd:option('-nesterov', false, 'Nesterov momentum')
-- Bookkeeping
cmd:option('-models_dir', 'models', 'directory to save models to')
cmd:option('-gpuid', -1, 'ID of gpu to run on')
cmd:option('-save_model_every_epoch', 1, 'how often to save model')
cmd:option('-notes', '', 'String of notes, e.g. using batch norm. To keep track of iterative testing / small modifications')
local opt = cmd:parse(arg)

---------------------------------------------------------------------------
-- Training
---------------------------------------------------------------------------
local network = require 'Network'
network:init(opt)
network:train()

