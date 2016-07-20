-- Main file to train network

---------------------------------------------------------------------------
-- Params
---------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Example:')
cmd:text('$ th main.lua -model duration -save_model_every_epoch 5')
cmd:text('Options:')
-- Training
cmd:option('-model', '', 'duration or acoustic')
cmd:option('-train_on_valid', false, 'Train on valid instead of training. Used to debug because it is faster')
cmd:option('-batchsize', 32, 'number of examples in minibatch')
cmd:option('-maxepochs', 100, 'max number of epochs to train for')
-- Load model
-- cmd:option('-load_model', false, 'start training from existing model')
-- cmd:option('-load_model_dir', '', 'directory to load model and params from')
-- cmd:option('-load_model_fn', '', 'fn of model to load')
-- Optimization
cmd:option('-method','sgd', 'which optimization method to use')
cmd:option('-lr', 1e-3, 'learning rate')
cmd:option('-lr_decay', 0, 'learning rate decay')
cmd:option('-mom', 0, 'momentum')
cmd:option('-damp', 0, 'dampening')
cmd:option('-nesterov', false, 'Nesterov momentum')
-- Bookkeeping
cmd:option('-models_dir', 'models', 'directory to save models to')
cmd:option('-gpuid', -1, 'ID of gpu to run on')
cmd:option('-dont_save', false, 'Save or not. Use true for testing / debugging')
cmd:option('-save_model_every_epoch', 5, 'how often to save model')
cmd:option('-notes', '', 'String of notes, e.g. using batch norm. To keep track of iterative testing / small modifications')
local opt = cmd:parse(arg)

---------------------------------------------------------------------------
-- Training
---------------------------------------------------------------------------
local network = require 'Network'
network:init(opt)
network:train(opt)

