-- Main file to train network

---------------------------------------------------------------------------
-- Params
---------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Example:')
cmd:text('$ th main.lua -model duration -save_model_every_epoch 5')
cmd:text('$ th main.lua -gpuid 1 -mode test -load_duration_model_path models/duration/2016_7_20___13_40_15/net_e50.t7 -load_acoustic_model_path models/acoustic/2016_8_2___18_21_10/net_e30.t7')
cmd:text('$ th main.lua -mode test -load_duration_model_path models/duration/2016_7_20___5_16_19/net_e9.t7 -load_acoustic_model_path models/acoustic/2016_7_20___14_30_32/net_e1.t7')
cmd:text('Options:')
-- Training, Testing
cmd:option('-model', '', 'duration or acoustic')
cmd:option('-mode', 'train', 'train or test')
cmd:option('-train_on_valid', false, 'Train on valid instead of training. Used to debug because it is faster')
cmd:option('-batchsize', 32, 'number of examples in minibatch')
cmd:option('-maxepochs', 100, 'max number of epochs to train for')
cmd:option('-save_test_dir', 'outputs/spectral', 'dir to save output spectral parameters')
-- Load model
-- cmd:option('-load_model', false, 'start training from existing model')
cmd:option('-load_duration_model_path', '', 'path to duration model')
cmd:option('-load_acoustic_model_path', '', 'path to acoustic model')
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
local network = require 'network'
network:init(opt)

if opt.mode == 'train' then
    network:train(opt)
elseif opt.mode == 'test' then
    -- network:test_duration_loss(opt)
    -- network:test_acoustic_loss(opt)
    network:test_acoustic_params(opt)
    network:test_full_pipeline(opt)
end
