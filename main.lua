-- Main file to train network
require 'pl'
require 'utils.lua_utils'

---------------------------------------------------------------------------
-- Params
---------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Example:')
cmd:text('th main.lua  -gpuid 0 -model duration -notes linear254to256_linear256to128_lstm128to128_linear128to1 -save_model_every_epoch 10 -maxepochs 100 -lr 0.001 -method adam')
cmd:text('$ th main.lua  -gpuid 1 -maxepochs 300 -save_model_every_epoch 10 -lr 0.0005 -method adam -model acoustic -notes linear254to512_linear512to512_lstm512to256_lstm256to256_linear256to84__QUINPHONE_f0INTERPOLATE')
cmd:text('$ th main.lua -gpuid 0 -mode test -load_duration_model_path models/cmuarctic/duration/2016_8_3___15_5_38/net_e100.t7 -load_acoustic_model_path models/cmuarctic/acoustic/2016_8_3___17_24_13/net_e270.t7')
cmd:text('$ th main.lua -mode test -load_duration_model_path models/cmuarctic/duration/2016_7_20___5_16_19/net_e9.t7 -load_acoustic_model_path models/cmuarctic/acoustic/2016_7_20___14_30_32/net_e1.t7')
cmd:text('Options:')
-- Training, Testing
cmd:option('-model', '', 'duration or acoustic')
cmd:option('-mode', 'train', 'train or test')
cmd:option('-dataset', 'cmuarctic', 'blizzard2013 or cmuarctic')
cmd:option('-two_datasets', false, 'use two cmu datasets - slt and clb (both female)')
cmd:option('-train_on_valid', false, 'Train on valid instead of training. Used to debug because it is faster')
cmd:option('-batchsize', 32, 'number of examples in minibatch')
cmd:option('-maxepochs', 100, 'max number of epochs to train for')
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
cmd:option('-gpuid', -1, 'ID of gpu to run on')
cmd:option('-dont_save', false, 'Save or not. Use true for testing / debugging')
cmd:option('-save_model_every_epoch', 5, 'how often to save model')
cmd:option('-notes', '', 'String of notes, e.g. using batch norm. To keep track of iterative testing / small modifications')

local opt = cmd:parse(arg)

local function get_dir_name(dataset, two_datasets)
    dir_name = ternary_op(
        opt.dataset == 'blizzard2013',
        'blizzard2013',
        ternary_op(
            opt.two_datasets,
            'cmuarctic_two_datasets', 
            'cmuarctic'
        )
    )
    return dir_name
end

opt.models_dir = path.join('models', get_dir_name(opt.dataset, opt.two_datasets))
opt.save_test_dir = path.join('outputs', get_dir_name(opt.dataset, opt.two_datasets), 'spectral')

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
    -- network:test_acoustic_params(opt)
    network:test_full_pipeline(opt)
end
