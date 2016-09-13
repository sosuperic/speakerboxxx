-- Create the training, validation, and test data splits for
-- Data splits are sorted by sequence length
-- Write recording names to tr.txt, va.txt, te.txt

require 'pl'
require 'hdf5'
require 'utils.lua_utils'

------------------------------------------------------------------------------------------------------------
-- Datasets, Parameters, Paths
------------------------------------------------------------------------------------------------------------
local PARAMS = {}

-- Currently for all datasets, when linguistic inputs include quinphone identities
PARAMS['UTT_DUR_IDX'] = 253         -- 97 without quinphone

PARAMS['cmuarctic'] = {}
PARAMS['cmuarctic']['NUM_VALID'] = 100
PARAMS['cmuarctic']['NUM_TEST'] = 100
PARAMS['cmuarctic']['OUT_PATH1'] = 'data/processed/cmu_us_slt_arctic/'
PARAMS['cmuarctic']['OUT_PATH2'] = 'data/processed/cmu_us_clb_arctic/'
PARAMS['cmuarctic']['LINGUISTIC_INPUTS_PATH1'] = 'data/processed/cmu_us_slt_arctic/linguistic_inputs_plus/'
PARAMS['cmuarctic']['LINGUISTIC_INPUTS_PATH2'] = 'data/processed/cmu_us_clb_arctic/linguistic_inputs_plus/'

PARAMS['blizzard2013'] = {}
PARAMS['blizzard2013']['NUM_VALID'] = 750
PARAMS['blizzard2013']['NUM_TEST'] = 750
PARAMS['blizzard2013']['OUT_PATH'] = 'data/processed/blizzard2013/'
PARAMS['blizzard2013']['LINGUISTIC_INPUTS_PATH'] = 'data/processed/blizzard2013/linguistic_inputs/'

------------------------------------------------------------------------------------------------------------
-- Class definition
------------------------------------------------------------------------------------------------------------
local class = require 'class'
local DataSplitter = class('DataSplitter')

function DataSplitter:__init(opt)
    self.dataset = opt.dataset
end


-- Split into buckets. Also load the sequence length for each recording
function DataSplitter:load_hdf5_array(path, name)
    -- print('path', path)
    local file = hdf5.open(path, 'r')
    local data = file:read(name):all()
    file:close()
    return data
end

-- Sort each split by sequence length. Then write recording names to file
function DataSplitter:sort_by_seq_length(split, idx)
    -- idx=2 -> phoneme; idx=3 -> duration
    local function sorter(a, b)
        if (a[idx] < b[idx]) then return true else return false end
    end
    table.sort(split, sorter)
    return split
end

function DataSplitter:write_to_file(tbl, fn)
    local f = io.open(fn, 'w')
    for i, rec_len_tuple in ipairs(tbl) do
        if i < #tbl then
            f:write(rec_len_tuple[1] .. '\n')
        else
            f:write(rec_len_tuple[1])
        end
    end
end

-- TODO: For blizzard 1hr_train, 5hr_train, 10hr_train , etc.
function DataSplitter:split_helper(num_valid, num_test, out_path, linguistic_inputs_path)
    local filepaths = dir.getfiles(linguistic_inputs_path)

    -- Randomly shuffle into train, valid, test
    local tr, va, te = {}, {}, {}
    idx = 0
    for _, fp in ipairs(filepaths) do
        local rec, _ = path.splitext(path.basename(fp))

        if fn:sub(1,1) ~= '.' then
            -- [*]_seq_len is sequence length for * model
            -- We sort so we can do curriculum learning
            -- print(path.join(linguistic_inputs_path, fn), type(path.join(linguistic_inputs_path, fn)))
            print(fp) 
            local x = self:load_hdf5_array(fp, 'x')
            -- local x = self:load_hdf5_array(path.join(linguistic_inputs_path, fn), 'x')
            local duration_seq_len = x:size(1)
            local acoustic_seq_len = x[1][PARAMS['UTT_DUR_IDX']]      -- Second to last feature is total time of sequence

            if idx <= num_valid then table.insert(va, {rec, duration_seq_len, acoustic_seq_len})
            elseif (idx > num_valid and (idx <= num_valid + num_test)) then table.insert(te, {rec, duration_seq_len, acoustic_seq_len})
            else table.insert(tr, {rec, duration_seq_len, acoustic_seq_len})
            end

            idx = idx + 1
        end
    end

    -- Sort and save to file
    local tr_duration = self:sort_by_seq_length(tr, 2)
    local va_duration = self:sort_by_seq_length(va, 2)
    local te_duration = self:sort_by_seq_length(te, 2)
    self:write_to_file(tr_duration, path.join(out_path, 'duration_split', 'train.txt'))
    self:write_to_file(va_duration, path.join(out_path, 'duration_split', 'valid.txt'))
    self:write_to_file(te_duration, path.join(out_path, 'duration_split', 'test.txt'))

    local tr_acoustic = self:sort_by_seq_length(tr, 3)
    local va_acoustic = self:sort_by_seq_length(va, 3)
    local te_acoustic = self:sort_by_seq_length(te, 3)
    self:write_to_file(tr_acoustic, path.join(out_path, 'acoustic_split', 'train.txt'))
    self:write_to_file(va_acoustic, path.join(out_path, 'acoustic_split', 'valid.txt'))
    self:write_to_file(te_acoustic, path.join(out_path, 'acoustic_split', 'test.txt'))
end

function DataSplitter:split()
    if self.dataset == 'blizzard2013' then
        self:split_helper(
                PARAMS[self.dataset]['NUM_VALID'],
                PARAMS[self.dataset]['NUM_TEST'],
                PARAMS[self.dataset]['OUT_PATH'],
                PARAMS[self.dataset]['LINGUISTIC_INPUTS_PATH']
            )
    elseif self.dataset == 'cmuarctic' then
        -- Dataset 1
        self:split_helper(
            PARAMS[self.dataset]['NUM_VALID'],
            PARAMS[self.dataset]['NUM_TEST'],
            PARAMS[self.dataset]['OUT_PATH1'],
            PARAMS[self.dataset]['LINGUISTIC_INPUTS_PATH1']
        )
        -- Dataset 2
        self:split_helper(
            PARAMS[self.dataset]['NUM_VALID'],
            PARAMS[self.dataset]['NUM_TEST'],
            PARAMS[self.dataset]['OUT_PATH2'],
            PARAMS[self.dataset]['LINGUISTIC_INPUTS_PATH2']
        )
    else
        print('Dataset must be blizzard2013 or cmuarctic')
    end
end

------------------------------------------------------------------------------------------------------------
-- Main
------------------------------------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:option('-dataset', 'cmuarctic', 'blizzard2013 or cmuarctic')
local opt = cmd:parse(arg)

ds = DataSplitter(opt)
ds:split()
