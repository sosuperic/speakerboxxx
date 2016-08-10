-- Create the training, validation, and test data splits for CMUArctic
-- Data splits are sorted by sequence length
-- Write recording names to tr.txt, va.txt, te.txt

require 'pl'
require 'hdf5'
require 'utils.lua_utils'

local LINGUISTIC_INPUTS_PATH = 'data/processed/cmu_us_slt_arctic/linguistic_inputs/'
local UTT_DUR_IDX = 97
-- local LINGUISTIC_INPUTS_PATH = 'data/processed/cmu_us_slt_arctic/linguistic_inputs_plus/'
-- local UTT_DUR_IDX = 253


local OUT_PATH = 'data/processed/'

local NUM_VALID = 100
local NUM_TEST = 100
-- Remaining are training

-- Split into buckets. Also load the sequence length for each recording
local function load_hdf5_array(path, name)
	local file = hdf5.open(path, 'r')
	local data = file:read(name):all()
	return data
end

local files = scandir(LINGUISTIC_INPUTS_PATH)
local tr, va, te = {}, {}, {}
for i, fn in ipairs(files) do
	local rec = fn:sub(1, #fn-3)		-- Strip '.h5' -> arctic_a0001

	-- [*]_seq_len is sequence length for * model
	-- We sort so we can do curriculum learning
	local x = load_hdf5_array(path.join(LINGUISTIC_INPUTS_PATH, fn), 'x')
	local duration_seq_len = x:size(1)
	local acoustic_seq_len = x[1][UTT_DUR_IDX] 		-- Second to last feature is total time of sequence

	if i <= NUM_VALID then table.insert(va, {rec, duration_seq_len, acoustic_seq_len})
	elseif (i > NUM_VALID and (i <= NUM_VALID + NUM_TEST)) then table.insert(te, {rec, duration_seq_len, acoustic_seq_len})
	else table.insert(tr, {rec, duration_seq_len, acoustic_seq_len})
	end
end

-- Sort each split by sequence length. Then write recording names to file
function sort_by_seq_length(split, idx)
	-- idx=2 -> phoneme; idx=3 -> duration
	local function sorter(a, b)
		if (a[idx] < b[idx]) then return true else return false end
	end
	table.sort(split, sorter)
	return split
end
local function write_to_file(tbl, fn)
	local f = io.open(fn, 'w')
	for i, rec_len_tuple in ipairs(tbl) do
		if i < #tbl then
			f:write(rec_len_tuple[1] .. '\n')
		else
			f:write(rec_len_tuple[1])
		end
	end
end

local tr_duration = sort_by_seq_length(tr, 2)
local va_duration = sort_by_seq_length(va, 2)
local te_duration = sort_by_seq_length(te, 2)
write_to_file(tr_duration, path.join(OUT_PATH, 'duration', 'train.txt'))
write_to_file(va_duration, path.join(OUT_PATH, 'duration', 'valid.txt'))
write_to_file(te_duration, path.join(OUT_PATH, 'duration', 'test.txt'))

local tr_acoustic = sort_by_seq_length(tr, 3)
local va_acoustic = sort_by_seq_length(va, 3)
local te_acoustic = sort_by_seq_length(te, 3)
write_to_file(tr_acoustic, path.join(OUT_PATH, 'acoustic', 'train.txt'))
write_to_file(va_acoustic, path.join(OUT_PATH, 'acoustic', 'valid.txt'))
write_to_file(te_acoustic, path.join(OUT_PATH, 'acoustic', 'test.txt'))
