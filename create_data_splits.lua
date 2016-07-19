-- Create the training, validation, and test data splits for CMUArctic
-- Data splits are sorted by sequence length
-- Write recording names to tr.txt, va.txt, te.txt

require 'pl'
require 'hdf5'
require 'utils.lua_utils'

local LINGUISTIC_INPUTS_PATH = 'data/processed/cmu_us_slt_arctic/linguistic_inputs/'
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
	local seq_len = load_hdf5_array(path.join(LINGUISTIC_INPUTS_PATH, fn), 'x'):size(1)

	if i <= NUM_VALID then table.insert(va, {rec, seq_len})
	elseif (i > NUM_VALID and (i <= NUM_VALID + NUM_TEST)) then table.insert(te, {rec, seq_len})
	else table.insert(tr, {rec, seq_len})
	end
end

-- Sort each split by sequence length
function sort_by_seq_length(split)
	local function sorter(a, b)
		if (a[2] < b[2]) then return true else return false end
	end
	table.sort(split, sorter)
	return split
end
local tr = sort_by_seq_length(tr)
local va = sort_by_seq_length(va)
local te = sort_by_seq_length(te)

-- Write recording names to file
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

write_to_file(tr, path.join(OUT_PATH, 'train.txt'))
write_to_file(va, path.join(OUT_PATH, 'valid.txt'))
write_to_file(te, path.join(OUT_PATH, 'test.txt'))