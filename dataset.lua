-- Implement dataset interface for torchnet
-- Load pre-processed split and sort by sequence length

require 'utils.lua_utils'
require 'hdf5'
require 'pl'

local LINGUISTIC_INPUTS_PATH = 'data/processed/cmu_us_slt_arctic/linguistic_inputs/'
local ACOUSTIC_TARGETS_PATH = 'data/processed/cmu_us_slt_arctic/acoustic_targets/'
local DURATION_TARGETS_PATH = 'data/processed/cmu_us_slt_arctic/duration_targets/'
local SPLIT_PATH = 'data/processed/'

local tnt = require 'torchnet'
local ArcticDataset, Dataset = torch.class('tnt.ArcticDataset', 'tnt.Dataset', tnt)

local function create_full_path(base, table_of_recs)
	local fps = {}
	for i, fn in ipairs(table_of_recs) do
		table.insert(fps, path.join(base, fn .. '.h5'))
	end
	return fps
end

local function load_hdf5_array(path, name)
	local file = hdf5.open(path, 'r')
	local data = file:read(name):all()
	return data
end

function ArcticDataset:__init(split)
	self.recs = lines_from(path.join(SPLIT_PATH, split .. '.txt'))
	self.linguistic_input_fps = create_full_path(LINGUISTIC_INPUTS_PATH, self.recs)
	self.acoustic_target_fps = create_full_path(ACOUSTIC_TARGETS_PATH, self.recs)
	self.duration_target_fps = create_full_path(DURATION_TARGETS_PATH, self.recs)
	self.n = #self.linguistic_input_fps
end

function ArcticDataset:size()
	return self.n
end

function ArcticDataset:get(idx)
	return {
				input = load_hdf5_array(self.linguistic_input_fps[idx], 'x'),
				target = load_hdf5_array(self.duration_target_fps[idx], 'y')
			}
end