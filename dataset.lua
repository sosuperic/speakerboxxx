require 'utils.lua_utils'
require 'hdf5'
require 'pl'

local LINGUISTIC_INPUTS_PATH = 'data/processed/cmu_us_slt_arctic/linguistic_inputs/'
local ACOUSTIC_TARGETS_PATH = 'data/processed/cmu_us_slt_arctic/acoustic_targets/'
local DURATION_TARGETS_PATH = 'data/processed/cmu_us_slt_arctic/duration_targets/'

local tnt = require 'torchnet'
local ArcticDataset, Dataset = torch.class('tnt.ArcticDataset', 'tnt.Dataset', tnt)

local function create_full_path(base, table_of_fns)
	for i, fn in ipairs(table_of_fns) do
		table_of_fns[i] = path.join(base, fn)
	end
	return table_of_fns
end

local function load_hdf5_array(path, name)
	local file = hdf5.open(path, 'r')
	local data = file:read(name):all()
	return data
end

function ArcticDataset:__init()
	self.linguistic_input_fps = create_full_path(LINGUISTIC_INPUTS_PATH, scandir(LINGUISTIC_INPUTS_PATH))
	self.acoustic_target_fps = create_full_path(ACOUSTIC_TARGETS_PATH, scandir(ACOUSTIC_TARGETS_PATH))
	self.duration_target_fps = create_full_path(DURATION_TARGETS_PATH, scandir(DURATION_TARGETS_PATH))
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