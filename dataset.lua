-- Implement dataset interface for torchnet
-- Load pre-processed split

require 'utils.lua_utils'
require 'hdf5'
require 'pl'
local tnt = require 'torchnet'

------------------------------------------------------------------------------------------------------------
-- Datasets, Parameters, Paths
------------------------------------------------------------------------------------------------------------
-- CMUArctic, 2 datasets of CMUArctic
local PATHS = {}

PATHS['cmuarctic'] = {}
PATHS['cmuarctic']['SPLIT_PATH1'] = 'data/processed/cmu_us_slt_arctic/'
PATHS['cmuarctic']['SPLIT_PATH2'] = 'data/processed/cmu_us_clb_arctic/'
PATHS['cmuarctic']['LINGUISTIC_INPUTS_PATH1'] = 'data/processed/cmu_us_slt_arctic/linguistic_inputs_plus/'
PATHS['cmuarctic']['LINGUISTIC_INPUTS_PATH2'] = 'data/processed/cmu_us_clb_arctic/linguistic_inputs_plus/'
PATHS['cmuarctic']['ACOUSTIC_TARGETS_PATH1'] = 'data/processed/cmu_us_slt_arctic/acoustic_targets_f0interpolate/'
PATHS['cmuarctic']['ACOUSTIC_TARGETS_PATH2'] = 'data/processed/cmu_us_clb_arctic/acoustic_targets_f0interpolate/'
PATHS['cmuarctic']['DURATION_TARGETS_PATH1'] = 'data/processed/cmu_us_slt_arctic/duration_targets/'
PATHS['cmuarctic']['DURATION_TARGETS_PATH2'] = 'data/processed/cmu_us_clb_arctic/duration_targets/'
-- Still experimental / old
-- local ACOUSTIC_TARGETS_PATH1 = 'data/processed/cmu_us_slt_arctic/acoustic_targets_normalized/'
-- local ACOUSTIC_TARGETS_PATH1 = 'data/processed/cmu_us_slt_arctic/acoustic_targets_zeromean/'
-- local ACOUSTIC_TARGETS_PATH1 = 'data/processed/cmu_us_slt_arctic/acoustic_targets/'
-- local ACOUSTIC_TARGETS_PATH1 = 'data/processed/cmu_us_slt_arctic/acoustic_targets_f0interpolate_normalized/'

-- Note: blizzard2013 doesn't have two datasets. Only split_path1, etc. is set
PATHS['blizzard2013'] = {}
PATHS['blizzard2013']['SPLIT_PATH1'] = 'data/processed/blizzard2013/'
PATHS['blizzard2013']['LINGUISTIC_INPUTS_PATH1'] = 'data/processed/blizzard2013/linguistic_inputs/'
PATHS['blizzard2013']['ACOUSTIC_TARGETS_PATH1'] = 'data/processed/blizzard2013/acoustic_targets/'
PATHS['blizzard2013']['DURATION_TARGETS_PATH1'] = 'data/processed/blizzard2013/duration_targets/'

----------------------------------------------------------------------------------------------------------------
-- Helper functions
----------------------------------------------------------------------------------------------------------------
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
	file:close()
	return data
end

-------------------------------------------------------------------- --------------------------------------------
-- Duration model dataset loader
----------------------------------------------------------------------------------------------------------------
local DurationDataset, Dataset = torch.class('tnt.DurationDataset', 'tnt.Dataset', tnt)

function DurationDataset:__init(dataset, split, two_datasets)
	self.recs1 = lines_from(path.join(PATHS[dataset]['SPLIT_PATH1'], 'duration_split', split .. '.txt'))
	self.linguistic_input_fps = create_full_path(PATHS[dataset]['LINGUISTIC_INPUTS_PATH1'], self.recs1)
	self.duration_target_fps = create_full_path(PATHS[dataset]['DURATION_TARGETS_PATH1'], self.recs1)

	if two_datasets then
		self.recs2 = lines_from(path.join(PATHS[dataset]['SPLIT_PATH2'], 'duration_split', split .. '.txt'))
		self.linguistic_input_fps = interleaf_tables(self.linguistic_input_fps,
			create_full_path(PATHS[dataset]['LINGUISTIC_INPUTS_PATH2'], self.recs2))
		self.duration_target_fps = interleaf_tables(self.duration_target_fps,
			create_full_path(PATHS[dataset]['DURATION_TARGETS_PATH2'], self.recs2))
	end

	self.n = #self.linguistic_input_fps
end

function DurationDataset:size()
	return self.n
end

function DurationDataset:get(idx)
	local rec = self.linguistic_input_fps[idx]:match( "([^/]+)$"):gsub(".h5", "")
	return {
				input = load_hdf5_array(self.linguistic_input_fps[idx], 'x'),
				target = load_hdf5_array(self.duration_target_fps[idx], 'y'),
				rec = rec
			}
end

----------------------------------------------------------------------------------------------------------------
-- Duration model dataset loader
----------------------------------------------------------------------------------------------------------------
local AcousticDataset, Dataset = torch.class('tnt.AcousticDataset', 'tnt.Dataset', tnt)

function AcousticDataset:__init(dataset, split, two_datasets)
	self.recs1 = lines_from(path.join(PATHS[dataset]['SPLIT_PATH1'], 'acoustic_split', split .. '.txt'))
	self.linguistic_input_fps = create_full_path(PATHS[dataset]['LINGUISTIC_INPUTS_PATH1'], self.recs1)
	self.acoustic_target_fps = create_full_path(PATHS[dataset]['ACOUSTIC_TARGETS_PATH1'], self.recs1)
	self.duration_target_fps = create_full_path(PATHS[dataset]['DURATION_TARGETS_PATH1'], self.recs1)

	if two_datasets then
		self.recs2 = lines_from(path.join(PATHS[dataset]['SPLIT_PATH2'], 'acoustic_split', split .. '.txt'))
		self.linguistic_input_fps = interleaf_tables(self.linguistic_input_fps,
			create_full_path(PATHS[dataset]['LINGUISTIC_INPUTS_PATH1'], self.recs2))
		self.acoustic_target_fps = interleaf_tables(self.acoustic_target_fps,
			create_full_path(PATHS[dataset]['ACOUSTIC_TARGETS_PATH1'], self.recs2))
		self.duration_target_fps = interleaf_tables(self.duration_target_fps,
			create_full_path(PATHS[dataset]['DURATION_TARGETS_PATH1'], self.recs2))
	end

	self.n = #self.linguistic_input_fps
end

function AcousticDataset:size()
	return self.n
end

function AcousticDataset:get(idx)
	-- For each phoneme in x, get its length in ms. Then stack frames of x together
	-- 50 ms long phoneme -> 10 frames of x back to back
	-- Also set the position of the current frame
	local x = load_hdf5_array(self.linguistic_input_fps[idx], 'x')
	
	-- Get number of frames for each phoneme
	-- Make sure we match the actual total number of frames. Handle the last phoneme speciallys
	local acoustic_target = load_hdf5_array(self.acoustic_target_fps[idx], 'y')

	local total_nframes = acoustic_target:size(1)

	local phoneme_durations = load_hdf5_array(self.duration_target_fps[idx], 'y')
	local phoneme_nframes = {}	-- Number of frames per phoneme, a frame being one time-step
	local plus_minus = 0		-- Keep track of difference. Will always be between -0.5 and 0.5
	for i=1,x:size(1) do 		-- 1 to number of phonemes in sequence
		local dur = phoneme_durations[i][1] / 0.005
		local dur_plus_minus = dur + plus_minus
		local n
		-- local tmp_old_plus_minus = plus_minus
		if i == x:size(1) then  -- last phoneme. Fill up.
			if i == 1 then 		-- only one phoneme
				table.insert(phoneme_nframes, total_nframes)
				break
			else
				n = total_nframes - table.reduce(phoneme_nframes, function(a,b) return a+b end)
			end
		else
			n = round(dur_plus_minus)
			plus_minus = dur_plus_minus - n
		end
		-- print(dur, tmp_old_plus_minus, dur_plus_minus, n)
		table.insert(phoneme_nframes, n)
	end

	-- Stack according to the number of frames per phoneme calculated above
	-- Split into second for loop here forc clarity
	local input = torch.Tensor(total_nframes, x:size(2))
	local cur_idx = 1
	for phone_idx,n in ipairs(phoneme_nframes) do
		for i=1,n do
			local cur_input_frame = x[phone_idx]:clone()
			-- Set bit for position of current frame
			cur_input_frame[98] = i

			input[cur_idx] = cur_input_frame
			cur_idx = cur_idx + 1
		end
	end

	local rec = self.linguistic_input_fps[idx]:match( "([^/]+)$"):gsub(".h5", "")

	return {
			input = input,
			target = load_hdf5_array(self.acoustic_target_fps[idx], 'y'),
			rec = rec,
		}
end