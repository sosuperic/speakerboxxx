-- Implement dataset interface for torchnet
-- Load pre-processed split and sort by sequence length

require 'utils.lua_utils'
require 'hdf5'
require 'pl'
local tnt = require 'torchnet'

-- local LINGUISTIC_INPUTS_PATH = 'data/processed/cmu_us_slt_arctic/linguistic_inputs/'
local LINGUISTIC_INPUTS_PATH = 'data/processed/cmu_us_slt_arctic/linguistic_inputs_plus/'
-- local ACOUSTIC_TARGETS_PATH = 'data/processed/cmu_us_slt_arctic/acoustic_targets_normalized/'
-- local ACOUSTIC_TARGETS_PATH = 'data/processed/cmu_us_slt_arctic/acoustic_targets_zeromean/'
-- local ACOUSTIC_TARGETS_PATH = 'data/processed/cmu_us_slt_arctic/acoustic_targets/'
local ACOUSTIC_TARGETS_PATH = 'data/processed/cmu_us_slt_arctic/acoustic_targets_f0interpolate/'
local DURATION_TARGETS_PATH = 'data/processed/cmu_us_slt_arctic/duration_targets/'
local SPLIT_PATH = 'data/processed/'

----------------------------------------------------------------------------------------------------------------
-- Helper Functions
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
	return data
end

----------------------------------------------------------------------------------------------------------------
-- Duration model dataset loader
----------------------------------------------------------------------------------------------------------------
local DurationDataset, Dataset = torch.class('tnt.DurationDataset', 'tnt.Dataset', tnt)

function DurationDataset:__init(split)
	self.recs = lines_from(path.join(SPLIT_PATH, 'duration', split .. '.txt'))
	self.linguistic_input_fps = create_full_path(LINGUISTIC_INPUTS_PATH, self.recs)
	self.duration_target_fps = create_full_path(DURATION_TARGETS_PATH, self.recs)
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

function AcousticDataset:__init(split)
	self.recs = lines_from(path.join(SPLIT_PATH, 'acoustic', split .. '.txt'))
	self.linguistic_input_fps = create_full_path(LINGUISTIC_INPUTS_PATH, self.recs)
	self.acoustic_target_fps = create_full_path(ACOUSTIC_TARGETS_PATH, self.recs)
	self.duration_target_fps = create_full_path(DURATION_TARGETS_PATH, self.recs)
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
		if i == x:size(1) then  -- Last phoneme. Fill up.
			n = total_nframes - table.reduce(phoneme_nframes, function(a,b) return a+b end)
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