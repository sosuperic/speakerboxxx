# Extract acoustic features from the relevant part of the wav file (i.e. ignoring pauses)
# Features: voiced, log F0, mel-cepstrum of spectrogram, mel-cepstrum of aperiodicity
using WAV
using WORLD
using HDF5

WAV_PATH = "data/cmu_us_slt_arctic/wav/"
PHONEME_LABELS_PATH = "data/cmu_us_slt_arctic/lab/"
LINGUISTIC_INPUTS_PATH = "data/processed/cmu_us_slt_arctic/linguistic_inputs/"
ACOUSTIC_TARGETS_PATH = "data/processed/cmu_us_slt_arctic/acoustic_targets/"
FRAME_EVERY_MS = 5.0 		# extract frame every 5 ms


function save_acoustic_features(rec)
	println(rec)
	########################################################################################
	### Get the start time and utterance duration in order to cut out the relevant
	### section in the audio, i.e. ignore the pauses
	########################################################################################

	### Get start time of first phoneme from .lab files
	f = open(joinpath(PHONEME_LABELS_PATH, "$rec.lab"))
	lines = readlines(f)
	start_time = 0
	for line in lines
		if startswith(line, "#")
			continue
		end
		splitted = split(line, " ")
		if strip(splitted[3]) != "pau"
			start_time = float(splitted[1])
			break
		end
	end
	close(f)
	println(start_time)
	start_frame = round(Int, floor(start_time / (FRAME_EVERY_MS / 1000.0)))
	println(start_frame)

	### Get utterance duration from linguistic_inputs/*.h5 files
	# These were calculated in _save_phoneme_durations of feature_extractor.py
	# The duration is calculated as the start of the first non-PAU phoneme until the 
	# the end of the last non-PAU phoneme
	data = h5read(joinpath(LINGUISTIC_INPUTS_PATH, "$rec.h5"), "x")	# (98, #phonemes)
	# println(size(data))
	utt_dur = data[97][1]
	println(utt_dur)
	nframes = round(Int, ceil(utt_dur / (FRAME_EVERY_MS / 1000.0)))
	println(nframes)


	########################################################################################
	# Read wav files and extract spectral features
	########################################################################################
	# Reading a speech signal
	filepath = joinpath(WAV_PATH, "$rec.wav")
	x, fs = wavread(filepath)
	x = vec(x) # monoral
	fs = convert(Int, fs)

	# Fundamental frequency (f0) estimation by DIO
	period = FRAME_EVERY_MS
	opt = DioOption(f0floor=71.0, f0ceil=800.0, channels_in_octave=2.0, period=period, speed=1)
	f0, timeaxis = dio(x, fs, opt)

	f0_by_dio = copy(f0)

	# F0 refinement by StoneMask
	f0 = stonemask(x, fs, timeaxis, f0_by_dio)

	# Spectral envelope estimatino by CheapTrick
	spectrogram = cheaptrick(x, fs, timeaxis, f0)

	# Aperiodicity ratio estimation
	aperiodicity = d4c(x, fs, timeaxis, f0)

	# Feature dimention reduces to 41 from 513!
	α = 0.41
	fftlen = get_fftsize_for_cheaptrick(fs)
	order = 40
	mc = sp2mc(spectrogram + 1e-10, order, α)			# Need eps otherwise NaNs
	mc_ap = sp2mc(aperiodicity + 1e-10, order, α)

	# To reconstruct
	# approximate_spectrogram = mc2sp(mc, α, fftlen)
	# approximate_aperiodicity = mc2sp(mc_ap, α, fftlen)

	# println(size(x))				# (53680,)
	# println(fs)						# 16000
	# println(size(f0))				# (672,)
	# println(size(spectrogram))		# (513,672)
	# println(size(aperiodicity))
	# println(size(mc))
	# println(size(mc_ap))			# (41,672)


	########################################################################################
	# Construct features and save
	########################################################################################

	# num_frames = size(f0)[1]
	feature_size = 1 + 1 + 41 + 41  # voiced, log f0, sp, ap
	features = zeros(nframes, feature_size)
	end_frame = start_frame + nframes - 1
	for i=start_frame:end_frame				# For each time-step (5ms sample)
		frame = zeros(1, feature_size)
		voiced = 1
		if abs(f0[i]) < 1e-6
			voiced = 0
		end
		frame[1] = voiced
		frame[2] = log10(f0[i] + 1e-10)		# Some are 0 so add eps
		frame[3:3+(41-1)] = mc[:,1]
		frame[3+41:end] = mc_ap[:,1]
		features[i - start_frame + 1,:] = frame
	end
	println(size(features))	# (573. 84), i.e. (# time-steps, feature size)

	# However, it seems numpy and julia arrays are transposed.
	# In order to get (573, 84) when reading from numpy, must transpose

	h5write(joinpath(ACOUSTIC_TARGETS_PATH, "$rec.h5"), "y", transpose(features))
end


recs = readdir(WAV_PATH)
for rec_fn in recs
	rec = replace(rec_fn, ".wav", "")
	save_acoustic_features(rec)
end
