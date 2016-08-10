# Extract acoustic features from the relevant part of the wav file (i.e. ignoring pauses)
# Features: voiced, log F0, mel-cepstrum of spectrogram, mel-cepstrum of aperiodicity
using WAV
using WORLD
using HDF5

WAV_PATH = "data/cmu_us_clb_arctic/wav/"
PHONEME_LABELS_PATH = "data/cmu_us_clb_arctic/lab/"
# LINGUISTIC_INPUTS_PATH = "data/processed/cmu_us_clb_arctic/linguistic_inputs/"
LINGUISTIC_INPUTS_PATH = "data/processed/cmu_us_clb_arctic/linguistic_inputs_plus/"
ACOUSTIC_TARGETS_PATH = "data/processed/cmu_us_clb_arctic/acoustic_targets_f0interpolate/"
# ACOUSTIC_TARGETS_PATH = "data/processed/cmu_us_slt_arctic/acoustic_targets_zeromean"
# ACOUSTIC_TARGETS_PATH = "data/processed/cmu_us_slt_arctic/acoustic_targets_normalized"
FRAME_EVERY_MS = 5.0 		# extract frame every 5 ms

NORMALIZE = false			# zero-mean, unit-variance.
UTT_DUR_IDX = 253			# Index in linguistic input that stores length in seconds of utterance

function calc_acoustic_features(rec)
	# rec = "arctic_a0112"
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
	# println(start_time)
	start_frame = round(Int, floor(start_time / (FRAME_EVERY_MS / 1000.0)))
	# println(start_frame)

	### Get utterance duration from linguistic_inputs/*.h5 files
	# These were calculated in _save_phoneme_durations of feature_extractor.py
	# The duration is calculated as the start of the first non-PAU phoneme until the 
	# the end of the last non-PAU phoneme
	data = h5read(joinpath(LINGUISTIC_INPUTS_PATH, "$rec.h5"), "x")	# (98, #phonemes)
	# println(size(data))
	utt_dur = data[UTT_DUR_IDX][1]
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

	# println("f0")
	# println(log10(f0 + 1e-10)[100])
	# println(log10(f0 + 1e-10)[200])
	# println(log10(f0 + 1e-10)[300])
	# println("mc")
	# println(mc[1:end, 100])
	# println(mc[1:end, 200])
	# println(mc[1:end, 300])
	# println("mc_ap")
	# println(mc_ap[1:end, 100])
	# println(mc_ap[1:end, 200])
	# println(mc_ap[1:end, 300])
	# quit()

	# To reconstruct
	# approximate_spectrogram = mc2sp(mc, α, fftlen)
	# approximate_aperiodicity = mc2sp(mc_ap, α, fftlen)
	# wav_len =  round(Int, size(f0)[1] * fs * period / 1000)
	# y = synthesis(f0, approximate_spectrogram, approximate_aperiodicity, period, fs, wav_len)
	# wavwrite(y, "tmp.wav", Fs=fs)

	# 53680 / 16000 / (5/ 1000) = 671
	# println(size(x))				# (53680,)
	# println(fs)					# 16000
	# println(size(f0))				# (672,)
	# println(size(spectrogram))	# (513,672)
	# println(size(aperiodicity))
	# println(size(mc))
	# println(size(mc_ap))			# (41,672)
	# print(length(x))

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
		frame[3:3+(41-1)] = mc[:,i]
		frame[3+41:end] = mc_ap[:,i]
		features[i - start_frame + 1,:] = frame
	end

	# Linearly interpolate F0 - if unvoiced, take average of prev and next
	prev_voiced_f0 = Inf16
	next_voiced_f0 = Inf16
	for i=1:size(features)[1]
		if features[i,1] == 0				# Not voiced
			if prev_voiced_f0 == Inf16		# Get next voiced F0
				j = i
				future_voiced = features[j,1]
				while future_voiced == 0
					j += 1
					future_voiced = features[j,1]
				end
				f0 = features[j,2]
				features[i,2] = f0
				prev_voiced_f0 = f0			# This will help redundant lookaheads, e.g. start of seq is notvoiced, notvoiced, notvoiced...
			else
				if next_voiced_f0 == Inf16
					j = i
					future_voiced = features[j,1]
					while future_voiced == 0
						j += 1
						if j > size(features)[1]	# At end
							break
						end	
						future_voiced = features[j,1]
					end
					if j > size(features)[1]		# At end, just use previous
						next_voiced_f0 = prev_voiced_f0
					else
						next_voiced_f0 = features[j,2]
					end
				end
				features[i,2] = (prev_voiced_f0 + next_voiced_f0) * 0.5
			end
		else
			prev_voiced_f0 = features[i,2]
		end
	end
	println(size(features))	# (573. 84), i.e. (# time-steps, feature size)

	return features
end

# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
function get_mean_and_std()
	recs = readdir(WAV_PATH)

	n = 0
	mean = zeros(83)					# Normalize each feature, exlcuding voiced
	M2 = zeros(83)
	for rec_fn in recs
		rec = replace(rec_fn, ".wav", "")
		features = calc_acoustic_features(rec)

		cur_n = n
		for i=2:size(features)[2]				# Normalize each feature, excluding voiced
			cur_n = n
			for j=1:size(features)[1]			# For each time step
				cur_n += 1
				delta = features[j,i] - mean[i-1]
				mean[i-1] += delta / cur_n
				M2[i-1] += delta * (features[j,i] - mean[i-1])
			end
		end
		n = cur_n
		println(mean)
		# println(M2)
	end
	var = M2 / (n-1)
	stddev = sqrt(var)
	return mean, stddev
end

function save_acoustic_features()
	if NORMALIZE
		mean, stddev = get_mean_and_std()
		println(mean)
		println(stddev)
		h5write(joinpath(ACOUSTIC_TARGETS_PATH, "mean.h5"), "data", mean)
		h5write(joinpath(ACOUSTIC_TARGETS_PATH, "stddev.h5"), "data", stddev)
	end

	# mean = h5read(joinpath("data/processed/cmu_us_slt_arctic/acoustic_targets_normalized/", "mean.h5"), "data")               # (time, features (84))
	# stddev = h5read(joinpath("data/processed/cmu_us_slt_arctic/acoustic_targets_normalized/", "stddev.h5"), "data")

	recs = readdir(WAV_PATH)
	for rec_fn in recs
		rec = replace(rec_fn, ".wav", "")
		features = calc_acoustic_features(rec)
		if NORMALIZE
			for j=2:size(features)[2] 				# Each feature to be normalized
				features[1:end,j] -= mean[j-1]
				features[1:end,j] /= stddev[j-1]
			end
		end

		# However, it seems numpy is row-major while julia is column-major
		# In order to get (573, 84) when reading from numpy, must transpose
		features = transpose(features)
		h5write(joinpath(ACOUSTIC_TARGETS_PATH, "$rec.h5"), "y", features)
	end
end

# Main
save_acoustic_features()
