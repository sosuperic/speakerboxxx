# Extract acoustic features from the relevant part of the wav file (i.e. ignoring pauses)
# Features: voiced, log F0, mel-cepstrum of spectrogram, mel-cepstrum of aperiodicity
# Outputs: (573, 84), where 573 is number of time steps, sampled at every 5 ms; 84 are the features
using WAV
using WORLD
using HDF5

WAVS_PATH = "../Prosodylab-Aligner/data/blizzard2013/train/segmented/wavn"
ACOUSTIC_TARGETS_PATH = "data/processed/blizzard2013/acoustic_targets"
FRAME_EVERY_MS = 5.0           # extract frame every 5 ms

# Slice out valid window (excluding silences)
# - start_time in seconds, up to hundreth place
# - end_time in seconds, up to hundreth place
# To calculate start_sample_idx (or end_time):
# 1) Get length of wav in seconds -> len_in_time_x
# 2) Use the following relationship:
# - (end_time / len_in_time_x) = (end_sample_idx / n_samples)
function extract_wav_window(wav_filepath, start_time, end_time)
    println("Extracting wav window")
    x, fs = wavread(wav_filepath)
    x = vec(x)  # monoral           # (53680, )
    fs = convert(Int, fs)           # sampling rate: 16000 samples per second

    n_samples = length(x)
    len_in_time_x = n_samples / fs
    start_sample_idx =  round(Int, n_samples * (start_time / len_in_time_x))
    end_sample_idx = round(Int, n_samples * (end_time / len_in_time_x))
    x = x[start_sample_idx:end_sample_idx]

    return x, fs
end

# Read wav file, extract f0, mel-cepstrum coefficients for spectral envelope and aperiodicity
function extract_raw_acoustic_features(x, fs)
    println("Extracting raw acoustic features")
    # Fundamental frequency (f0) estimation by DIO
    period = FRAME_EVERY_MS
    opt = DioOption(f0floor=71.0, f0ceil=800.0, channels_in_octave=2.0, period=period, speed=1)
    f0, timeaxis = dio(x, fs, opt)

    # f0 refinement by Stonemask
    f0_by_dio = copy(f0)
    f0 = stonemask(x, fs, timeaxis, f0_by_dio)          # (672, )

    # Spectral envelope estimation by CheapTrick
    spectrogram = cheaptrick(x, fs, timeaxis, f0)       # (513, 672)

    # Aperiodicity ratio estimation
    aperiodicity = d4c(x, fs, timeaxis, f0)             # (513, 672)

    # Feature dimension reduces from 513 to 41
    # Note: this is why we are using the Julia wrapper and not the Python wrapper
    # i.e. the existence of sp2mc
    alpha = 0.1
    fftlen = get_fftsize_for_cheaptrick(fs)
    order = 40
    mc = sp2mc(spectrogram + 1e-10, order, alpha)       # need eps otherwise NaNs; (41, 672)
    mc_ap = sp2mc(aperiodicity + 1e-10, order, alpha)   # (41, 672)

    # Transpose just so it matches f0, where first dimension is a time step
    mc = transpose(mc)                                  # (41, 672) -> (672, 41)
    mc_ap = transpose(mc_ap)                            # (41, 672) -> (672, 41)      

    return f0, mc, mc_ap
end

# Convert raw features into features ready for model
# - Add voiced flag
# - Take log of f0
# - Extract valid window (excluding silences)
function process_raw_features(f0, mc, mc_ap)
    println("Processing raw features")
    feature_size = 1 + 1 + 41 + 41                      # voiced, log f0, sp, ap
    nframes = size(f0)[1]
    features = zeros(nframes, feature_size)
    for i=1:nframes
        frame = zeros(1, feature_size)
        voiced = abs(f0[i]) < 1e-6 ? 0 : 1
        frame[1] = voiced 
        frame[2] = log10(f0[i] + 1e-10)                 # some are 0 so add eps
        frame[3:3+(41-1)] = mc[i,:]
        frame[3+41:end] = mc_ap[i,:]
        features[i,:] = frame
    end
    return features
end

# Linearly interpolate f0 when frame is silent
# If unvoiced, take average of prev and next
function lin_interp_f0(features)
    println("Linearly interpolating f0")
    prev_voiced_f0 = Inf16
    next_voiced_f0 = Inf16
    for i=1:size(features)[1]
        if features[i,1] == 0               # not voiced
            if prev_voiced_f0 == Inf16      # get next voiced F0
                j = i
                future_voiced = features[j,1]
                while future_voiced == 0
                    j += 1
                    future_voiced = features[j,1]
                end
                f0 = features[j,2]
                features[i,2] = f0
                prev_voiced_f0 = f0         # this will help redundant lookaheads, e.g. start of seq is notvoiced, notvoiced, notvoiced...
            else
                if next_voiced_f0 == Inf16
                    j = i
                    future_voiced = features[j,1]
                    while future_voiced == 0
                        j += 1
                        if j > size(features)[1]    # at end
                            break
                        end 
                        future_voiced = features[j,1]
                    end
                    if j > size(features)[1]        # at end, just use previous
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

    return features
end

# Main function that uses the above functions and actually saves features
function extract_and_save_acoustic_features()
    # Read windows of valid audo for which to extract audio features
    f = open("recording_windows_for_audio_features.txt")
    lines = readlines(f)
    close(f)

    i = 0
    for line in lines
        splitted = split(line, ",")

        # Read relevant info from file
        id = splitted[1]
        print(line)
        # println(i, id)
        i += 1
        out_path = joinpath(ACOUSTIC_TARGETS_PATH, "$id.h5")
        if isfile(out_path)
            continue
        end
        start_time, end_time = float(splitted[2]), float(strip(splitted[3]))

        # Slice out valid window (excluding silences) using info
        wav_filepath = joinpath(WAVS_PATH, "$id.wav")
        x, fs = extract_wav_window(wav_filepath, start_time, end_time)

        # Get raw features using WORLD
        f0, mc, mc_ap = extract_raw_acoustic_features(x, fs)

        # Process raw features 
        features = process_raw_features(f0, mc, mc_ap)

        # # Linearly interpolate f0
        features = lin_interp_f0(features)
    
        # # Transpose because it seems numpy is row-major while julia is column-major
        # # In order to get (573, 84) when reading from numpy, must transpose
        features = transpose(features)
        println(size(features))
        h5write(out_path, "y", features)
    end
end

# Main
extract_and_save_acoustic_features()
