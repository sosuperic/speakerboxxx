# Create wavs from the parameters generated by test.lua
# 1st command line argument is dataset (cmuarctic, cmuarctic_two_datasets, blizzard2013)
# 2nd command line argument is folder name set of spectral params (acoustic_only, full_pipeline, full_pipeline_f0interpolate)
# 3rd command line argument is folder name to save wavs in

using WAV
using WORLD
using HDF5

NORMALIZED = false
SILENCE_IF_NOTVOICED = false

########################################################################################################################
########################################################################################################################
function generate_and_save_wavs(spectral_params_path, outputs_wavs_path)
    # Make output directory to save wavs in
    if !ispath(outputs_wavs_path)
        mkdir(outputs_wavs_path)
    end

    if NORMALIZED
        mean = h5read(joinpath("data/processed/cmu_us_slt_arctic/acoustic_targets_f0interpolate_normalized/", "mean.h5"), "data")               # (time, features (84))
        stddev = h5read(joinpath("data/processed/cmu_us_slt_arctic/acoustic_targets_f0interpolate_normalized/", "stddev.h5"), "data")
    end

    recs = readdir(spectral_params_path)
    for rec_fn in recs
        if rec_fn == ".DS_Store"
            continue
        end

        # Get prompt id and data
        rec = splitext(rec_fn)[1]
        println(rec)
        fp = joinpath(spectral_params_path, rec_fn)
        data = h5read(fp, "data")               
        data = transpose(data)                  # (time, features (84))

        # Restore mean stddev if data was normalized originally
        if NORMALIZED
            for i=2:size(data)[2]                  # features
                data[1:end, i] *= stddev[i-1]
                data[1:end, i] += mean[i-1]
            end
        end

        # Get voiced and f0 values
        voiced = data[1:end,1]
        log_f0  = data[1:end,2]
        f0 = 10.^(log_f0 - 1e-10)               # (381,)        NOTE the eps
        
        # Silence if voiced flag is 1
        if SILENCE_IF_NOTVOICED
            for i=1:size(voiced)[1]
                if voiced[i] < 0.25
                    f0[i] = 0.0
                end
            end
        end

        # Get cepstrum coefficients
        sp_mc = data[1:end, 3:3+(41-1)]         # (381, 41)
        ap_mc = data[1:end, 3+41:end]           # (381, 41)
        sp_mc = transpose(sp_mc)
        ap_mc = transpose(ap_mc)

        # Get full parameters from cepstrum coefficents
        α = 0.41
        fs = 16000
        period = 5.0
        fftlen = get_fftsize_for_cheaptrick(fs)
        wav_len = round(Int, size(f0)[1] * fs * period / 1000)
        approx_sp = mc2sp(sp_mc, α, fftlen)
        approx_ap = mc2sp(ap_mc, α, fftlen)

        # Read targets to compare - see how generated parameters compare against actual target parameters
        # targets = h5read(joinpath("data/processed/cmu_us_slt_arctic/acoustic_targets/", "$rec.h5"), "y")
        # targets_f0 = targets[2,1:end]
        # targets_sp_mc = targets[3:3+(41-1), 1:end]
        # targets_ap_mc = targets[3+(41-1):end, 1:end]
        # writedlm("log_f0.txt", log_f0)
        # writedlm("targets_f0.txt", squeeze(targets_f0, 1))
        # imshow(10log10(targets_sp_mc + 1e-6), origin="lower", aspect="auto")
        # colorbar()
        # quit()

        # timeaxis = Array(linspace(0, length(f0) * 0.005, length(f0)))
        # save(f0, "test_f0.png")
        # # imwrite(timeaxis, f0)
        # quit()

        # Generate and save
        y = synthesis(f0, approx_sp, approx_ap, period, fs, wav_len)
        wavwrite(y, joinpath(outputs_wavs_path, "$rec.wav"), Fs=fs)
    end
end

########################################################################################################################
# Main
# Example: julia WavGenerator.jl cmuarctic full_pipeline_f0interpolate wav_tmp
spectral_params_path = joinpath("outputs", ARGS[1], "spectral", ARGS[2])
outputs_wavs_path = joinpath("outputs", ARGS[1], ARGS[3])
generate_and_save_wavs(spectral_params_path, outputs_wavs_path)
