 #!/bin/bash
th main.lua -mode test -load_duration_model_path models/duration/2016_7_20___5_16_19/net_e9.t7 -load_acoustic_model_path models/acoustic/2016_7_20___14_30_32/net_e1.t7 -batchsize 32
julia generate_test_wavs.jl