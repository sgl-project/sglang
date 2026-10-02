#!/bin/bash

card=7
# ppudbg --dpmop set 1 1 10 1 --device $card
# ppudbg --dpmop set 3 1 12 1 --device $card
# ppudbg --dpmop set 1 1 11 17000 --device $card
export CUDA_VISIBLE_DEVICES=$card


# ln -snf /workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_valu_moe.py /workspace/sglang_python/srt/layers/moe/fused_moe_triton/fused_valu_moe.py
# ln -snf /workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py /workspace/sglang_python/srt/layers/moe/fused_moe_triton/fused_moe.py
# export FUSEDMOE_USE_AIU=0
# export TRITON_INTERPRET=1
# export TRITON_DEBUG=1

# export FUSEDMOE_BLOCK_SIZE_M=16
# export FUSEDMOE_DIVERSE_RATIO=0.1

# export CUDA_LAUNCH_BLOCKING=1
# export USE_ACEXT_CUDA=1
export FUSEDMOE_OPT=1
# optimal_moe, reference_fused_moe, reference_acext_impl, actual_fused_moe
func_opts="--func actual_fused_moe"
# base_opts="--num-tokens 192 --num-experts 160 --topk 6 --immediate-size 208 --hidden-size 5120 --dtype bfloat16"
# base_opts="--num-tokens 200 --num-experts 160 --topk 6 --immediate-size 208 --hidden-size 5120 --dtype float16"
base_opts="--num-tokens 1024 --num-experts 128 --topk 8 --immediate-size 192 --hidden-size 4096 --dtype bfloat16"
imbalance_opts="--absent-ratio 0 --temperature 0.05 --seed 2025"
metrics_opts="--perf-iters 10" #  --save-imbalance-plot
accuracy_opts="--accuracy --rtol 1e-2 --atol 1e-2"
# PROFILE="asys profile -o imbalance_test -f true -c hgtx -p main-test"
# PROFILE="acu --graph-profiling node --kernel-name regex:moe_persistence -f --set=full --launch-skip 8 --launch-count 4 -o qwen3-bs96-a2"
# PROFILE="hggc-memcheck --tool memcheck --destroy-on-device-error kernel"
# DEBUG="-m pdb"

${PROFILE} python3 ${DEBUG} benchmark_imbalance_experts.py ${func_opts} ${base_opts} ${imbalance_opts} ${metrics_opts} ${metrics_opts} ${accuracy_opts}
