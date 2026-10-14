"""The DeepSeek-V4.1 gfx950 monokernels against SGLang's ops, one script a launch
(``torchrun --nproc-per-node {2,4} -m sglang.test.kernels.mono.deepseek_v41_hip.check_ffn``):
per-output numerics against the spec's bounds, bitwise graph replay across
widths, and every rank agreeing. ``test/registered/kernels/ops/mono`` runs them in CI."""
