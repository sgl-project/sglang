# Beam SM103 decode GEMV: manual correctness and latency checks

`sglang.kernels.ops.moe.beam_decode_gemv` serves one BF16 token through
Beam's routed experts on SM103. `python/sglang/srt/models/beam.py` already
imports and calls it when both `SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED=1` and
`SGLANG_OPT_BEAM_DECODE_GEMV=1` are set and the layer has BF16 experts with MoE
TP4/EP1. These files are manual checks, not CI tests. Run them from the
repository root with this tree's `python/` importable, on one SM103 GPU,
correctness first:

```bash
python test/manual/kernels/moe/test_beam_decode_gemv.py
# fused paths with the config fused_experts_impl selects at run time
python test/manual/kernels/moe/beam_decode_gemv_probes/bench_decode_gemv.py
# fused paths with an explicit config
python test/manual/kernels/moe/beam_decode_gemv_probes/bench_decode_gemv.py \
    --fused-config '{"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4}'
```

## Shapes and numerics

Both checks use Beam's TP4 expert shard: 256 experts, hidden size 5120,
intermediate size 640, top-k 8, one contiguous BF16 token and FP32 routing
weights. `w13` is `[256, 1280, 5120]` with gate and up rows interleaved (row
`2j` is gate `j`, row `2j + 1` is up `j`); `w2` is `[256, 5120, 640]`. No
checkpoint, server or download is needed.

`../test_beam_decode_gemv.py` compares the kernel's routing-weighted
per-expert outputs with an FP64 reference that keeps the kernel's BF16 rounding
points (gate/up, activation, weighted down projection), at
`rtol=1.6e-2, atol=1e-3`. It covers int32 and int64 expert ids, each eagerly
and through five CUDA-graph replays whose inputs change between replays.

## Latency probe

`bench_decode_gemv.py` times three paths that return the same `[1, 5120]` BF16
sum over the 8 routed experts:

- `fused_moe`: the existing Triton `fused_experts_impl` with interleaved
  SwiGLU (`fuse_swiglu_interleaved=True`), `routed_scaling_factor=1.0` and its
  own expert combine;
- `fused_moe_defer`: `fused_experts_impl(defer_sum=True)`, the call Beam makes
  when the GEMV is off, followed by the same reducer as `decode_gemv`;
- `decode_gemv`: `beam_decode_gemv` followed by an explicit FP32 expert sum
  rounded to BF16.

`fused_moe_defer` and `decode_gemv` share their reducer. In the model neither
reducer runs, because Beam sums the experts in its fused all-reduce. No path
includes TP communication, so this is a synthetic operator measurement; model
accuracy and serving performance are not measured here. The paths are not
bitwise equal, so before timing the probe checks both fused sums against the
GEMV sum at the correctness test's tolerance.

Without `--fused-config`, both fused paths use the config `fused_experts_impl`
selects for this shape at one token: the GPU's tuned config file when one
exists, otherwise its built-in default. B300 has no tuned file for E=256,
N=640, so a comparison against that default cannot by itself establish a win
over a tuned fused baseline. `--fused-config` applies one config to both fused
GEMMs through `override_config`, as the fused-MoE tuning script does; the
config above was the fastest of a bounded sweep on B300, not a full tuning run.
The probe prints the effective config with the GPU, SM count, torch, CUDA,
driver and Triton versions.

Protocol:

- each path is warmed up and captured in its own CUDA graph before timing;
- every timed replay follows an untimed write of at least 5x the device's L2
  size, because the 8 routed experts' weights (about 157 MB) are comparable to
  L2 and would otherwise stay hot between replays;
- each run is a fixed number of single replays (`--repeats`, default 20), each
  timed with its own pair of CUDA events; the probe prints every sample with
  the run's mean and median;
- runs follow the palindromic order `fused_moe, fused_moe_defer, decode_gemv,
  decode_gemv, fused_moe_defer, fused_moe` in at least 5 blocks (`--blocks`,
  default 6). The summary gives the median and range, over blocks, of each
  path's mean and of the GEMV's difference from each fused path.

On B300 the individual samples fell into clusters about 2 us apart, while
values within the data still differed in steps as small as 0.032 us. A run's
median can therefore move by a whole cluster, so compare means and their
ranges, and keep the raw samples.
