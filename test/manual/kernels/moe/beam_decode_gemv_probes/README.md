# Beam SM103 decode GEMV: manual correctness and latency checks

`sglang.kernels.ops.moe.beam_decode_gemv` serves one BF16 token through
Beam's routed experts on SM103 (GB300). These files are manual checks, not CI
tests. Run them from the repository root with this tree's `python/` importable,
on one SM103 GPU, correctness first:

```bash
python test/manual/kernels/moe/test_beam_decode_gemv.py
python test/manual/kernels/moe/beam_decode_gemv_probes/bench_decode_gemv.py
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
`rtol=1.6e-2, atol=1e-3`. It covers int32 and int64 expert ids eagerly and five
CUDA-graph replays whose inputs change between replays.

## Latency probe

`bench_decode_gemv.py` times two paths that return the same `[1, 5120]` BF16
sum over the 8 routed experts:

- `fused_moe`: the existing Triton `fused_experts_impl` with interleaved
  SwiGLU (`fuse_swiglu_interleaved=True`), `routed_scaling_factor=1.0` and its
  own expert combine;
- `decode_gemv`: `beam_decode_gemv` followed by an explicit FP32 expert sum
  rounded to BF16.

Both include the expert reduction. Neither includes TP communication or the
fused all-reduce boundary, so this is not a measurement of the model's
deferred-sum integration.

Protocol:

- each path is warmed up and captured in its own CUDA graph before timing;
- every timed replay follows an untimed write of at least 5x the device's L2
  size, because the 8 routed experts' weights (about 157 MB) are comparable to
  L2 and would otherwise stay hot between replays;
- each run is the median of a fixed number of single replays timed with CUDA
  events (`--repeats`, default 20);
- runs alternate `fused_moe, decode_gemv, decode_gemv, fused_moe` in at least
  5 blocks (`--blocks`, default 6). Every block's pair, the medians and the
  median delta are printed with the GPU, SM count, torch and Triton versions.

The probe prints the maximum absolute difference between the two summed
outputs for information only; correctness is the test above.

## Status

Both scripts are checked statically. The correctness test passed on one GB300.
The latency probe has not been run, so no speedup is claimed. The opt-in model
dispatch for this kernel lives in the Beam model part and moves with it when
that part is restacked.
