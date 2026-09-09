# NEO-Unify GPU attention

The feature includes SenseNova support from SGLang PR #36606, head
`d9a2188f52027485c73b10ab578cd3453893f400`. Its model entry point currently
serves T2I; the attention adapter also covers the model's reference-image
prefill helper. This does not add an image-editing serving endpoint.

## Backends and semantics

`neo_prefill_backend` and `neo_denoise_backend` are independent entries in
`--attention-backend-config`. Values: `auto`, `legacy`, `torch`, `triton`,
`fa3`. `legacy` preserves the PR's original implementation; `torch` is a
dense FP32 oracle and is intended for small correctness cases. Non-CUDA
`auto` retains the original model path.

On CUDA, `auto` selects the image-aware FA3 build on Hopper when available,
otherwise Triton. Denoising uses SGLang's ordinary FA3 on Hopper and Triton
elsewhere. Explicit `fa3` never silently falls back. Unsupported optimized
dtypes/head dimensions use the Torch path only in `auto` mode.

Mixed prefill is causal except within each contiguous image block. An image
query cannot attend to later text or another later image. Metadata consists
of exclusive logical KV end positions (`image_token_end`); rotary positions
and `image_gen_indicators` are not interchangeable with these positions.
Metadata is built once per prefix. Denoising is fully non-causal over
`[prefix KV, current image KV]` and retains the existing cache writes.

The initial interface accepts unpadded `[B, S, H, D]` batches, FP16/BF16
optimized inputs, GQA, head dimensions 32/64/128/256, and different Q/KV
lengths. It does not implement ragged batches, sliding windows, or chunking
through the middle of a cached bidirectional image block. The base PR's
generation stage dispatches one request at a time.

## FA3 dependency

Ordinary SGLang FA3 is sufficient for denoising and pure-text prefill.
Image-aware FA3 additionally requires the upstream-linked fork:

- https://github.com/WANDY666/flash-attention/tree/support_neo
- Reference commit: `e2077ee6e568e64d0d01c6b44d8ce4ee24e7932b`
- Required Python entry point: `flash_attn_interface.flash_attn_with_kvcache`
  with `image_token_end` (not the older boolean `image_token_tag`).

Use that fork's Hopper build instructions in a separate Linux CUDA
environment. Both its Python interface and compiled extension must come
from the same revision. The initial FA3 dispatch is limited to SM90;
Ampere/Blackwell use Triton. Importing the model does not import this optional
extension. The actual CUDA build, numeric checks and performance must be
validated on the target GPU before treating this implementation as qualified.

## Correctness

From the repository root, with SGLang installed in the environment:

```bash
python test/registered/unit/attention/test_neo_unify.py
python test/registered/unit/attention/test_sensenova_attention.py
python test/registered/kernel/attention/test_neo_unify.py
```

The GPU suite compares each path to FP32 attention, including tile-crossing
image spans, GQA, strided tensors and unequal Q/KV lengths. Optional FA3
cases skip when its build or Hopper is unavailable; that skip is not FA3
validation. The model tests compare prefix outputs/cache and repeated
denoising steps against the original implementation using a tiny randomly
initialized attention layer, without downloading a checkpoint.

## Operator benchmark

```bash
python benchmark/sensenova/bench_attention.py --mode prefill \
  --backends eager sdpa triton fa3 --query-length 1024 --prefix-length 128 \
  --output prefill.json
python benchmark/sensenova/bench_attention.py --mode denoise \
  --backends sdpa triton fa3 --query-length 4096 --prefix-length 128 \
  --output denoise.json
```

Run correctness first. Repeat with actual checkpoint head counts/dimensions,
batch sizes and image token lengths. Dense baselines may OOM at long lengths;
run them separately instead of reducing only their sequence lengths. Timings
include adapter/layout work; JIT compilation is excluded by warmup. Both the
baseline mask and optimized metadata are prepared outside the timed loop,
as in the model. These are operator timings, not end-to-end speedups.

## Model A/B runs

Use the same checkpoint, GPU, precision, prompt, seed, resolution and steps:

```bash
sglang generate --model-path sensenova/SenseNova-U1.5-8B-MoT \
  --prompt "A mountain lake at sunrise" --width 1024 --height 1024 \
  --num-inference-steps 50 --guidance-scale 4 --seed 42 \
  --attention-backend-config '{"neo_prefill_backend":"legacy","neo_denoise_backend":"legacy"}' \
  --output-file-path legacy.png
```

Repeat for `triton/legacy`, `legacy/triton`, and `triton/triton`; then FA3
combinations on Hopper. Compare output images and denoising trajectories,
not bitwise image equality across attention implementations. Reuse
`python -m sglang.multimodal_gen.benchmarks.bench_offline_throughput` and
`bench_serving` for end-to-end measurements. Record checkpoint/revisions,
software versions, prefill time, denoising time, latency and peak memory.
No end-to-end speedup or image-quality claim is made without those runs.
