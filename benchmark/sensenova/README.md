# NEO-Unify GPU attention

The model entry point currently serves T2I; the attention adapter also covers
the model's reference-image prefill helper. This does not add an image-editing
serving endpoint.

## Backends and semantics

`neo_prefill_backend` and `neo_denoise_backend` are independent entries in
`--attention-backend-config`. Values: `auto`, `legacy`, `torch`, `triton`,
`fa3`. `legacy` preserves the PR's original implementation; `torch` is a
dense FP32 oracle and is intended for small correctness cases. Non-CUDA
`auto` retains the original model path.

On CUDA, prefill `auto` selects the image-aware FA3 build on Hopper when
available, otherwise Triton. The pinned `support_neo` build also contains
SM80-SM89 kernels; use explicit `fa3` to validate those GPUs until target-device
performance data justifies automatic selection. Denoising `auto` uses SGLang's
ordinary FA3 on Hopper and preserves the existing FlashAttention/SDPA path on
other GPU architectures. Explicit `triton` remains available for denoising
experiments, and explicit `fa3` never silently falls back.

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

SGLang's built-in FA3 handles denoising on SM90. Image-aware FA3, and explicit
FA3 selection on SM80-SM89 in this adapter, require the upstream-linked fork:

- https://github.com/WANDY666/flash-attention/tree/support_neo
- Reference commit: `e2077ee6e568e64d0d01c6b44d8ce4ee24e7932b`
- Required Python entry point: `flash_attn_interface.flash_attn_with_kvcache`
  with `image_token_end` (not the older boolean `image_token_tag`).

Use that fork's build instructions in a separate Linux CUDA environment. Both
its Python interface and compiled extension must come from the same revision.
With PyTorch 2.9 or newer, wrap the `std::array<int64_t, 1>{total_q}` shape
expression at `hopper/flash_api_stable.cpp:1220` outside `STD_TORCH_CHECK` to
prevent its template comma from being parsed as a macro argument.
The fork contains image-aware forward kernels for SM80-SM90. `auto` remains
limited to SM90; explicit `fa3` enables target-device validation on Ampere and
Ada. Importing the model does not import this optional extension. The actual
CUDA build, numeric checks and performance must be validated on the target GPU
before treating this implementation as qualified.

## Correctness

From the repository root, with SGLang installed in the environment:

```bash
python test/registered/unit/attention/test_neo_unify.py
python test/registered/unit/attention/test_sensenova_attention.py
python test/registered/kernel/attention/test_neo_unify.py
```

The GPU suite compares each path to FP32 attention, including tile-crossing
image spans, GQA, strided tensors and unequal Q/KV lengths. Optional FA3
cases skip when the extension is unavailable or the GPU is outside SM80-SM90;
that skip is not FA3 validation. The model tests compare prefix outputs/cache and repeated
denoising steps against the original implementation using a tiny randomly
initialized attention layer, without downloading a checkpoint.

## Operator benchmark

```bash
python benchmark/sensenova/bench_attention.py --mode prefill \
  --backends legacy sdpa triton --query-length 1024 --prefix-length 128 \
  --output prefill.json
python benchmark/sensenova/bench_attention.py --mode denoise \
  --backends legacy sdpa triton --query-length 4096 --prefix-length 128 \
  --output denoise.json
```

Run correctness first. Repeat with actual checkpoint head counts/dimensions,
batch sizes and image token lengths. Dense baselines may OOM at long lengths;
run them separately instead of reducing only their sequence lengths. Timings
include adapter/layout work; JIT compilation is excluded by warmup. Both the
baseline mask and optimized metadata are prepared outside the timed loop,
as in the model. `legacy` reports the implementation it resolved to: eager for
image-aware prefill, and FlashAttention when installed or SDPA otherwise for
denoising. Add `fa3` after installing the pinned `support_neo` build on
SM80-SM90; ordinary SGLang FA3 is also available for denoising on Hopper. These
are operator timings, not end-to-end speedups.

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
combinations on SM80-SM90 with the optional extension. Compare output images
and denoising trajectories, not bitwise image equality across attention
implementations. Reuse
`python -m sglang.multimodal_gen.benchmarks.bench_offline_throughput` and
`bench_serving` for end-to-end measurements. Record checkpoint/revisions,
software versions, prefill time, denoising time, latency and peak memory.
No end-to-end speedup or image-quality claim is made without those runs.
