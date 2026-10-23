# Nomas development validation

Generate a fixed, small random pipeline for protocol and condition-sensitivity
tests (Linux, NVIDIA CUDA, SGLang diffusion dependencies):

```bash
python python/sglang/multimodal_gen/test/scripts/create_nomas_fixture.py \
  --output /path/to/fixed-random-pipeline
NOMAS_RANDOM_PIPELINE=/path/to/fixed-random-pipeline \
python -m pytest -q python/sglang/multimodal_gen/test/unit/test_nomas_pipeline_smoke.py
```

The fixture uses the real component implementations with reduced dimensions.
It does not produce meaningful video. The public-API regression checks repeated
seeds, changed text/camera conditions and non-aligned event frame counts. The
latent/session checks live in `test_nomas.py` and `test_nomas_reference.py`.

The default diffusion unit suite collects these tests. Tests requiring a
fixture, upstream source or additional GPUs skip when those are unavailable;
real-weight HTTP/WebSocket checks below were manual H200 runs, not CI results.

## Validation scope

The existing H200 validation uses PyTorch 2.14.1+cu130, Transformers 5.19.0,
Diffusers 0.37.0, FlashAttention 2.8.3.post2 built against that environment,
BF16 weights and Flex attention. The five Nomas unit suites
pass 76 tests; two TP=2 × SP=2 tests require four GPUs and are skipped on a pair.

With a fixed random pipeline fixture and the pinned upstream source checkout
(`98a28976f23552bb2e2cf7941986e9cd10a3118a`), run from the repository root:

```bash
NOMAS_RANDOM_PIPELINE=/path/to/fixed-random-pipeline \
NOMAS_REFERENCE_ROOT=/path/to/Hy-Nomas-source NOMAS_TEST_FULL=1 \
python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_nomas.py \
  python/sglang/multimodal_gen/test/unit/test_nomas_pipeline.py \
  python/sglang/multimodal_gen/test/unit/test_nomas_realtime.py \
  python/sglang/multimodal_gen/test/unit/test_nomas_reference.py \
  python/sglang/multimodal_gen/test/unit/test_nomas_parallel.py
```

With full trained weights and two 120-frame events at 704×1280, the second batch
of optimizations reduces the warmed DiT median from 33.406 to 32.533 seconds
(four samples each, alternating groups). Latents, all active K/V entries and
RNG states match the preceding SGLang implementation exactly. A public offline
API video also matches its baseline after MP4 decoding.

The all-resident component harness measures 55.084 → 54.005 seconds for the
complete single-GPU request, including output conversion and MP4 saving. Its
experimental two-GPU component placement measures 53.794 seconds, with peak
allocated memory of 40.77/33.00 GiB instead of 68.03 GiB on one GPU. These are
two-sample medians using a custom placement harness, not online throughput or
public EPD support. The dynamic-shape RoPE implementation improves decoder time
by about 1.33% in that harness; the 7.85% static-shape prototype result does not
describe this implementation.

Compilation and first-use overhead require separate accounting: one default
offload API first request takes 68.56 seconds excluding model loading. The
warmed numbers above do not establish a default-configuration first-request
speedup. Full trained-weight numerical parity with the upstream application
remains unresolved; optimization parity against SGLang's baseline is a separate
check. These tests do not establish PE quality, arbitrary-length session
capacity, or multi-session throughput.

### Trained-weight online checks

Actual `sglang serve` testing caught two HTTP integration gaps: the generic CUDA
generator default overrode Nomas's required CPU RNG, and undeclared model fields
were dropped by the video API. The model-owned request hooks now preserve
segments, continuation prompts, prompt format and normalized extrinsics, and
enforce CPU RNG. Five new regression cases cover this conversion and field
isolation from other models.

On one H200, HTTP creation, status polling and content download produce a
240-frame 704×1280 MP4 whose decoded pixels exactly match the offline API.
The warmed service completes this request in 50.605 seconds (50.623 seconds
including loopback download). This is one observation, not a throughput test.

The WebSocket test sends the next event only after receiving the previous
chunk's frames and statistics. Two 120-frame events emit 84, 120 and finally
36 frames; `end` flushes the withheld tail. Original RGB bytes match the
preceding SGLang implementation, and a new session with the same seed repeats
the same output. The candidate's cold request takes 73.548 seconds; a subsequent
request takes 55.685 seconds, with its first frame batch arriving at 22.360
seconds. The baseline's subsequent request takes 56.736 seconds. These are
single observations in separate server processes, not an alternating statistical
performance benchmark. They include frame conversion and raw WebSocket
transport, but exclude loading and MP4 encoding.

Additional trained-weight sessions use 17+32+9 frames at 352×640. Invalid events
are rejected while the session remains usable; a second concurrent connection
is rejected; disconnecting during generation releases worker state. A new
connection then reproduces the same 58 frames. GPU 0 idle memory returns to
17497 MiB after both complete control sessions and 17499 MiB after the interrupted
session. These whole-device readings include resident weights and allocator
state; they show no accumulating growth in this bounded test, not a proof of
unbounded-session memory safety.

Real trained-weight TP=2 and SP=2 WebSocket tests also complete 240 frames at
352×640 with 8 steps. SP output matches the same-size single-GPU RGB exactly.
TP differs from single GPU (uint8 mean absolute difference 6.533687, relative
L2 0.105485, max absolute difference 245), so trained-weight TP accuracy is not
certified by this validation. A separate run of the preceding implementation
in TP=2 produces exactly the same RGB as this implementation in TP=2: the
observed single/TP difference predates these kernel changes. Cold-request
timings of 30.484 s (single), 31.917 s (TP) and 29.477 s (SP) are functional-test
observations, not evidence of multi-GPU steady-state speedup.
