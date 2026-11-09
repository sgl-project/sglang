# Full KV reuse during GDN replay

This manual benchmark exercises real HTTP serving, scheduling, prefix matching,
Full KV allocation/borrowing, GDN checkpoint tracking, and request cleanup. It
compares `SGLANG_ENABLE_GDN_KV_REUSE=0` against `1` on the same source snapshot.

## Request fixture

With chunk size 4096, page size 256, and
`--mamba-max-states-per-path 1`:

1. A 12545-token seed creates Full KV through the long prefix. Its final
   checkpoint allows an older unlocked GDN checkpoint to be evicted while
   retaining Full KV.
2. A divergent 4352-token branch shares the first 4096 tokens and recreates a GDN
   checkpoint at that fork.
3. The target shares seed tokens through 8192 and adds a nine-token suffix. Its
   common cache hit is 4096, so GDN replays 4096 tokens. Reuse borrows Full KV for
   that replay chunk. No cache or state tensors are injected.

`bench.py` asserts the target's common cache hit is 4096. For multi-token
correctness runs it repeats the target and asserts an 8192-token cache hit.
Optimized server logs must contain
`GDN_KV_REUSE ... C=4096 H=8192 tokens=4096`.

## Run

Install the SGLang runtime and model dependencies first. Use a single GPU with
enough memory for BF16 weights and the configured caches. The recorded experiment
used one RTX 5090, TP=1, and BF16 conv/SSM state as well as weights. This explicitly
overrides a model's configured FP32 SSM default. Both arms use identical settings.

Start one server, using flag `0` for baseline or `1` for reuse:

```bash
CUDA_VISIBLE_DEVICES=0 bash benchmark/gdn_kv_reuse/launch_server.sh \
  /path/to/Qwen3.5-4B 0 30000 > base-server.log 2>&1
```

After the server is ready, run the correctness request, three additional warmup
trials, and 16 timing trials:

```bash
python benchmark/gdn_kv_reuse/bench.py --model-path /path/to/Qwen3.5-4B \
  --output-dir base-correctness --new-tokens 32
python benchmark/gdn_kv_reuse/bench.py --model-path /path/to/Qwen3.5-4B \
  --output-dir base-warmup --new-tokens 1 --repeats 3 --rid-prefix warmup-
python benchmark/gdn_kv_reuse/bench.py --model-path /path/to/Qwen3.5-4B \
  --output-dir base-timing --new-tokens 1 --repeats 16
```

Stop the server and repeat with mode order **baseline/reuse/reuse/baseline**,
using distinct output directories and the same GPU. Discard the first timing
trial of each block, leaving 30 trials per mode. Retain the server's
`ReqTimeStats(rid=target-..., output_len=1)` forward durations. Warmup request IDs
start with `warmup-` and must be excluded. Compare all four 32-token outputs,
selected-token logprobs, and per-step top-5, plus each warm-cache control.

Server forward duration includes replay, the new suffix, LM head, and first-token
processing; it excludes seed/branch preparation and HTTP round trips. It is not a
CUDA-event measurement of the replay kernel alone. Client times are saved
separately in `responses.json`.

Replace the model path with `Qwen3.5-9B` or `Olmo-Hybrid-7B` to test the other
models. For Olmo, keep the final path component `Olmo-Hybrid-7B`, as the launcher
uses that name to select its external fixture.

## Experimental Olmo fixture

This fork has no native Olmo-Hybrid implementation. `olmo_fixture` supplies the
external model registration, weight loading and model equations, and an Olmo GDN
backend with `beta = 2 * sigmoid(b)`. It uses native SGLang RadixAttention,
RadixLinearAttention, paged KV, and checkpoint machinery. Separate checkpoint
q/k/v conv weights are concatenated in that order and all parameters are checked
for loading.

The fixture supports unquantized BF16, TP/PP=1, NoPE, and the explicit Triton
backends used here. It preserves Olmo's separate projections and Transformers
norm/MLP operations. Both benchmark modes use this same adapter. Its
`sglang_olmo_hybrid_adapter` marker allows the explicit replay path; ordinary
unregistered Olmo configurations fall back. This is experimental benchmark
coverage, not a claim of tuned native Olmo model support. Source hashes record
that the local external fixture used for the runs matches the fixture supplied
here, apart from formatting.

The original fixture was independently checked against unmodified Transformers
on 128/512-token prompts with eight greedy output tokens: token IDs matched,
with maximum selected-token logprob differences below 8e-5. This is not full
vocabulary or internal-state equality.

## Results

The current-fork comparison and per-block samples are recorded in `results.json`.
The PR description reports the three models on this common source base. Treat
sub-percent differences cautiously when between-block variation is comparable to
the measured saving. No Nsight trace or kernel-level attribution is claimed.

All three rows below use upstream base `3825fcd39` plus this branch's replay
changes, GPU 0, and the same ABBA protocol above. These replace the earlier
mixed-base experiments. Each mode retains 30 timing samples per model.

| Model | Implementation | Baseline median | Reuse median | Saved | Reduction |
|---|---|---:|---:|---:|---:|
| Qwen3.5-4B | Native Qwen3.5 | 215.92 ms | 214.95 ms | 0.97 ms | 0.45% |
| Qwen3.5-9B | Native Qwen3.5 | 350.45 ms | 347.81 ms | 2.64 ms | 0.75% |
| Olmo-Hybrid-7B | External Olmo fixture | 386.46 ms | 373.70 ms | 12.76 ms | 3.30% |

All 12 block-level 32-token outputs, selected-token logprobs, per-step top-5,
and each warm-cache control matched exactly within its model. Every optimized
block logged 17 target replay activations plus three warmup replay activations;
baseline blocks logged none. Fourteen CPU ownership/fallback tests also pass.
These are single-prompt regression checks, not task-level accuracy evaluation.
Qwen gains remain below 1%; the per-block samples in `results.json` expose the
variation rather than establishing a universal speedup. Olmo is experimental
adapter coverage; its absolute performance and fusion level differ from native
Qwen implementations.
