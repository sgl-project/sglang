# Full KV reuse during GDN replay

This manual benchmark exercises real HTTP serving, scheduling, prefix matching,
Full KV allocation/borrowing, GDN checkpoint tracking, and request cleanup. It
compares `SGLANG_ENABLE_GDN_KV_REUSE=0` against `1` on the same source snapshot.

## Pure 8192-token replay

Use `bench_pure_replay.py` for an **8192-token input and one output token**.
No query suffix is appended. The GDN prefix is 4096; Full KV is cached through
8192. Each request therefore replays exactly one 4096-token chunk, all of whose
Full KV can be reused, including at concurrency 8 or 16.

The untimed preparation submits a seed of 8449 tokens and then extends it to
8705, allowing the native path-state cap to evict the unlocked 8192 checkpoint
without retaining a long KV suffix. A divergent branch recreates the 4096 GDN
checkpoint. No cache or state tensors are injected.

```bash
CUDA_VISIBLE_DEVICES=0 MAX_RUNNING_REQUESTS=16 MAX_MAMBA_CACHE_SIZE=64 \
  MAX_TOTAL_TOKENS=81920 MEM_FRACTION_STATIC=0.9 PYTHON=/path/to/venv/bin/python \
  bash benchmark/gdn_kv_reuse/launch_server.sh /path/to/Qwen3.5-4B 1
python benchmark/gdn_kv_reuse/bench_pure_replay.py \
  --model-path /path/to/Qwen3.5-4B --concurrency 16 \
  --new-tokens 1 --output-dir pure-replay
```

The normal matcher leaves the last input token for logits, so it may request an
interior GDN checkpoint at 7936. This does not prevent KV reuse: the extra Full
pin and protected prefix remain at 8192 until a later checkpoint lock covers
them, or until teardown. See [admission audit](ADMISSION_AUDIT.md).

## Historical fixture with a new suffix

The historical results below used a nine-token query suffix and an 8201-token
input. They are not measurements of the pure 8192-token case above.

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

## Recorded single-request results

The single-request snapshot at `b5f213ae3` and its per-block samples are recorded
in `results.json`.
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

## Multiple requests and eight-concurrent benchmark

Replay eligibility is checked per request. A batch can contain both reusable
chunks and ordinary chunks: reused requests borrow and pin their cached Full KV,
while other requests allocate fresh slots. Q and any attention gate are computed
for all tokens; K/V are computed and written only for ordinary chunks. The
single-request restrictions above still apply to each reusable chunk, including
alignment and the requirement that the entire chunk is cached. Both normal
and overlap scheduling are supported. Prefill graphs and quantized projection
weights remain outside this path. Native Qwen projections use rank-local
weights; there is no blanket TP/PP=1 or BF16-only gate. The external Olmo fixture
still has its own TP/PP=1 and model-equation restrictions.

Set `ENABLE_OVERLAP=1` with the launcher to use SGLang's overlap scheduler.
For eight requests, also set `MAX_MAMBA_CACHE_SIZE=64`: overlap uses five
Mamba state slots per request in this extra-buffer configuration, so the
historical 32-slot fixture would cap the scheduler at six requests. For example:

```bash
CUDA_VISIBLE_DEVICES=0 ENABLE_OVERLAP=1 MAX_MAMBA_CACHE_SIZE=64 \
  MAX_RUNNING_REQUESTS=8 MAX_TOTAL_TOKENS=98304 \
  MAMBA_SSM_DTYPE=float32 MEM_FRACTION_STATIC=0.9 \
  PYTHON=/path/to/venv/bin/python \
  bash benchmark/gdn_kv_reuse/launch_server.sh /path/to/Qwen3.5-4B 1
```

The scheduler's existing per-forward field snapshot retains the compact K/V
index tensor across streams. A request's temporary Full pin remains held until
a checkpoint lock covers its entire borrowed KV prefix, or until teardown. If the previous replay still holds that
pin when the request is scheduled again, the new chunk falls back rather than
replacing the in-flight pin. No additional stream synchronization is introduced.

The finite overlap regression in `overlap_results.json` covers Qwen3.5-4B,
Qwen3.5-9B, and the external Olmo-Hybrid-7B fixture. Each model runs the long
`D=4096` case, eight shared-prefix reusable requests, and four reusable plus
four ordinary requests. Each request generates 32 tokens. With fixed-order MM,
Base/reuse token sequences, selected logprobs, and top-5 agree exactly, including
warm controls. Separate ordinary CUDA runs verify completion and actual reuse;
they do not assert bitwise equality. Server logs confirm real eight-request
shared/mixed batches. Both prefill and decode graphs are disabled in these
checks; this is functional validation, not a latency benchmark.

Start a server with enough KV capacity for eight distinct cached continuations:

```bash
CUDA_VISIBLE_DEVICES=0 MAX_RUNNING_REQUESTS=8 MAX_TOTAL_TOKENS=98304 \
  MAMBA_SSM_DTYPE=float32 MEM_FRACTION_STATIC=0.9 \
  bash benchmark/gdn_kv_reuse/launch_server.sh /path/to/Qwen3.5-4B 0 30000
```

Run eight server requests together through the batched generation API:

```bash
python benchmark/gdn_kv_reuse/bench_concurrent.py \
  --model-path /path/to/Qwen3.5-4B --output-dir base-eight-correctness \
  --new-tokens 32
python benchmark/gdn_kv_reuse/bench_concurrent.py \
  --model-path /path/to/Qwen3.5-4B --output-dir base-eight-warmup \
  --new-tokens 1 --repeats 2 --rid-prefix warmup-
python benchmark/gdn_kv_reuse/bench_concurrent.py \
  --model-path /path/to/Qwen3.5-4B --output-dir base-eight-timing \
  --new-tokens 1 --repeats 9 --rid-prefix timing-
```

The eight seeds share their first 4096 tokens and diverge at the next token.
Each target has its own continuation with `C=4096`, `H=8192`, and `D=4096`,
so completing one target cannot warm another's GDN replay path. A round flushes
and seeds the cache before submitting its eight targets; preparation is excluded
from `wave_ms`. This is a repeated burst test, not steady-state serving under a
continuous stream of arrivals. `wave_ms` measures the client time to complete
all eight requests. Request throughput is eight divided by mean wave time.

Keep chunk size 4096. This is the scheduler's aggregate prefill token budget,
so eight concurrent long requests do not form one eight-request replay batch.
Scheduling can split the replay into smaller chunks and mix a cached chunk with
another request's uncached tail. Such mixed batches are supported, while an
individual chunk containing both cached and uncached tokens still falls back.
Record `GDN_KV_REUSE_BATCH` and per-request `GDN_KV_REUSE` logs to check actual
batch sizes and reuse coverage.

Two additional checks fit eight requests into that same 4096-token budget:

```bash
python benchmark/gdn_kv_reuse/bench_concurrent.py \
  --model-path /path/to/Qwen3.5-4B --output-dir eight-shared \
  --small-batch-check all-reuse --new-tokens 32
python benchmark/gdn_kv_reuse/bench_concurrent.py \
  --model-path /path/to/Qwen3.5-4B --output-dir eight-mixed \
  --small-batch-check mixed --new-tokens 32
```

These use `C=256`, `H=512`, and `D=256`. The first submits eight identical
cached continuations to exercise shared slots and independent pins. The second
mixes four such requests with four ordinary chunks containing a new suffix.
Expected optimized logs include `size=8 reused=8 fresh_tokens=0` and
`size=8 reused=4 fresh_tokens=1060`, respectively. These checks validate the
multi-request path; their latency is not part of the long-replay timing table.

Repeat on the same GPU with flag order **0/1/1/0** for all three models. Discard
the first timing wave per block, retaining 16 waves (128 requests) per mode.
Compare generated token IDs, and record logprob differences separately rather
than assuming bitwise equality across different batch/chunk shapes.

## Precision and correctness checks

Use FP32 SSM state for the new concurrent checks. The historical single-request
measurements explicitly used BF16 state; that setting rounds recurrent
checkpoints and is not an oracle for exact equality across replay partitions.

The Triton GDN path now preserves the FP32 accumulator when snapshotting an
interior prefill checkpoint. Previously it restored that checkpoint through
BF16 intermediate `h`, even with a FP32 state pool. The snapshot is now passed
through the existing low-level kernel, with the same behavior in eager and
captured prefill. The new `test_gdn_track_state.py` compares against an exact
FP32 truncated-prefix reference and verifies unchanged outputs/final states.

For a controlled Base/reuse equivalence check, prepend `BATCH_INVARIANT_MM=1`
to **both** server launches, along with `MAMBA_SSM_DTYPE=float32`. This loads the
explicit correctness-only `batch_invariant_mm` fixture, overriding only CUDA
`aten::mm` with SGLang's existing fixed-order Triton matrix multiplication.
Real cache matching, scheduling, KV borrowing, and attention remain enabled.
The fixture also loads the Olmo registration when needed. It is outside the
production path and must be disabled for ordinary latency measurements.

Different GEMM shapes can round differently in the ordinary CUDA path; a close
greedy decision can change despite mathematically equivalent operations.
Record those differences rather than claiming default-path bitwise equality.
The global `--enable-deterministic-inference` flag cannot serve as this oracle:
it automatically disables radix caching with FlashInfer in this source base.

Compare Base and reuse targets, and separately compare their warm controls.
A cold-versus-warm comparison within either mode changes the cache partition
and is an additional backend diagnostic, not an interchangeable Base/reuse
comparison. Keep such differences visible; task-level accuracy, cancellation,
OOM/eviction pressure, and Rust TreeCore validation remain separate work.

## Pure 8192-token concurrency comparison

`pure8k_results.json` records the completed three-model comparison at commit
`70d677788`: input 8192, output 1, chunk 4096, GDN hit 4096, KV hit 8192.
All replay KV rows were reused at concurrency 1/8/16. Forward p50 reductions
were 0.68–0.74% for Qwen3.5-4B, 0.68–0.89% for Qwen3.5-9B and
3.01–3.29% for the experimental Olmo fixture. These measurements predate
the subsequent main merge; they are not performance measurements of the merged code.

Reproduce the ABBA suite on one available GPU with the three model directories
(and their `download_revision.json` provenance files) under MODEL_DIR:

```bash
python benchmark/gdn_kv_reuse/run_pure8k_concurrency.py \
  --model-dir MODEL_DIR --output-dir NEW_RESULT_DIR
```

The suite uses physical GPU 0, starts its own servers on port 18110 and requires
enough free GPU memory. Each round uses the same 16 targets, submitted in
groups of 1/8/16; each arm retains 30 rounds / 480 requests per concurrency.
The global 4096-token budget usually serializes the 4096-token prefills;
serving concurrency is not prefill batch size. Cache preparation is untimed.
The suite checks replay coverage and records ordinary-CUDA one-token output
comparisons; it does not assert general numerical equivalence.
