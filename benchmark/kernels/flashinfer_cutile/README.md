# cuTile NVFP4 validation and benchmarks

These scripts prepare reproducible evidence for the opt-in MiniMax cuTile
backend on SM120/SM121. A successful CPU test run does **not** validate CUDA
kernels, NCCL, model loading, or serving quality. GPU results must be collected
before describing any of these paths as validated.

## Environment and provenance

Use a dedicated single host. TP2 and TP4 need two and four visible GPUs,
respectively. A layer slice can fit on smaller SM120 GPUs; full-model serving
needs a separate capacity calculation that includes all checkpoint weights,
KV cache, temporary allocations, and accumulated workspaces.

Use an already provisioned GPU development environment and a local checkpoint.
These scripts do not install packages, download weights, or rent instances.

- Pin the SGLang checkout, checkpoint revision, PyTorch, FlashInfer, and CUDA
  toolchain. Keep FlashInfer 0.7.3 stock and 0.7.3 plus the exact patch from
  flashinfer-ai/flashinfer#9129 in separate environments and result directories.
  A label alone does not prove which patch was installed.
- cuda-tile's tileiras, nvcc, and nvvm packages must use a compatible matching
  major/minor toolchain. Record package versions and the driver. Marlin JIT
  compilation also needs a working full CUDA toolkit and headers.
- Archive the launch commands, package lock/freeze, source diff, and
  `nvidia-smi -q` / `nvidia-smi topo -m` output. For TP runs, use
  `NCCL_DEBUG=INFO` and archive the console log so the actual transport is visible.
- The benchmark records Git state, source hashes, device information, dependency
  versions, and raw samples. Checkpoint revision is an operator-supplied claim;
  verify the local shard files against that revision before publishing.
- Use a new output directory for every invocation. Existing directories are
  rejected. A partial result or failed status is not a passing measurement.

Run from the repository root with its Python package selected:

```bash
export PYTHONPATH="$PWD/python:$PWD"
export PYTHONDONTWRITEBYTECODE=1
```

## 1. GPU correctness

```bash
python test/manual/layers/moe/test_flashinfer_cutile_nvfp4.py -v
```

This covers distinct gate/up scales, partial and complete negative-ID padding,
clamped SwiGLU, output scaling, repeated calls, changed-input CUDA Graph replay,
stream isolation, and runtime-context reset. The real-checkpoint cases require
local `config.json`, the weight index, and the corresponding shard files:

```bash
export SGLANG_TEST_MINIMAX_LAYER30_DIR=/models/MiniMax-M3.1-Flash
python test/manual/layers/moe/test_flashinfer_cutile_nvfp4.py -v
```

The default real case uses the first 32 experts. Enable the existing large-case
switch only when all 128 layer-30 experts and sufficient memory are available:

```bash
SGLANG_TEST_CUTILE_LARGE_BATCH=1 \
  python test/manual/layers/moe/test_flashinfer_cutile_nvfp4.py -v
```

The matched W4A4 reference uses unpacked weights, cuTile's NVFP4 activation
midpoint rule, FP32 GEMM accumulation, and BF16 rounding at the cuTile GEMM
output boundaries. The separate BF16-activation reference exposes intentional
activation quantization loss. Neither reference implies bit identity with
Marlin W4A16. The relative-L2 gate remains 0.05. Investigate failures before
changing it and disclose any threshold change.

The stream test warms each execution before interleaving repeated calls and
retains independent output snapshots. This exercises isolation but does not
prove physical overlap without profiling. The context-reset tests distinguish
new execution/warmup state from reclamation of the old execution object after
Graph teardown. The weak-reference assertion does not quantify reclaimed GPU
bytes; allocator-reserved memory is not required to return to zero.
New GPU assertions must be run on hardware before they are reported as passing.

## 2. Layer latency and real tensor parallelism

Start with a small real-weight run; then use the full token matrix:

```bash
python benchmark/kernels/flashinfer_cutile/bench_cutile_nvfp4.py \
  --checkpoint-dir /models/MiniMax-M3.1-Flash \
  --checkpoint-revision CHECKPOINT_SHA \
  --experts 32 --tokens 1 64 412 4096 8192 8193 16384 \
  --backends cutile marlin --execution eager graph \
  --implementation-label pr49795 --flashinfer-variant stock-0.7.3 \
  --output-dir /results/pr49795-stock-tp1-e32
```

Repeat with `--experts 128`, and separately with `--input-std 4` to exercise
clamping. The default input standard deviation is 0.5. The `direct` backend
provides a public FlashInfer API comparison; it is not the Marlin baseline.
Synthetic mode is useful for smoke checks but must not be labeled real-weight
validation.

Use `torchrun` for genuine per-rank weights and NCCL SUM reduction:

```bash
NCCL_DEBUG=INFO torchrun --standalone --nproc-per-node=2 \
  benchmark/kernels/flashinfer_cutile/bench_cutile_nvfp4.py \
  --checkpoint-dir /models/MiniMax-M3.1-Flash \
  --checkpoint-revision CHECKPOINT_SHA \
  --experts 32 --tokens 64 412 4096 \
  --implementation-label pr49795 --flashinfer-variant stock-0.7.3 \
  --output-dir /results/pr49795-stock-tp2-e32
```

Repeat with four processes for TP4 and with 128 experts for the full expert
count. Every rank retains all selected experts; TP partitions the intermediate
dimension (3072 / TP), not the expert dimension. A sequential single-GPU sum
is not evidence of distributed TP support. `--distributed-timeout` sets the
process-group timeout in seconds (default 600); it must be finite and positive.
After output setup, a distributed worker that fails records its stage and exits
nonzero without entering collective cleanup. Initialization failures before
output setup remain in the launcher log. Launch TP cases with `torchrun`, which
terminates the other workers on failure. CPU tests of this path do not establish a bound
for real CUDA/NCCL failures; a GPU fault-injection check remains necessary.

Each latency result includes at least 100 measured calls, p50/p95, per-rank raw
samples, synchronized host timing, both numerical references, repeatability,
and Graph replay checks. NCCL is included in TP timing. The slowest rank is
selected for each sample before percentiles are computed. Timing rotates four independent input buffers (and four Graph captures),
without an input copy immediately before each sample. Changed-buffer replay
is checked separately. Loading, reference calculation, JIT, and warmup are
outside the timed region. FP32 references
default to the first 64 tokens of each input set; increase
`--reference-tokens` when broader numerical coverage is required. Full-tensor
repeatability and Graph checks are separate from that sampled reference.
Full output shape and finiteness checks also run outside the timed region,
including tokens beyond the reference prefix. Completed execution modes are
saved incrementally, so a later Graph failure preserves earlier eager samples.
The invocation still fails; partial results must not be reported as a passing run.

Marlin repetition/replay keeps its original relative-L2 limit of 0.001; cuTile
continues to require bit identity. Record measured repeat/replay errors as well
as identity flags. Preserve the existing 1-token Marlin failure. To keep one
known failure from preventing unrelated experiments, run token cases in separate
processes and new result directories, then report every pass and failure. Do not
silently skip a failed case or reuse an uncertain distributed process group.

The baseline calls SGLang's Marlin preparation and runner explicitly.
It does not claim the default SM12x model dispatcher selects Marlin.

## 3. Cumulative workspace memory

Use a fresh process for this experiment and keep the same runner, runtime
context, and capacity alive while visiting buckets and streams:

```bash
python benchmark/kernels/flashinfer_cutile/bench_cutile_nvfp4.py \
  --mode workspace --checkpoint-dir /models/MiniMax-M3.1-Flash \
  --checkpoint-revision CHECKPOINT_SHA --experts 128 \
  --tokens 1 64 412 4096 8192 8193 16384 8192 --capacity 16384 --streams 2 \
  --implementation-label pr49795 --flashinfer-variant stock-0.7.3 \
  --output-dir /results/pr49795-stock-workspace-e128
```

`workspace.json` and `workspace.csv` distinguish allocated, reserved, peak,
and cumulative memory. Streams are visited sequentially: this tests cumulative
residency, not concurrent throughput. No cache eviction or allocator clearing
occurs between steps. The adapter limits each FlashInfer call to 8192 tokens:
8193 becomes 8192 + 1, and 16385 becomes 8192 + 8192 + 1. The scheduler's total
accepted capacity is unchanged. Each chunk is copied into its original output
rows immediately; output scaling and the caller's NCCL reduction still apply
to the complete result. Small batches keep one call without this assembly copy.
Capture requires eager warmup of every chunk shape on the same stream,
including the tail.

This limits individual workspace buckets, not their cumulative byte total.
FlashInfer retains previously visited buckets; each stream has its own
execution. Never free a workspace referenced by a live CUDA Graph. Measure
the additional output/copy cost too: less memory does not imply lower latency.

### Chunk-boundary accuracy, Graph and memory protocol

Enable the manual large-batch tests for 8191/8192/8193/16384/16385 tokens,
including strided input, padded routes and scaling. Existing empty/one-token
checks remain separate. Real boundary cases sample 64 FP32-reference rows
across the batch, including both sides of each 8192 boundary and the last row.
This is not a full FP32 calculation of every row. Shape/finiteness, independent
same-partition output, repeatability and changed-input Graph checks still
cover the whole output.

Select `--reference-sampling chunk-boundaries` for the latency harness.
The default stays `prefix` to preserve the historical measurement protocol;
results record selected indices. `direct-chunked` calls public FlashInfer
independently per 8192 rows and concatenates results. It must match the adapter
bit for bit. `direct` uses one call for the full batch. Above 8192 this changes
GEMM shapes, so its cross-shape delta is reported against the existing 0.05 L2
bound. It does not replace the strict same-partition check.

```bash
python benchmark/kernels/flashinfer_cutile/bench_cutile_nvfp4.py \
  --checkpoint-dir /models/MiniMax-M3.1-Flash \
  --checkpoint-revision CHECKPOINT_SHA --experts 32 \
  --tokens 8191 8192 8193 16384 16385 --capacity 32768 \
  --backends cutile direct-chunked direct --execution eager graph \
  --reference-sampling chunk-boundaries --reference-tokens 64 \
  --implementation-label pr49795-chunked --flashinfer-variant stock-0.7.4 \
  --output-dir /results/pr49795-chunked-boundaries-tp1-e32
```

Repeat with 128 experts and torchrun TP2/TP4. The entire chunked MoE, output
assembly and one NCCL SUM are inside each timed call/Graph. All ranks must agree
after reduction. Changed input/routing replays exercise the tail too. The
manual real TP2 partition sum versus TP1 is a separate single-GPU numerical
experiment (the existing synthetic slice test's 2% bound), not NCCL validation.
The real-weight std=4 clamp test requires measurable clipping and retains both
W4A4 and BF16 references.

Collect three separate memory experiments for each expert count:

1. **Cold boundary control:** new process per token count (8192, 8193, 16384,
   16385), `--mode workspace --streams 1`. Compare
   `--workspace-backend cutile` to `--workspace-backend direct` in separate
   directories with the same capacity and inputs. Direct is an unchunked
   public-API control with one execution; multi-stream direct control is
   rejected to avoid sharing scratch buffers.
2. **Matched cumulative sequence:** keep the historical token sequence,
   capacity and two-stream order unchanged across before/after adapter
   checkouts. Fresh process per checkout; no context reset or cache clearing
   between visits. Report live allocation, reserved memory and peaks separately.
3. **All buckets up to 8192:** run the 21-bucket sequence below with two streams
   on both before/after checkouts. Confirm boundaries for the pinned FlashInfer
   version before using this protocol with another version.

```bash
python benchmark/kernels/flashinfer_cutile/bench_cutile_nvfp4.py \
  --mode workspace --checkpoint-dir /models/MiniMax-M3.1-Flash \
  --checkpoint-revision CHECKPOINT_SHA --experts 128 \
  --tokens 1 2 4 8 16 32 64 128 256 512 768 1024 1280 1536 1792 2048 2560 3072 3584 4096 8192 \
  --capacity 16384 --streams 2 \
  --implementation-label pr49795-chunked --flashinfer-variant stock-0.7.4 \
  --output-dir /results/pr49795-chunked-all-buckets-e128
```

Do not compare an old short sweep with a new 21-bucket sweep and attribute the
difference only to chunking. Capture is tested separately from eager memory
measurements. Preserve all historical Marlin failures and their original gates.
New GPU checks are unvalidated until executed.

## 4. Full-model serving and memory saver

Start the model server separately and retain its complete launch log. A first
smoke configuration can use TP4, EP1, BF16, `flashinfer_cutile`,
`--moe-a2a-backend none`, a conservative chunked-prefill size of 8192, and low
concurrency. Speculative decoding and CPU offload are outside this backend's
supported scope. First establish eager serving, then test the project's Graph
configuration separately. Do not assume a layer benchmark proves the attention
backend, KV layout, or full-model loading works.

Use the actual served model name and a local tokenizer directory:

```bash
python benchmark/kernels/flashinfer_cutile/bench_cutile_service.py \
  --base-url http://127.0.0.1:30000 --model MiniMax-M3.1-Flash \
  --tokenizer /models/MiniMax-M3.1-Flash \
  --server-revision SERVER_SHA --checkpoint-revision CHECKPOINT_SHA \
  --implementation-label pr49795 --flashinfer-variant stock-0.7.3 \
  --output-dir /results/pr49795-stock-serving
```

The default matrix is input lengths 128/2048/8192, output lengths 128/256,
and concurrency 1/8/16. It invokes the current `sglang.benchmark.serving`
module with fixed-length random token IDs, fixed seed, warmup, and cache flush.
An unscored full-workload pilot precedes each measured case; its artifacts
are retained separately.
Every request must succeed and match the requested lengths before a case
passes. Raw JSONL/logs and `serving-summary.csv` retain TTFT, TPOT, and throughput.

`--dry-run` prints the commands without files, network requests, or subprocesses.
For a first smoke, select `--input-lens 128 --output-lens 128 --concurrencies 1`.
The metadata distinguishes the benchmark client from the model server; attach
the server's dependency versions and source diff as well.
`--case-timeout` limits each pilot and measured client process separately
(default 1800 seconds). It is distinct from `--http-timeout`, which controls
the wrapper's own HTTP requests. On timeout the driver retains the case/stage
and client log, fails the invocation, and stops further cases. It terminates
the benchmark client, not the model server; inspect pending server requests
before starting another run.

On a dedicated server started with **both** `--enable-memory-saver` and
`--enable-weights-cpu-backup`, run a separate `--mode lifecycle` invocation.
It verifies deterministic cold-cache outputs before releasing weights and
KV cache, attempts resume even after an ambiguous release failure, then checks
fresh token IDs and output logprobs after restoration. Each generation must
finish normally with the requested 64 tokens; completion-token counts, token
IDs, and logprob counts must agree. Aborted or truncated outputs cannot pass
merely because they repeat, and the default logprob tolerance remains zero.
Inspect the resource-cycle
record and failed status if either endpoint fails; do not continue reporting
throughput from an uncertain server state. The server-info memory fields
are startup accounting, not a measurement of physically released GPU memory.
Collect separate server-side NVML snapshots before release, while released,
and after resume before claiming a memory-saving amount.

Random-token throughput and lifecycle generation are not model quality scores.
Add the project's GSM8K evaluation on the same fixed examples and decoding
settings for both implementations, preserving per-example outputs and truncation
counts. Report the sample size and uncertainty; do not equate one successful
generation with an accuracy evaluation.

## Benchmark protocol and review evidence

Use the same harness files and immutable input protocol when comparing
implementation revisions in separately provisioned checkouts. Select
`--cutile-api runner-core` for this adapter or `--cutile-api fused` for a
compatible fused entry point. The shim calls the selected implementation;
it does not substitute the direct FlashInfer kernel.
Copying the harness into another checkout must preserve its file hashes and
must not overwrite that checkout's production implementation. Provisioning and
checking out those revisions are separate preparation steps.

Run stock FlashInfer and the exact #9129 patch as separate experiments.
Alternate implementation order, hold clocks/power policy and model/routing
settings fixed, and retain slowdowns or OOM failures alongside speedups.
Report per-case latency and memory before any aggregate speedup.
A result from another GPU, reference arithmetic, token count, or toolchain
is not a like-for-like baseline.

A review report should include:

- Implementation/checkpoint/harness revisions, environment and topology.
- Correctness results, thresholds, sample coverage, and explicit skips.
- TP1/TP2/TP4 evidence, including NCCL logs and per-rank status.
- Eager/Graph latency with raw samples and cumulative memory data.
- Full-model launch, service results, quality evaluation, and lifecycle result.
- Known unsupported operations, failed cases, and untested hardware.

Do not label this evidence complete until the actual GPU and server runs finish.

## Extended validation protocol

The 8192-token adapter limit bounds **each FlashInfer call**, not the sum of
resident workspaces. Different visited buckets and streams still accumulate.
Keep graph-referenced workspaces alive until their captures are destroyed.

The independent coverage-v1 bundle contains run_cutile_coverage_v1.py,
collect_cutile_marlin_performance_v2.py, profile_cutile_kernels_v1.py, and
test_cutile_coverage_v1.py. These remain outside the production repository.
Their source hashes, the six-file candidate diff, environment versions, full
commands, and raw results must travel with the report.

The independent coverage index groups the validation assertions:
A01–A14 (correctness), S01–S08 (timing), P01–P04 (profiling),
M01–M07 (memory), T01–T04 (tensor parallelism), C01–C04 (integration).
The bundle's fixed coverage-plan.json and audit crosswalk provide the assertion
mapping. An experiment count does not prove that an assertion passed.

### Comparable inputs and references

The historical input protocol preserves the CUDA BF16 generator and normalized
routing weights. The `--input-protocol peer-compatible` option uses
CPU float32 generation, BF16 conversion, distinct randperm expert IDs, and
unnormalized random weights. Dense router scores satisfy the Marlin API; this
does not benchmark a learned router. Four inputs have full-tensor SHA256 hashes,
including router scores. Every TP rank must agree on all hashes.

The --reference-protocol both option retains the corrected NVFP4 reference as
the 5% gate and adds a separately labeled FP32-intermediate diagnostic. Marlin remains
W4A16 and is checked against the BF16-activation reference; its repeat/replay
relative L2 limit remains 0.001. cuTile must remain bit-identical across repeats
and eager/Graph execution.

The v2 collector may finish timing only for the already recorded Marlin repeat
failures at 1/4096/8192/8193/16384 tokens. Those rows still fail their original
gate. Any cuTile failure, new token failure, malformed/nonfinite output, reference
failure, or execution error stops the stage.

### Software and measurement scope

Keep the current stack (PyTorch cu132 / FlashInfer 0.7.4) separate from the
cu130 / FlashInfer 0.7.3 stack. Both stock and the fixed
[FlashInfer #9129](https://github.com/flashinfer-ai/flashinfer/pull/9129) patch are
measured; the latter is upstream work, not an optimization attributed to this
adapter. Compiler variants 13.3/13.4/13.5 use isolated environments.
Unsupported toolchains or profiling permissions remain explicit gaps.

The standard protocol uses four inputs, ten warmups/input, 100 raw samples,
twenty repeat checks, and seven Graph mutation/restore checks. The separate
compatibility bridge uses a one-warmup/input timing protocol and the direct
Marlin API. Extra correctness checks run in a separate validation
pass. Do not combine these timing rows into an unlabeled average.

Profiling uses one warmed MoE call inside CUDA profiler start/stop.
Record GEMM1/GEMM2, registers, launch dimensions, and static/dynamic shared memory.
Profiled duration is not used as the ordinary benchmark latency. Preserve CUBIN
bytes and hashes with their compiler and software-stack provenance.

### Memory and actual distributed checks

The --workspace-warm-repeats option checks live allocation after dropping each
output. --require-all-buckets validates the sequence against the installed
FlashInfer mapping. Record cold/warm peak, output-live allocation, reserved
memory, actual cache keys, owner identity, and unique storage bytes separately.
The benchmark fixture retains source weights; only the production preparation
test supports conclusions about released source copies.

The --compare-tp1 option requires real checkpoints and TP2/TP4 launched with
torch.distributed.run. It compares actual BF16 NCCL-reduced outputs to a separate
full-weight TP1 call outside timing/capture, using the existing 2% TP bound.
Output hashes and both reference errors are recorded. A single-GPU sum of TP
slices is a distinct test and cannot replace this experiment.

Historical results describe their original source hashes and do not establish
quality for later source snapshots. Service regression covers eager/decode Graph,
8192/16384 prefill chunks, and observed MoE call/chunk sizes; prior full GSM8K and
high-concurrency evidence retain their original source labels.
