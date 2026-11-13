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
occurs between steps. The current backend has no automatic 8192-token chunking;
an 8193-token workspace increase must remain visible. Never free a workspace
while a live CUDA Graph can still reference it.

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

## Fair comparison and review evidence

For #49795 versus #49791, use the same harness files and immutable input protocol
in separately provisioned checkouts. Select `--cutile-api runner-core` for the
former and `--cutile-api fused` for the latter. The shim calls each implementation's
real entry point; it does not substitute the direct FlashInfer kernel.
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
