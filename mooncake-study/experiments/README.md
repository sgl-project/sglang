# H100 Capture Experiments

Use the image digest and dependency overlay in `h100-runtime-lock.json`.
`requirements-h100.txt` is an overlay on that image, not a standalone lock for
an arbitrary host. The initial image's Torch 2.11/kernel 0.4.4 pair cannot run
this source checkout; changing only the kernel produces a Torch ABI failure.

The resident lab uses these GPU-visible paths:

- Lab: `/gpfs/users/fuxuanwei-1/dspark-maas-lab`
- Source: `sglang/python`, selected through `PYTHONPATH`
- Experiment Python: `venvs/capture/bin/python`
- Offline wheel cache: `wheels`
- Model: `/gpfs/models/huggingface.co/Qwen/Qwen3-0___6B`
- Logs/results: `state/logs` and `state/results`, indexed by submitted job ID

The experiment venv inherits the base image packages and installs the locked
overlay with `--no-index --find-links <wheel-directory> --no-deps`. The source
is installed editable with `SGLANG_BUILD_RUST_EXTS=none` for these text HTTP
tests; this does not validate the native Rust entrypoints. Optional inherited
FlashInfer 0.6.12 cubin/JIT-cache distributions were removed because they are
incompatible with FlashInfer 0.6.17. Supported source JIT is used with version
checks enabled. The idle worker continues to use the base Python/Torch pair.

Every GPU experiment goes through the resident worker so the idle load is
stopped and reaped before test execution, then resumed afterward:

```bash
LAB=/gpfs/users/fuxuanwei-1/dspark-maas-lab
python3 "$LAB/bin/gpu_worker.py" --state-dir "$LAB/state" submit \
  --cwd "$LAB/sglang" --timeout-seconds 900 \
  --env "PYTHONPATH=$LAB/sglang/python" \
  --env "PATH=$LAB/venvs/capture/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -- "$LAB/venvs/capture/bin/python" \
  test/registered/storage/test_training_capture_runtime.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B -v
```

Stage changed source before submission with `rsync -aR --exclude=__pycache__`,
using paths relative to the implementation checkout. The submit command returns
a job ID; inspect its result JSON for a terminal outcome and exit code, and its
log for assertions. A queue or heartbeat file alone is not test completion.

Other experiment commands, under the same queue/environment:

```bash
python -m unittest discover -s test/registered/unit/training_capture -v
python -m pytest test/registered/storage/test_training_snapshot_mooncake.py -v -s
python mooncake-study/experiments/diagnose_qwen3_kv.py --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B
```

The real capture test starts isolated Mooncake master/data-node/client processes
and an HTTP Catalog test double. The segment owner outlives the producer to
test snapshot lifetime. It starts one observed ordinary SGLang server and then
one normal CUDA-graph SGLang server. Synthetic request tokens and temporary
model-observer dumps stay inside the test's temporary directory.

Two AR overlap servers then run with ordinary execution and decode CUDA graphs.
They check one-iteration result lag, three-request batches padded to a four-row
graph, prefix dedup/remapping, exact Host capacity, EOS, delayed grammar sampling
and a streamed request aborted while results are pending. Each mode waits for the
abort's fenced Catalog failure and reads its completed snapshots after producer
exit. The correctness driver waits for enough spare capture reservations before
each request/batch; it is not a saturation or throughput benchmark.

The overlap observer follows actual forward input tokens and preserves the first
KV observation for each position, because later Radix prefix dedup can change the
physical mapping. It reads raw vocabulary scores before serving-side processing.
Its synchronous reads are test-only. Ordinary AR and static DSpark capture both
support default overlap scheduling. Speculative observation also follows actual
verify inputs and model-accepted output prefixes, since CPU output IDs can lag.

`diagnose_qwen3_kv.py` compares HF eager/SDPA and BF16/FP32 without any Mooncake
or capture code. Reloading each dtype is intentional: casting an entire model
to BF16 would also narrow FP32 RoPE frequency buffers and invalidate the
comparison. This numerical diagnostic is separate from the zero-error online
capture check. No real training, cross-node RDMA or performance SLO is certified
by these experiments.

The runtime test also exports synthetic `DSparkTargetKVDraftModel` checkpoints
from a retrieved sample's teacher/KV contract. Four additional servers check
ordinary and CUDA-graph speculative generation with and without overlap,
including batches, both verify accept/reject branches and continued serving
after management API rejections. A test-only observer compares projected draft
KV against per-layer reference math and checks actual graph replay with no
target hidden capture.
The overlap pair also checks three-request graph padding, exact Host capacity
and streamed abort while verify results are pending. The abort helper waits for
fenced Catalog failure and reservation recycling, and rejects quarantined slots.
The fixture's forced proposal weights are not a trained draft. This extends the
same command above; allow 900 seconds and retain the job log for evidence.

The command also starts an ordinary overlap server with adaptive admission
enabled. Its test-only entrypoint holds the first background write using a file
beside the publication journal. While a spare Host slot remains available, a
second generation request completes and is excluded from capture by the stalled
writer's cooldown. Removing the pause lets the original snapshot publish; after
ratio recovery, a later request produces another validated Store snapshot. This
exercise adds two completed snapshots and does not measure a latency SLO.

That adaptive server also enables `--enable-metrics`. The test scrapes the
actual HTTP `/metrics` endpoint while the writer is held and after release,
checking multiprocess export of the zero/recovered sampling ratio, READY and
adaptive-exclusion counters, writer age, reset reservation gauges and zero
quarantine. The producer's monitoring thread is independent of its writer and
Catalog threads. Unit tests additionally cover bounded labels, repeated metric
updates, exporter failure/retry and a blocked Catalog. The provisioned dashboard
is `examples/monitoring/grafana/dashboards/json/training-capture-dashboard.json`;
its serving-latency panels do not establish capture-overhead thresholds.

An additional ordinary overlap server enables `adaptive.latency` with 500ms
scheduler TTFT/TPOT budgets, a 1.5s observation window and one-observation
minimum. These short settings are test parameters. The test-only server delays
result processing by 750ms while `latency.pause` exists in its journal directory.
The first admitted request still publishes, while the next request completes
without capture. After removing the delay and expiring the window, the controller
remains paused until an unsampled healthy request supplies fresh observations.
Capture then recovers and a fourth request publishes. All four output sequences
agree, both snapshots are validated again after producer exit, and HTTP metrics
report four TTFT observations. This verifies feedback wiring and recovery, not
capture-induced overhead or a service SLO. The retained result and scope are in
[`capture-latency-feedback.json`](capture-latency-feedback.json).

The runtime fixture uses the Mooncake master's default read lease (5 seconds
in the pinned SDK). An earlier 100ms override produced `LEASE_EXPIRED (-707)`
during readback after a producer exited. This is the transfer's read lease,
separate from object hard pinning and Catalog retention. The adapter continues
to reject the read and quarantine its destination on negative SDK status;
the unit test explicitly covers `-707`. The transport-focused small-object test
still uses its short lease. Runtime correctness does not require a 100ms
transport deadline during process teardown.

### Capture On/Off Benchmark

`benchmark_training_capture.py` launches normal SGLang servers with overlap,
decode CUDA graphs and tokenizer/detokenizer enabled. It reuses
`sglang.benchmark.serving` for native streaming HTTP measurements. Each phase
owns an isolated real Mooncake master/data segment and a Catalog test double,
including baseline phases, so unrelated Store contents are never touched.
Capture-off means that `--training-capture-config` is absent.

Run through the resident worker, allowing 1800 seconds for the default matrix:

```bash
python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  --source-revision <implementation-git-commit> \
  --output-dir /absolute/new/experiment-directory \
  --num-prompts 2048 --input-len 128 --output-len 32 --concurrency 8 \
  --ratios 0.001 0.01 0.1 --repeats 2
```

The output directory must be new. Source revision is supplied explicitly because
the GPU staging directory does not contain `.git`; the report also records
producer-file and driver digests and installed runtime versions. The default
10% upper ratio is an experimental setting, not a production recommendation.
Adjust selected layers, Host slots and segment capacity for another model.

Every phase uses the same seeded, fixed-length `random-ids` workload. Warmup
runs at batch sizes one and the concurrency limit, then flushes the prefix cache.
The existing benchmark runs with streaming, greedy sampling, `ignore_eos` and
no additional warmup. Concurrency is closed-loop: client semaphore wait is not
part of request TTFT. The report retains cache-hit counts; this is synthetic
load, not a claim about real request lengths, arrival rates or prefix sharing.

Each round is bracketed by capture-off runs, and odd rounds reverse capture
ratio order. `report.json` includes client TTFT/TPOT p50/p95/p99, output throughput,
request success counts, baseline drift, ratios against the mean of the two
baseline measurements, and per-phase capture counters. These ratios are
descriptive measurements, not confidence intervals or an SLO pass/fail gate.

Warmup counters/publications are excluded. After timed requests finish, the
driver drains capture, stops the producer, then reads every measured READY
snapshot from the surviving Store and validates its digest, tensor content and
sequence lengths. It reports admitted/selected/READY fractions, backpressure,
Host state and separate post-client drain time. The request stream never waits
for capture slots. A zero-sample run is explicitly marked, so low-rate absence
of data cannot be mistaken for validated capture. Payload and manifest byte
totals describe successfully read objects, not measured D2H/RDMA traffic.

The experiment currently leaves GPU capture-kernel time, D2H/wire counters and
production SLO status unset. Those require separate profiling/transport evidence
and deployment-specific thresholds. All servers and Store resources are closed
before the worker resumes idle load; per-phase client logs/results and the
incremental report remain in the experiment directory.

`profile_training_capture.py` is a separate experiment for GPU attribution:

```bash
python mooncake-study/experiments/profile_training_capture.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  --source-revision <implementation-git-commit> \
  --output-dir /absolute/new/profile-directory
```

It collects capture-off/on traces for separate 128-to-1 prefill and 1-to-32
decode workloads, batch size eight, ten warmup batches and five measured batches
per workload. The server uses a test-only entrypoint that places CPU profiler
scopes around teacher top-k/LSE, KV export, teacher D2H and position D2H. The
underlying implementations are unchanged; no observer copies are introduced.
Each probe waits for spare capture slots, so all measured requests in the enabled
case must reach READY. This waiting and profiler overhead make its wall times
unsuitable for the serving comparison above. The artifact records capture counts
and trace paths. Analyze the traces with the repository's profiler triage skill;
CPU scope time is not GPU time, and payload bytes are not a DMA measurement.

For explicit capture attribution, run the standard-library trace summarizer on
each enabled trace (omit `--require-capture` for the disabled traces):

```bash
python mooncake-study/experiments/summarize_capture_trace.py \
  /path/to/on/decode/decode-TP-0.trace.json.gz \
  --require-capture --output /path/to/capture-attribution.json
```

It follows CUDA correlation IDs from runtime/driver calls inside the CPU
`user_annotation` ranges to kernels and D2H activity. Same-name GPU annotations
are excluded from CPU counts. Missing byte fields remain unknown. The output
includes trace hashes, operation counts, summed device durations and DMA bytes;
these sums do not establish critical-path latency or an overlap opportunity.
Overlap can enqueue work beyond the eventual terminal prefix, so actual DMA
bytes can exceed the committed tensor sizes. The prefill probe also includes a
small number of one-ahead decode forwards and is not a pure prefill trace.

`post_client_capture_drain_seconds` in the serving benchmark begins after the
client subprocess exits, including its result processing. It measures only the
remaining drain at that point, not latency from the last response to READY.

The retained H100 run is in
[`capture-serving-performance.json`](capture-serving-performance.json), with
scope, raw paths/hashes and two bracketed rounds of 0.1%, 1% and 10% sampling.
[`capture-profile-analysis.md`](capture-profile-analysis.md) records the source
attribution and the limits of the profiler's optimization suggestions.

Both benchmark and profiler drivers accept `--kv-d2h-batch-tokens 16
--device-mib 16` to exercise bounded KV staging. Without these arguments they
retain direct per-forward D2H. Keep sampling rates, lengths, concurrency and
bracketed capture-off phases identical when comparing runs. The batch option
does not alter the Store layout or change the amount of committed training data.
The profiler adds a separate `training_capture.kv_d2h` scope for staging flushes;
sum its D2H activity with that in `training_capture.kv`, which still covers
large prefill transfers. The normal runtime correctness test enables 16-token
staging across its AR, DSpark, graph, overlap and failure scenarios; CUDA unit
tests also retain direct-transfer coverage.
[`capture-batched-kv-d2h.json`](capture-batched-kv-d2h.json) retains the 16-token
H100 results: decode KV DMA calls fall from 7,920 to 720 with unchanged bytes;
the two normal 10%-capture phases still lose 15.5% and 16.3% throughput against
their bracketed off baselines. Reduced DMA work alone is not a service SLO.

### BF16 Rounding Isolation

The actual fixed-input parity command remains the serving gate. The following
separate diagnostic compares one retained checkpoint/fixture with the production
path, existing unified-prefix/block Triton path, reference auxiliary arithmetic
with real Triton attention, and
reference auxiliary/attention arithmetic while retaining real paged KV writes
and reads. It never writes `validation/parity.json` or certifies serving, and
blocks execution of the target decoder.

```bash
PYTHONPATH=python:/path/to/SpecForge python mooncake-study/experiments/diagnose_target_kv_parity_bf16.py \
  --checkpoint /path/to/retained-draft \
  --target-path /models/Qwen3-0.6B \
  --reference-attention sdpa
```

This distinguishes layout/checkpoint errors from numerical differences in the
same-input operators. It deliberately recomputes diagnostic attention after
the actual paged backend runs, so its timing is not production performance and
its equality result is not evidence for the unmodified attention backend.

The attention-only experiment keeps the production auxiliary arithmetic and
reads the actual paged pool. It compares PyTorch SDPA's selected implementation,
explicit math/Flash/cuDNN implementations, FP32/FP64 mathematical references,
and FlashInfer cuDNN on continuous and page-size-one KV. Each substitute is
also exercised through the complete draft backbone. Its cuDNN probes require
the pinned CUDA/cuDNN/FlashInfer environment; they are not serving backends.

```bash
PYTHONPATH=python:/path/to/SpecForge python mooncake-study/experiments/diagnose_target_kv_attention.py \
  --checkpoint /path/to/retained-draft \
  --target-path /models/Qwen3-0.6B \
  --dump-attention /path/to/attention-inputs.safetensors
```

The optional dump contains only diagnostic Q/K/V and attention outputs, not a
training export or serving certificate. The profiler records the SDPA operator
actually selected; `sdpa` alone does not identify its numerical implementation.
The fixed-input gate now includes observed SDPA operators and CUDA/cuDNN
versions in both success and failure reports. This does not alter its tolerance.

The pinned SpecForge DSpark offline example uses `flex_attention`. To compare
that training choice explicitly, run the earlier BF16 diagnostic with
`--reference-attention flex_attention`. Keep reports for different training
backends separate; one backend's result does not certify another backend.
Add `--dump-flex-code /path/to/generated-code` to save the actual Inductor
kernel source selected by those reference calls. This is diagnostic output,
not a production dependency. The diagnostic separately disables the KV-draft
kernel when comparing the legacy two-stage and unified kernels.

The KV-draft logical-order kernel is now the normal Triton serving path for
`DSparkTargetKVDraftModel`. To measure its CUDA-graph attention latency against
the old prefix/block kernel on identical paged inputs, run:

```bash
PYTHONPATH=python python mooncake-study/experiments/benchmark_target_kv_attention.py
```

The microbenchmark covers BF16, 16 query heads, 8 KV heads, dimension 128,
batch sizes 1/4/16, prefix lengths 160/1024/8192 and block widths 3/16. It
includes direct prefix/block index reads, but excludes KV writes, projections,
sampling, capture publication and HTTP scheduling. It cannot establish an
end-to-end throughput gain or the P10 capture-on/off SLO.

`target-kv-log2-probe.patch` preserves a failed numerical experiment outside the
production kernel. It changes the unified kernel's exponential base and uses
the dot-product accumulator for the running output. To reproduce it, apply the
patch in an isolated checkout, then add `--probe-log2` to the BF16 diagnostic.
The command rejects this option when the patch is absent. The patch has not
passed the full parity gate, and its causal, masked, sink and non-BF16 branches
have not been validated; it is not a deployment configuration.

Focused regressions:

```bash
python -m pytest -q \
  test/registered/unit/spec/test_dspark_target_kv.py \
  test/registered/unit/training_capture \
  test/registered/spec/dspark/test_dspark_stacked_ctx_kv_parity.py \
  test/registered/unit/managers/test_io_struct.py \
  test/registered/unit/model_executor/runner/test_decode_cuda_graph_runner.py \
  test/registered/spec/dspark/test_dspark_draft_path_default.py
```

## Fixed-Input Draft Parity

The additional reference uses SpecForge commit
`e10ea2fa3c248a4f60d636791dd71efb67338c9c1`. Stage its `specforge` package under
`$LAB/specforge-reference/` and include both `$LAB/sglang/python` and
`$LAB/specforge-reference` in each queued experiment's `PYTHONPATH`.
The SpecForge repository itself does not need modifications.

Run the small-model checks through the same resident queue:

```bash
python -m pytest -q test/registered/spec/dspark/test_dspark_target_kv_parity.py
```

These compare actual training/serving layers for three Markov head variants,
backpropagate cached CE+TV128, take an optimizer step, export the weights and
compare again after a fresh serving load. They also check future-token/KV
isolation and failed validation reports. They are test-only adapters, not the
production SpecForge training pipeline.

The real captured fixture is retained outside Git at
`$LAB/fixtures/qwen3-target-kv-parity-20261001`. It contains snapshot tensors read
back from Mooncake and the synthetic two-layer draft used by the runtime test.
It includes the target's shared-weight references, not the target decoder.
Run the explicit gate under the queue with:

```bash
python -m sglang.test.dspark_target_kv_parity \
  --checkpoint "$LAB/fixtures/qwen3-target-kv-parity-20261001" \
  --target-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B
```

`--reference-attention` accepts `eager`, `sdpa` and `flex_attention`; serving uses
Triton. `validation/parity.json` records all numerical stages and runtime/artifact
identities, and a failed comparison exits nonzero. Failed reruns cannot leave an
older successful report. Run one validation at a time per checkpoint directory.
The real BF16 gate currently fails; see `../IMPLEMENTATION_STATUS.md`. The normal
capture runtime test remains an independent, exact Store readback check.

For diagnosis only, FP32 weights/activations and native auxiliary operators can
be compared while preserving the actual paged-attention path:

```bash
python mooncake-study/experiments/diagnose_target_kv_parity_fp32.py \
  --checkpoint "$LAB/fixtures/qwen3-target-kv-parity-20261001" \
  --target-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B
```

The production RMSNorm kernel cannot execute FP32 activations; this diagnostic
substitutes native norm/activation/RoPE and disables fused in-place QK norm.
It reads only saved KV/teacher tensors and shared weights, forbids target decoder
execution, and prints its result without writing a serving parity certificate.
Passing it does not satisfy the BF16 gate or establish training quality.
