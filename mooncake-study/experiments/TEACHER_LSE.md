# Single-Pass Teacher LSE

CUDA teacher extraction now uses the existing `layers.logsumexp.row_logsumexp`
kernel for FP16/BF16/FP32 scores. It reads the unpadded selected rows with FP32
accumulation and combines the returned maximum and log-sum into the manifest's
absolute FP32 full-vocabulary LSE. Top-128 selection and int32 vocabulary IDs
remain on `torch.topk`; this does not implement fused top-128 selection. The
existing fused logprob kernel supports only small k and is not used here.

CPU and other floating dtypes retain the original Torch LSE path, including its
FP32 conversion before reduction. Nonfinite inputs are not sanitized into valid
training rows. Existing publication validation rejects them. All returned
teacher tensors retain independent storage, and operations remain on the
caller's current stream before serving-side mutation.

Serving returns FP32 logits. `warmup_teacher_capture()` initializes that path
inside local target-contract binding, including P/D prefill workers, before
admission or Store initialization. It uses one vocabulary row and synchronizes
the current device stream once during startup. Compilation/device failures
participate in the existing binding vote. Disabled capture does not run warmup.
Direct CUDA callers using other supported dtypes still initialize their kernel
variant on first use; no hot-path synchronization is added.

## Sources and Reproduction

Baseline is `323df135666b7ed0cc3676592683fc8f0fd39d15`, including publication-boundary
validation. Frozen directories on the resident H100 are:

- `/gpfs/users/fuxuanwei-1/dspark-maas-lab/sglang-teacher-lse-before`
- `/gpfs/users/fuxuanwei-1/dspark-maas-lab/sglang-teacher-lse-v1`

Use the lab capture venv, `OMP_NUM_THREADS=1`, the source's `python` directory
as `PYTHONPATH` and the [pinned runtime](h100-runtime-lock.json). The worker
pauses its idle workload and runs experiments serially. The standalone
microbenchmark was added to the final frozen directory after the unit tests;
their production/test bytes are unchanged.

```bash
python test/registered/unit/training_capture/test_teacher_cuda.py -f
python test/registered/unit/training_capture/test_cuda_snapshot.py -f
CUDA_VISIBLE_DEVICES=999 python -m unittest discover \
  -s test/registered/unit/training_capture -p 'test_*startup.py' -f
CUDA_VISIBLE_DEVICES=999 python test/registered/unit/training_capture/test_buffers.py -f
python mooncake-study/experiments/benchmark_teacher_capture.py \
  --output NEW_REPORT_PATH --source-revision SOURCE_REVISION
python test/registered/storage/test_training_capture_runtime.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B -f
TRAINING_CAPTURE_TEST_MODEL=/gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  python test/registered/storage/test_training_capture_pd.py -f
python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  --source-revision SOURCE_REVISION --output-dir NEW_OUTPUT_DIRECTORY \
  --num-prompts 512 --input-len 16 --output-len 32 \
  --concurrency 8 --capture-slots 16 --ratios 0.1 --repeats 2
```

## Numerical Coverage

Four new CUDA methods cover FP16/BF16/FP32, vocabulary sizes 128/16,385/151,936,
padding with larger logits, noncontiguous rows and columns, duplicate/reordered
row selections, empty rows and CUDA FP64 fallback. The reference sums the FP32
scores in FP64 and compares the stored FP32 LSE at `rtol=atol=1e-6`. Top-k IDs
and raw values compare exactly against Torch on the same scores.

Additional cases cover large positive/negative offsets, a dominant logit,
near-zero LSE, masked rows, all-negative-infinity, positive infinity and NaN.
Graph capture/replay checks changing inputs and source overwrite after
extraction. Existing CUDA ownership tests cover D2H, stream changes, staging
reuse, tail fencing and quarantine. A startup fault test checks that compilation
failure is observed inside the distributed binding callback before Store
connection. Existing real Gloo startup and cohort rollback tests remain active.

## Complete Extraction Microbenchmark

`benchmark_teacher_capture.py` retains the original algorithm as an explicit
reference. It measures complete top-128/LSE extraction, output conversions and
optional row selection, rather than only the new reduction. It alternates
reference/current order over three repetitions and records medians and raw
observations for all 32 cases: FP32/BF16, vocabulary 32,768/151,936, rows
1/4/32/128, with and without explicit row selection.

Eager wall time includes Python dispatch and final device synchronization.
The graph measurement records 30 calls per graph and times ten replays with
CUDA events, amortizing host submission. Neither measurement includes D2H,
Store, Catalog, model computation or real request latency. Production capture
still runs after model replay; the graph variant isolates device work for this
microbenchmark and is not a new serving capture graph.

All 32 cases pass exact top-k comparisons and the existing LSE tolerance. The
largest difference from the original FP32 LSE is 1.90735e-6. Eager median call
time is lower in every case, by 12.83%-42.83%; graph time is lower by
7.72%-44.54%. The following FP32, 151,936-vocabulary cases include explicit
row selection and are closest to the serving teacher input type:

| Rows | Reference eager us | Current eager us | Reference graph us | Current graph us |
| --- | --- | --- | --- | --- |
| 1 | 170.98 | 142.37 | 92.55 | 85.36 |
| 4 | 170.04 | 141.67 | 106.31 | 95.10 |
| 32 | 220.96 | 173.71 | 185.34 | 145.49 |
| 128 | 701.09 | 531.61 | 656.58 | 493.82 |

Startup warmup across both vocabulary sizes takes 0.789s in this process. The
shared Triton cache had already been used by the preliminary probe and unit
tests; this is not a clean-cache compilation measurement or a startup SLA.

## Regression Results

All 39 test methods pass with the final producer implementation:

| Suite | Methods | Seconds | Job |
| --- | --- | --- | --- |
| New CUDA teacher contracts | 4 | 4.933 | `01790951243920986113-f61ea11db4b5` |
| CUDA ownership and staging | 7 | 1.040 | `01790951244372725755-29dc583a1d17` |
| Binding and cohort startup | 6 | 57.172 | `01790951244743588332-cfe65da7fd54` |
| CPU teacher/buffer contracts | 18 | 0.013 | `01790951245030771209-b06a2f63a870` |
| Full actual inference/lifecycle | 2 | 720.805 | `01790951596851645501-83e8e6ac240d` |
| Actual P/D handoff/failures | 2 | 172.192 | `01790951597180394802-4312b942d2e2` |

The full runtime file includes AR, static and ragged DSpark, ordinary/graph
execution, overlap, retraction, adaptive/latency recovery and lifecycle checks.
P/D covers eager and graph/overlap source parity, missing/bad handoffs, cancellation
and post-exit Store reads. Numerical tolerances were not relaxed. These are
functional tests on one H100 with Qwen3-0.6B, local TCP and a Catalog test double;
their total duration is not a serving performance metric.

Five changed/new Python files pass Black and Ruff I/F. The four files other
than `coordinator.py` pass full Ruff; that coordinator retains its nine existing
BLE001 diagnostics, checked against the baseline. The 13 listed executable/test
files match the frozen after tree. Additional model identities, GPU platforms,
distributed teacher performance, production Catalog retention and trained draft
quality require separate acceptance.

## Serving Results

Each source completes two off/on/off rounds using 512 requests per phase,
16 input/32 output tokens, concurrency eight, 16 Host slots and 10% fixed
sampling. Target is Qwen3-0.6B BF16, with overlap and decode graphs enabled,
prefill graphs disabled, local TCP Store and an HTTP Catalog test double.
The serving driver bytes match; only `teacher.py` and `coordinator.py` differ
between the production sources. Warmup, launch, final writer drain, metrics
scraping and readback are excluded from client timing.

| Source | Round | READY | Requests/s | Output tokens/s | Throughput vs off | p99 TTFT ms | p99 TPOT ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original Torch LSE | 1 | 50 | 63.46 | 2030.75 | 72.07% | 62.24 | 3.482 |
| Original Torch LSE | 2 | 50 | 64.00 | 2048.15 | 69.38% | 59.83 | 3.460 |
| Single-pass LSE | 1 | 50 | 64.96 | 2078.83 | 70.55% | 60.70 | 3.587 |
| Single-pass LSE | 2 | 50 | 63.46 | 2030.87 | 68.79% | 62.05 | 3.501 |

Every selected request is admitted and published. All 200 measured snapshots
pass post-exit readback, covering 124,704,000 tensor bytes and 2,800 payload
objects. Four actual Prometheus scrapes match status for all four timing fields
across 13 stages. Normal stage counts equal publications, `catalog_written`
counts twice and `recovery_read` is zero. There are no measured stage errors,
Catalog failures, admission backpressure or quarantined slots.

Across both capture-on phases, request throughput is 63.73 versus 64.21/s and
output throughput is 2039.41 versus 2054.57 tokens/s, an observed difference of
0.74%. Both sources capture 100 samples and the same payload byte count. This
does not establish an end-to-end serving improvement: results overlap, tails
vary, and capture-off throughput drifts by 11.19% across the first baseline
bracket. Capture-on retains only about 69%-72% of off throughput in this test.
The microbenchmark improvement is established for its measured cases; the
serving SLO and cause of remaining overhead remain open.

## Retained Evidence

The [evidence index](teacher-lse.json) binds source/runtime hashes, all ten
successful terminal jobs (including the preliminary probe), the standalone
microbenchmark, both serving reports, all four metrics scrapes and logs/results.
The final tested source matches the checkout. The resident H100 returned to its
60% idle workload with no active or queued experiment; no additional GPU was
allocated. The original checkout's staged index is unchanged.

Production target identity and TTFT/TPOT/throughput acceptance thresholds still
need to be specified. The small model, synthetic tokens and short sequential
repetitions do not certify a production deployment, clean-cache startup latency,
distributed performance, live dashboard acceptance or trained draft quality.
