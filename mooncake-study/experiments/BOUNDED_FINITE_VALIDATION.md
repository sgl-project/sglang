# Bounded Host Payload Finiteness Validation

The mandatory snapshot validator scans every floating payload before publication
or consumption. The previous `torch.isfinite(tensor).all()` path was the largest
individual function in a CPU profile of a synthetic 512-prompt/128-response
snapshot: 0.638 of 1.422 seconds across 100 full validations, before counting
the final Torch reduction. The profile includes hashing and all semantic checks.

## Implementation

The validator now reuses the same contiguous little-endian Host byte view used
for SHA-256. BF16, FP16 and FP32 values are viewed as unsigned 16/32-bit integers;
an all-ones exponent identifies both infinities and all NaN encodings. Sign,
mantissa and subnormal values do not require floating-point conversion. This
does not change tensor contents, their dtype or the manifest schema.

The scan operates on at most 262,144 elements at a time. The bitwise result and
comparison require at most 1.25 MiB of element storage for FP32, independent of
the tensor's total size. The byte and integer views share source storage.
This bound covers the new finite scan, not SHA-256, other semantic checks,
retained payloads or the full writer's memory usage.

Shape, dtype, device, contiguity, checksums, coverage, positions, masks, token
ranges, top-128 uniqueness/order and full-vocabulary normalization checks still
run at their existing boundaries. An invalid snapshot cannot register objects
or publish. Source ownership and Catalog fencing remain unchanged.

## Correctness Coverage

The protocol tests compare every BF16 and FP16 bit pattern against Torch's
finite classification. FP32 tests add explicit signed-zero, subnormal, maximum
finite, infinity, signaling/quiet NaN and random bit patterns. Large arrays
exercise both sides of chunk boundaries and the final short chunk, assert the
source is unchanged and bound the submitted chunk sizes.

Full manifests with valid checksums but nonfinite K, V, raw logits or LSE must
still fail semantic validation, for both supported KV dtypes. Existing writer,
coordinator and context tests cover cancellation during validation, source
lifetime, publication faults and recovery.

```bash
python test/registered/unit/training_capture/test_protocol.py -f
python -m unittest discover -s test/registered/unit/training_capture \
  -p 'test_*writer.py' -f
python -m unittest discover -s test/registered/unit/training_capture \
  -p 'test_*coordinator.py' -f
python -m unittest discover -s test/registered/unit/training_capture \
  -p 'test_*context.py' -f
```

## Serving Comparison

Use the pinned capture venv and identical benchmark driver on the resident H100.
The baseline production source is commit `8fc5b3815`. Candidate source changes
only `training_capture/protocol.py`; the additional protocol tests do not run
inside serving. Freeze each tree before submission and set `PYTHONPATH` to its
`python` directory, with `OMP_NUM_THREADS=1`.

```bash
python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  --source-revision SOURCE_REVISION --output-dir NEW_OUTPUT_DIRECTORY \
  --num-prompts 64 --input-len 512 --output-len 128 \
  --concurrency 4 --capture-slots 4 --ratios 1.0 --repeats 1

python test/registered/storage/test_training_capture_runtime.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  TestTrainingCaptureRuntime.test_chunk_prefix_single_token_and_raw_teacher_reference -f
```

Each benchmark uses an off/on/off bracket, a local TCP Mooncake Store and the
HTTP test Catalog. Client measurement excludes warmup, final capture drain and
post-producer readback. Every measured READY sample must pass readback, and the
real metrics endpoint must agree with writer stage status. The independent
inference regression compares captured tensors with online sources across AR,
overlap, graph replay, target-KV DSpark and lifecycle/admission scenarios.

An isolated finite-scan speedup does not establish serving improvement. Report
admitted/READY samples together with latency and throughput: faster writer slot
reuse can admit more captures and increase total collection work. These short
synthetic runs do not certify production SLOs, long-duration memory stability,
production Catalog retention or trained draft quality.

## Measured Results

Both microprofiles use the same script and CPU thread limit. Full validation
of the synthetic 5 MiB KV snapshot takes 1.422 seconds before and 0.662 seconds
after across 100 profiled calls, approximately 53.4% lower. SHA-256 remains the
largest measured function after the change. These profiled durations include
profiling overhead and do not describe service latency.

The real-model baseline and candidate each complete one off/on/off bracket.
Every phase contains 64 requests with 512 input and 128 output tokens. The
capture-on results are:

| Source | READY / 64 | Writer validation ms/sample | Samples/s | Requests/s | Throughput vs off | p99 TTFT ms | p99 TPOT ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 61 | 26.811 | 5.144 | 5.397 | 75.24% | 666.606 | 3.562 |
| Bounded finite scan | 64 | 13.257 | 5.372 | 5.372 | 74.68% | 674.106 | 3.517 |

Mandatory writer validation is 50.56% lower per sample. Sample throughput rises
4.43% while request throughput is 0.47% lower. All 125 admitted measured samples
reach READY and pass post-producer readback: 8,500 tensor objects and
1,000,672,000 payload bytes. There are no measured stage errors, recovery reads,
Catalog errors or quarantined Host slots. Both metrics scrapes agree with stage
status. Both versions allocate 38,329,344 registered Host bytes and no device
staging.

The measured improvement is lower validation work and full admission in this
candidate run, not an established end-to-end serving speedup. Both capture-on
phases remain near 75% of their bracketing off throughput. This single sequential
pair does not separate run-order effects or establish a production SLO.

## Regression And Evidence

All 114 protocol, writer, coordinator and context test methods pass in 56.490
seconds. The real-model inference method passes in 534.308 seconds, covering
ordinary AR, graph replay, two overlap modes, four static target-KV DSpark
combinations, four DSpark memory-pressure cases, adaptive/latency admission and
two cache lifecycle modes. Its existing source comparisons and post-exit Store
reads remain enabled. These cases use an untrained draft fixture.

The [evidence index](bounded-finite-validation.json) binds six successful worker
jobs, both benchmark reports and metric scrapes, raw microprofiles, test logs,
source and cleanup. All 5,031 Python files match the frozen candidate; the only
Python changes from baseline are the protocol implementation and its tests.
Ruff introduces no new diagnostics: the protocol retains its two existing
findings, and the protocol tests are clean. Both files pass formatting checks.

No additional GPU is allocated. Serving and Store processes have exited, and
the resident worker has an empty queue with its idle task live. The initial
archive copy could not enter service-owned private journal directories. Direct
container inspection confirms each contains only `producer.lock`, with no
pending publications; those directories are excluded from the final archive.

The 63-artifact archive is
`/gpfs/user/fuxuanwei/mooncake-lab-archive/finite-validation-20261003`, with
artifact manifest SHA-256
`bdaa3ca74db82b22b1457e925c3e4c86cfbb4a9ee0a801dfd90d670010422561`.
