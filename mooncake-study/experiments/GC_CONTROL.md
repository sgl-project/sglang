# Serving GC Ownership And Controlled Intervention

The [clock diagnostic](LATENCY_DIAGNOSTICS.md) reproduced generation-two
tokenizer collections of roughly half a second. This experiment exercises the
existing `/freeze_gc` operation after real warmup, and checks request lifetime
without enabling a new serving default.

## Ownership Repair

Previously, graph capture always called `gc.unfreeze()` on leaving its temporary
freeze. Python's permanent generation is process-wide, so this also revoked a
freeze requested through the serving API. Nested and overlapping graph scopes
could revoke another active scope's freeze.

`sglang.srt.utils.gc_control` now coordinates the existing serving and graph
helpers. The serving API records a permanent owner; temporary graph scopes have
a shared depth and release their freeze only when the last scope exits and no
serving owner exists. An API request during capture promotes the freeze. A
reentrant lock serializes transitions, not the graph-capture body. The existing
graph helper remains the common entry point for decode and prefill runners.

An existing serving freeze is preserved without freezing new request objects
on every subsequent graph capture. Entry collection still cleans the ordinary
generations. GC enablement and thresholds are unchanged. Explicit repeated API
requests still freeze the then-current tracked objects, as before.

This coordinates SGLang's two freeze users, not arbitrary third-party calls to
`gc.freeze()` or `gc.unfreeze()`. There is no serving unfreeze API. Restart the
process to reset its serving-freeze policy. A nonzero `get_freeze_count()` alone
does not identify the owner: the pinned runtime reports a small permanent
generation even without a SGLang serving freeze. Regression assertions use
actual cyclic-object weak references instead.

## Reproduction

Use the resident worker, matching source `PYTHONPATH`, and
`h100-runtime-lock.json`. Independent processes are required for the ownership
tests because GC's permanent generation is global.

```bash
PYTHONPATH=python:test/registered/unit/model_executor/runner \
  python -m unittest test_graph_gc_ownership test_hidden_state_graph_recapture \
  test_graph_gc_cuda -v

python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /models/Qwen3-0.6B --source-revision CHECKOUT_REVISION \
  --output-dir NEW_DIRECTORY --num-prompts 1024 \
  --input-len 16 --output-len 32 --concurrency 8 --capture-slots 16 \
  --ratios 0.1 --repeats 1 --request-details --latency-diagnostics \
  --gc-lifecycle --gc-policy freeze-after-warmup
```

Run the same command with `--gc-policy unchanged` and a fresh output directory
for the control. Use identical source and instrumentation for both policies.
Each invocation starts fresh off/on/off serving processes. Client timing excludes
warmup, policy changes, lifecycle probes, capture drain and post-exit readback.
Maximum TTFT and request counts above 500ms accompany p99, since a stall that
affects fewer than one percent of requests can be invisible in that percentile.

## Acknowledgement And Lifecycle Scope

After warmup and capture drain, the driver calls the native `/freeze_gc` route.
Its HTTP response only acknowledges tokenizer handling. The experiment waits
for the existing completion logs from tokenizer, scheduler and detokenizer
before measuring; missing acknowledgement fails the experiment.

Only the test-only diagnostic entrypoint, with the explicit
`SGLANG_TEST_CAPTURE_GC_LIFECYCLE=1` environment variable, installs
`/test_training_capture_gc_state`. It records tokenizer RSS, GC state, active
request count and bounded weak references to real `ReqState` instances. The
driver requires complete measured-request coverage and rejects dropped weak
references. No production HTTP route or per-request tracker is added.

The probe observes before and after policy application and after the timed
workload. A final explicit collection checks that a newly created cycle is
collectable and completed request states do not exceed the premeasurement live
baseline. It also verifies process identity, freeze ownership, GC enablement and
thresholds are unchanged. `gc-lifecycle.json` is written before assertions to
retain failing observations. These checks do not measure scheduler/detokenizer
object lifetime or establish long-duration memory stability; RSS is an
observation, not an asserted leak-free bound.

The CUDA regression performs three real captures and six replays after serving
freezes GC, with exact output checks and old/new cyclic-object lifetime checks.
The producer resource suite additionally runs after a permanent serving freeze
to exercise abort, transfer fencing, quarantine and explicit buffer release.
Those unit cases do not represent an HTTP cancellation workload.

## Results

Both policies use the same v2 source and instrumentation. Each row measures
1,024 requests; the unchanged bracket runs before the frozen bracket.

| Policy | Capture | Requests/s | p99 TTFT ms | Maximum TTFT ms | In-Flight Tokenizer GC ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Unchanged | Off | 88.506 | 98.326 | 494.165 | 448.498 |
| Unchanged | 10% | 74.919 | 61.722 | 495.003 | 449.379 |
| Unchanged | Off | 88.935 | 57.740 | 477.947 | 432.519 |
| Freeze after warmup | Off | 93.175 | 46.847 | 88.752 | None >=20ms |
| Freeze after warmup | 10% | 77.981 | 55.693 | 66.792 | None >=20ms |
| Freeze after warmup | Off | 92.801 | 44.486 | 63.227 | None >=20ms |

Each unchanged phase's generation-two tokenizer collection overlaps requests
176-183. The frozen phases have no observed >=20ms tokenizer/scheduler GC or
client GC/timer delay overlapping requests. No phase exceeds 500ms TTFT in this
pair, including the controls with 433-449ms collections: a fixed threshold alone
would miss the reproduced stall. The prior uninstrumented spike remains
unattributed; the lifecycle tracker also changes allocation history compared
with earlier diagnostic versions.

The freeze intervention removes the observed GC pause for this workload. It
does not eliminate capture overhead: capture-on throughput is **84.44%** of
its surrounding off average with unchanged GC, and **83.86%** with frozen GC.
Raw capture-on throughput rises 4.09%, but off throughput also rises. One pair
without reversed order does not establish a significant capture optimization
or a production SLO.

All nine phases, including a separate 64-request-per-phase frozen smoke,
have zero active requests and zero live tracked request states after drain and
explicit collection, complete weak-reference coverage, and collectable new
cycles. Across the 1,024-request phases, tokenizer RSS growth from post-policy
to post-collection is 0.465-1.043 MiB unchanged and 0.066-0.855 MiB frozen.
These are short-run observations, not a leak-free memory bound.

The three experiments complete **6,336 requests / 202,752 output tokens** and
validate **248 complete snapshots / 154,632,960 payload bytes** after producer
exit. The paired 10% phases each validate 92 samples; the 100% smoke validates
64. Three Prometheus scrapes match capture stage status. There is no measured
capture failure, writer stage error or quarantine. Offline replay reproduces
all nine request summaries and pause correlations from hashed artifacts.

There are **164 passing test methods**: 13 ownership/graph-selection methods,
25 CUDA/diagnostic/native-streaming methods, and 126 producer resource methods
run under an explicit serving freeze. The corrected pre-fix regression fails
three of four ownership cases. An earlier probe with invalid zero-permanent-
generation assumptions is retained separately and is not counted as proof of
four bugs. The ownership class is AST-identical between tested v1 and final v2;
only formatting and its CI runtime estimate change.

All eight worker jobs are terminal: the two baseline probes fail as recorded,
and all six candidate jobs pass. The H100 returns to its idle workload. Seven
Python files pass Black, five pass full Ruff, and the two touched upstream files
have exactly their baseline's 130 and seven Ruff diagnostics, with no new
findings. Source, commands, results and artifacts are bound in
[gc-control.json](gc-control.json). Production SLO, Catalog retention,
SpecForge integration and trained-draft quality remain separate requirements.
