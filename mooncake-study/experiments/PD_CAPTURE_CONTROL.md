# P/D Live Capture Control

This validation exercises the existing `POST /control_training_capture` endpoint
on separately running prefill and decode services. The target remains
Qwen3-0.6B. Both services share the resident H100 and use actual Mooncake TCP
transfer and Store APIs, an independent Store data segment, and the test Catalog.
The serving implementation remains the one introduced in `9fb873039`.

## Cases

| Boundary | Commands and Expected Outcome |
| --- | --- |
| Both endpoints idle | Pause D, pause P; generation succeeds without a capture |
| Ordered resume | Resume P while D stays paused; no new teacher/capture, then resume D and publish a fresh sample |
| Partial prefill | Pause D and P after the first chunk; the admitted sample completes handoff and publishes |
| P abort before handoff | Abort/resume P before releasing the chunk; the old epoch sends no teacher, D fails only the sample |
| D abort before handoff | Abort/resume D before releasing the chunk; P sends a valid late teacher payload, D does not accept it |
| Both endpoints abort | Abort both, resume P then D; old capture remains failed, fresh capture succeeds |
| P paused, D running | D's selected sample fails due to missing teacher, while the generation request succeeds |
| Decode already running | Abort D after teacher import; generation finishes all 200 output tokens and only the capture fails |

Each case follows the real HTTP/scheduler route. The test checks a successful
fresh publication after every abort mode. Capture control never calls the
ordinary request-abort endpoint in this test.

Every inference request forces token ID 100 with the existing logit-bias sampling
option and checks the full requested output length. Independent target observers
retain the raw, pre-bias scores and canonical KV source rows. For published
samples, readback checks exact tokens, response masks, absolute positions, valid
KV rows, raw top-128 IDs/values and KV values. LSE uses the existing FP32 comparison
tolerance of `rtol=atol=1e-6`. Both producers exit before a new Store reader checks
the published data. An absent/corrupt or partially collected sample cannot pass
the manifest and tensor checks.

## Deterministic Handoff Boundary

`sglang.test.pd_capture_control_server` is an isolated test entrypoint. For tagged
200-token prompts it lets the first 128-token chunk finish, then returns an empty
prefill plan until a filesystem gate is released. The scheduler keeps processing
real HTTP control messages during this interval. A 60-second timeout bounds a
forgotten gate, and the test releases it in `finally`.

The gate records the live request's capture selection and partial prompt end.
The final handoff observer records capture/current epochs and whether a real
teacher payload exists. These observations distinguish a valid late payload
discarded by D from a payload that P never produced after its own abort.
Production modules do not contain this gate and are not modified by the tests.
The entrypoint explicitly rejects TP/PP sizes other than one.

The additional CPU test starts with an already materialized teacher tensor, or
its already encoded PP handoff, then aborts and resumes P. Neither representation
may be exported or accepted under the new epoch. Thus the partial-prefill runtime
case is supplemented by a check after teacher materialization and serialization.

## Scope

The runtime matrix covers ordinary autoregressive decode and static KV-input
DSpark, each in eager/synchronous and decode CUDA graph + overlap modes. DSpark
uses the repository's synthetic KV-input checkpoint exporter; its purpose here
is execution/ownership coverage, not training-quality or acceptance-rate evidence.
Prefill graphs are disabled in this test.

This lane does not prove multi-GPU TP/PP control, cross-node management or RDMA
control, confidence-scheduled DSpark, a global drain acknowledgement, production
Catalog retention, or deployment SLOs. The test observers deliberately copy raw
tensors and write reference files, so these timings are not serving benchmarks.
The route's authentication was tested separately in
[the initial operator-control run](CAPTURE_CONTROL.md).

## Results

The final matrix passes all four methods in **433.711 seconds** on the resident
H100. Each cell completes 15 ordinary generation requests, publishes eight
captures and reads all eight after both producers exit. Across the matrix this
is **60 generation requests and 32 validated snapshots**. The synthetic draft's
setup sample is excluded from these counts.

| Decode | Execution | Validated Snapshots | Capture Graph / Overlap Forwards | Speculative Verify / Commit Copies |
| --- | --- | --- | --- | --- |
| AR | Eager | 8 | 0 / 0 | 0 / 0 |
| AR | Graph + overlap | 8 | 137 / 137 | 0 / 0 |
| Static target-KV DSpark | Eager | 8 | 0 / 0 | 35 / 35 |
| Static target-KV DSpark | Graph + overlap | 8 | 43 / 43 | 43 / 43 |

Each D endpoint records 15 considered requests, 13 admitted captures, eight
READY samples, three `operator_aborted` failures and two `pd_handoff_failed`
failures. The two paused requests are excluded. P records 12 selections and ten
teacher copies/handoffs; P's two aborts suppress their old handoffs. All four
cells finish without admission backpressure, quarantined buffers, Catalog
errors or writer-stage errors. Pausing retains spare Host arenas and leases, as
specified by the control API; it does not unload capture resources.

The CPU P/D suite passes **15 methods in 9.725 seconds**, including the new
materialized/encoded teacher epoch test. Together with the final matrix this is
**19 unique methods**. The earlier two-method AR-only run also passed in
183.962 seconds; it is an interim run, not an additional matrix or dataset count.

- Final matrix: `01790958592873053017-48078a2b1409`.
- CPU P/D: `01790958218127326389-a0a47f10e371`.
- Interim AR-only: `01790958129062045441-df8d934bc1ac`.

All three worker jobs are terminal with return code zero. The final matrix used
the frozen `sglang-pd-control-v2` mirror. The evidence index records matching
hashes for the three changed Python files and nine direct runtime dependencies.
No additional GPU was allocated, and the resident worker resumed its idle load.

## Reproduction

Use a local Qwen3-0.6B model and the locked capture runtime, with CUDA and
`mooncake_master` available:

```bash
export PYTHONPATH="$PWD/python"
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
  python test/registered/storage/test_training_capture_pd_control.py -v
python test/registered/unit/training_capture/test_pd_capture.py -v
```

The runtime test launches real P and D processes for each matrix cell. The test
process owns the Store data segment independently of P/D and keeps it alive
while the producers terminate and a new client reads back their publications.
Commands, terminal results, source hashes and collected per-mode state are retained
in [the evidence index](pd-capture-control.json).
