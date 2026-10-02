# Live Training Capture Control

The Python HTTP server can pause new capture, resume admission, or abort collecting
samples while target generation continues. The change implements a producer
rollback mechanism in P10; production SLO, fleet rollout and training quality
acceptance remain open.

## Interface

Start the service with its existing `--training-capture-config`. Use its management
credentials and address:

```bash
curl -sS -X POST http://127.0.0.1:30000/control_training_capture \
  -H "Authorization: Bearer $SGLANG_ADMIN_API_KEY" \
  -H 'Content-Type: application/json' -d '{"action":"pause"}'
```

The other actions are `resume` and `abort`. The route uses `ADMIN_OPTIONAL`, just
like other SGLang management routes: the administrator key takes precedence when
configured; otherwise the normal API key applies. A deployment with neither key
has an open management endpoint. Invalid/missing actions return HTTP 400; absent
or incorrect configured credentials return HTTP 401. Capture must already be
configured at startup; this API does not create a producer dynamically.

`success` reports command handling. `results` contains the replying scheduler's
`success`, `message`, and capture `state`. A server without capture returns
`success=false` and HTTP 400. The new IPC structs follow the existing array-based
msgspec wire format; the HTTP body is the usual object-shaped request.

| Action | New admission | Collecting samples | Background publication |
| --- | --- | --- | --- |
| `pause` | Paused | Continue | Continue |
| `resume` | Allowed subject to original gates | Continue | Continue |
| `abort` | Paused | Invalidated | Already handed-off snapshots may publish |

Manual pause has its own flag. Resume cannot clear a model-identity failure,
transport failure, publication recovery, adaptive cooldown or latency gate. It
does not change sampling configuration or revive an aborted context. A request
skipped while paused remains excluded after resume; an already-issued distributed
ticket is allowed to bind and drain during pause.

Abort invalidates collecting contexts and unbound cohort reservations. Request
generation and accepted token output remain intact. The existing writer/service
protocol waits for D2H completion before recycling memory and preserves uncertain
publication recovery. Aborting collection is not a transaction that revokes
already-published data. Catalog retention, hard pins and GC remain unchanged.

Both producer variants stop reservation refill while manually paused. Cohorts use
the already-voted readiness from every rank; no new inference-thread collective
or Catalog/Store call is introduced. Existing lease renewals continue, and an
allocation already in progress can finish. Spare arenas/leases can remain held.

## Deployment Boundaries

The existing scheduler communication forwards control through TP/PP. HTTP replies
come from the usual replying scheduler, not from a new all-rank barrier. Do not
interpret one reply or zero local active contexts as proof of global drain or
completed Store cleanup. Inspect per-rank metrics and Catalog terminal states.
The final PP stage remains the auxiliary publication owner.

For P/D, pause D first, allow existing P handoffs to drain, then pause P if needed.
Resume P before D. An immediate abort sends the action to both endpoints. P stores
a local abort epoch on each capture state so old teacher rows remain invalid after
a quick resume. This is not an atomic fleet-wide operation. The external router,
Rust-native management server and production rollout controller are not extended
by this change.

State and Prometheus expose `admission_paused` separately from `disabled_reason`.
The updated dashboard shows both. Event labels for operator actions are bounded;
no request IDs or caller-supplied reason strings become metric labels. Dashboard
JSON is checked locally; a live Grafana deployment was not exercised.

## Validation

Environment: the existing single H100 allocation, Qwen3-0.6B, the locked capture
runtime, Mooncake TCP with an independent data segment, and a test HTTP Catalog.
No additional GPU was allocated. Results do not establish production latency or
multi-node control behavior.

- 103 CPU test methods pass across coordinator, cohort coordinator/service,
  P/D capture, request routing and metrics. Both immediate and staged copying are
  covered. The four-process Gloo suite adds all-unbound and mixed-bound operator
  cancellation, preserving ownership until every actor drains.
- The new live HTTP test passes in eager and CUDA graph + overlap modes. In each,
  two requests generate while admission is paused, one active capture drains
  through pause, a second active capture is aborted without changing output, and
  a fresh request publishes after resume. Missing/wrong management credentials
  and malformed actions leave capture state unchanged.
- Each HTTP mode admits four captures, publishes three, and fails exactly one
  with `operator_aborted`. All six published snapshots pass manifest/tensor
  validation and exact token/mask readback after producer exit. Quarantine is zero.
  The graph run records 307 graph forwards and 311 overlap forwards.
- The existing two real P/D tests pass: eager and graph + overlap, handoff faults,
  cancellation, source parity and five post-exit Store snapshots each. Live HTTP
  controls on separate P/D endpoints and real multi-GPU TP/PP control still need
  dedicated integration coverage; unit/control-protocol evidence is not that gate.

Unit job: `01790955971085003936-cd670ad63a29` (80.134 s).
Eager HTTP: `01790956196564215278-85cff0a67a54` (49.126 s).
Graph/overlap HTTP: `01790956196894794094-f3338265b007` (36.576 s).
P/D regression: `01790956197235188513-ab4993c662d3` (173.327 s).

Early runs exposed missing readiness gating on cohort reservation refill, fixed
before these passing runs. Other initial failures were in validation setup:
unittest module paths, expecting HTTP 422 instead of the server's HTTP 400,
iterating the Catalog's publication dictionary instead of its returned values,
and using an object-shaped request for an array-based IPC schema check. All
terminal attempts are retained in the [evidence index](capture-control.json).

The functional runs use immutable `sglang-capture-control-v3`/`v4` source mirrors.
After them, only equivalent typing annotations changed in the IPC response,
tokenizer helper and test header declaration. The final mirror has a separate
IPC roundtrip and unconfigured-scheduler check. Hashes distinguish these sources;
no claim is made that earlier test bytes equal later annotation-only edits.

## Reproduction

With the runtime dependencies, local model and `mooncake_master` available:

```bash
export PYTHONPATH="$PWD/python"
python -m unittest discover -s test/registered/unit/training_capture -p 'test_coordinator.py' -v
python -m unittest discover -s test/registered/unit/training_capture -p 'test_cohort_coordinator.py' -v
python -m unittest discover -s test/registered/unit/training_capture -p 'test_cohort_service.py' -v
python test/registered/storage/test_training_capture_control.py --model-path /path/to/Qwen3-0.6B -v
python test/registered/storage/test_training_capture_control.py --model-path /path/to/Qwen3-0.6B --cuda-graph-overlap -v
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B python test/registered/storage/test_training_capture_pd.py -v
```

The worker commands and terminal results in the evidence index identify the
actual checkout paths, environment and additional CPU modules used in this run.
