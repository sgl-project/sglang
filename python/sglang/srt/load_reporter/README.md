# SGLang Embedded Load Reporter

## Overview

The load reporter is a **Worker-process-internal gRPC service** that listens on
its own fixed port (`--load-reporter-port`) and streams scheduler load snapshots
to an external Router. It is not a separate deployment, a discovery service, or
a reuse of the HTTP / inference port — it is one more service owned by the
Worker process.

Transport direction: the **external Router dials INTO** the Worker's reporter
port and drives a single bidirectional gRPC stream
(`LoadMonitorService.Monitor`). The Router sends a `RegisterRequest` first; the
Worker replies with an ack immediately, then the reporter's fire loop performs
one bounded snapshot pull and sends the first `LoadReport`. A successful pull
therefore makes the first report a completed current snapshot; a hung or
invalid pull produces an explicit `UNREACHABLE` report after the bound.
Periodic reports are then broadcast on the negotiated deadlines, anchored at
each session's registration time (not at first-report completion).

The reporter is **opt-in and disabled by default**. When `--load-reporter-port`
is unset there is zero overhead: no socket, no task, no binding, and the
optional `grpc`/`protobuf` stack is never imported.

> **Router scope.** This component only delivers the Worker-side reporter and
> its wire contract (canonical proto + fixed-port config). The external Router
> client — its addressing/discovery, retries, registry reconciliation, and any
> load-aware routing policy — is delivered separately by the Router owner and is
> **not part of this PR**. Tests here validate the Worker wire contract with a
> real `grpc.aio` fake Router; that is not a claim that the production Router is
> implemented. Keep the reporter disabled in production until that client ships.

## Serving-mode matrix

Every mode shares one reporter service / runtime / proto / snapshot source.
Only the startup site and the snapshot source differ.

| Serving mode | Reporter start site | Snapshot source | Snapshot fires |
|---|---|---|---|
| HTTP | FastAPI `lifespan` | `TokenizerManager` | registration + negotiated deadlines |
| native gRPC (`--grpc-port`) | reuses the same FastAPI lifespan (no second listener) | `TokenizerManager` | registration + negotiated deadlines |
| multi-tokenizer HTTP (`--tokenizer-worker-num > 1`) | sole `MultiTokenizerRouter` owns the port | Router shared-memory snapshot reader | registration + negotiated deadlines |
| standalone SMG RPC (`--smg-grpc-mode`) | `grpc_server.py::_on_request_manager_ready` (`start_load_reporter`) | `GrpcRequestManager.get_loads(include=["core"])` | registration + negotiated deadlines |

> **Multi-tokenizer native gRPC is not supported.** `ServerArgs` rejects
> `--grpc-port` together with `--tokenizer-worker-num > 1`, so the reporter does
> not claim that combination. Multi-tokenizer applies to HTTP only.

**All serving modes use request-independent snapshot fires.** The reporter owns
exactly one timer: it wakes at the earliest report deadline (or lease expiry)
across registered Router sessions, performs one bounded pull, and broadcasts
the resulting report to every session whose deadline has fired. Request
dispatch, completion, and abort events have no edge into this graph. Topology
changes only update the expected DP-rank set; the next fire observes it.

## Push-channel semantics

- **One timer.** `LoadReporterRuntime` owns a single fire-loop task. Sessions
  are passive bookkeeping (deadline, lease, queue); they own no task and no
  timer.
- **One pull per fire.** Each fire reads the Scheduler snapshot exactly once
  (with one bounded retry when the expected DP-rank set changed mid-pull) and
  builds exactly one report.
- **Broadcast to due sessions.** Every session whose deadline has fired at the
  fire receives the same report simultaneously. When all Routers negotiate the
  same interval, every fire broadcasts to every registered Router — a pure push
  channel anchored by the first registration. When sessions differ, each still
  receives only at its own negotiated deadline while the shared pull runs at
  the union of deadlines.
- **Coalesced registration.** Sessions registered before the next fire share
  that fire's pull for their initial report.
- **No persistent store.** There is no latest-snapshot store and no
  cross-report merge: a report contains only the ranks returned by its own
  pull attempt.

## Freshness model

End-to-end snapshot freshness is bounded by the Scheduler snapshot publication
path (`load_snapshot_publish_interval`), not by the gRPC delivery cadence.
The report interval controls how often a report is sent; it does not control
how old the snapshot data inside that report is. A pull at the same cadence
can observe data of the same age because both consume the same
Scheduler-published snapshots.

What the push stream provides over polling:

- **Negotiated delivery cadence.** Each Router session picks its own report
  interval; the Worker delivers reports on that schedule.
- **Connection reuse.** One persistent bidirectional gRPC stream per Router
  session, not per-request connections.
- **Multi-Router fan-out.** Multiple Routers can register independently; the
  fire loop shares one pull per deadline across all due sessions.
- **Leases.** Each session has a TTL; the Worker stops reporting when the
  lease expires, preventing stranded streams.
- **Bounded backpressure.** Each session queue is capacity-1, latest-wins;
  a slow Router never accumulates historical reports.

## Architecture

```text
Scheduler
  -> existing SHM/ZMQ latest LoadSnapshot publication
  -> reporter single fire timer (min next deadline across Router sessions)
  -> one bounded snapshot_source.get_loads() per fire
  -> validate this pull's complete DP-rank set (one retry on rank-set change)
  -> build one LoadReport per fire (no previous-report merge)
  -> broadcast to every session due at this fire (capacity-one queues)
  -> bidirectional gRPC streams
```

Request execution has no edge into this graph. Starting, completing, aborting,
or cancelling a request neither pulls snapshot data nor changes a report
deadline.

Non-request control-plane events may still wake the fire loop when required
for correct lifecycle behavior:

- registering or replacing a session;
- applying an interval or lease update so the timer uses the new schedule;
- changing the expected DP rank set after elastic scaling (no immediate pull);
- shutting down the reporter.

These events are bounded by reporter/session lifecycle changes rather than
request volume.

## Composition root

`start_load_reporter(server_args, snapshot_source) -> Optional[LoadReporterHandle]`
is the single serving-mode-agnostic entry point. Serving entrypoints only ever
see the returned handle's `close()` and `update_expected_dp_ranks()`; no
reporter-internal type leaks into them.

- `load_reporter_port is None` → returns `None` before importing grpc/protobuf.
- an enabled reporter requires a non-`None` `snapshot_source`; multi-tokenizer
  HTTP workers do not call the composition root, and only the
  `MultiTokenizerRouter` process owns the port and runtime.
- otherwise → owns a `LoadReporterRuntime` + a `grpc.aio` server on
  `host:load_reporter_port`.

`LoadReporterHandle.close()` is idempotent and tears down in order: stop the
gRPC server, close the runtime.

`LoadReporterHandle.update_expected_dp_ranks()` propagates elastic topology
changes to the snapshot source. It does not pull; the next fire observes the
new rank set.

## Network and deployment model

- The reporter's network boundary is `--load-reporter-port` on
  `server_args.host`. The Router derives the Worker host from the Worker URL it
  already knows and connects to a **paired fixed reporter port** it is
  configured with — it does not discover the port via `/server_info`, gRPC info,
  or port scanning.
- **Non-host networking assumption:** in the common case each Pod/container has
  its own IP, so many Workers can reuse the same reporter port number.
- **host-network is out of scope:** multiple Workers sharing a host's network
  namespace must be given distinct ports by deployment config; this PR does not
  implement dynamic allocation or discovery.
- **h2c only:** the reporter uses `grpc.aio` insecure (h2c) — no TLS, mTLS, or
  gRPC authentication.
- A bind failure on the fixed port fails explicitly; there is no silent
  fallback to a random port.

## Protocol

Canonical IDL: `proto/sglang/router/loadmonitor/v1/load_monitor.proto` (package
`sglang.router.loadmonitor.v1`). Regenerate the Python bindings with `protoc`
(grpcio-tools) into `python/sglang/srt/load_reporter/proto/`. It is the single
wire-contract input for the external Router; do not copy it or its generated
artifacts into Router-side test fixtures.

```protobuf
service LoadMonitorService {
  rpc Monitor(stream RouterFrame) returns (stream WorkerFrame);
}
```

- `RouterFrame` = `register | update_config | keep_alive | stop`. The first
  frame MUST be `register`; any other first frame yields
  `WorkerFrame(error=StreamError(code="INVALID_FIRST_FRAME"))`.
- Registration timing must be positive and `router_id` must be non-empty.
  `update_config` distinguishes absent fields from explicit values; every
  present timing field must be positive and starts a new deadline when the
  Worker accepts the frame. Invalid input terminates the stream with
  `StreamError(code="INVALID_ARGUMENT")`.
- `WorkerFrame` = `registered | report | error`. On valid register the Worker
  sends the ack immediately, then a bounded first-fire report, followed by
  periodic `LoadReport`s on the negotiated deadline.
- Same `router_id` re-registering on a new stream replaces the old session;
  different `router_id`s coexist. Each session's response queue is capacity-1,
  latest-wins, so a slow Router never accumulates historical reports.
- EOF, cancel, `stop`, lease timeout, and server shutdown all run the same
  idempotent cleanup.
- `LoadReport.worker` remains compatibility metadata; the Router should
  associate a report with the Worker identity of the outbound task, not trust
  `worker_addr`.

**Report status** describes the single attempt that produced the report:

- `HEALTHY`: this attempt returned a complete valid rank set and every rank
  timestamp is within the stale threshold.
- `STALE`: this attempt returned a complete valid rank set but at least one
  rank timestamp is older than the threshold; ranks are still included.
- `UNREACHABLE`: this attempt timed out, raised, decoded invalid data, or
  returned an incomplete rank set; ranks are empty and `last_error` explains
  the attempt failure. No historical ranks are substituted.

A rank timestamp that regresses is forwarded and evaluated as stale — it is
never replaced with historical data. A rank whose Scheduler timestamp is
absent or non-positive is reported with the pull-completion wall clock and
therefore evaluates as fresh; freshness claims are only as strong as the
published timestamp.

## Configuration

| `ServerArgs` field | Default | Description |
|---|---|---|
| `load_reporter_port` | `None` | Fixed port for the Worker reporter gRPC service. `None` fully disables the reporter (no socket/task/binding). Valid range `1..65535`. |

Reporter-internal constants live in `config.py` and are intentionally not CLI
arguments: the stale threshold (`SNAPSHOT_STALE_AFTER_MS`, default `3000`), the
per-fire pull bound (`SNAPSHOT_PULL_TIMEOUT_SECONDS`), and the shutdown bound
(`SHUTDOWN_TIMEOUT_SECONDS`).

The external Router's paired reporter-port configuration is a delivery contract
only; its parameter name is chosen by the Router owner and is not defined here.

## Module layout

| File | Responsibility |
|---|---|
| `lifecycle.py` | Composition root (`start_load_reporter`) and `LoadReporterHandle`. |
| `service.py` | `LoadMonitorService.Monitor` bidi handler (depends only on runtime + proto). |
| `runtime.py` | `LoadReporterRuntime`: session table, the single fire loop, bounded shutdown. |
| `snapshot_source.py` | `LoadSnapshotSource` protocol; `ManagerLoadSnapshotSource` / `RouterLoadSnapshotSource` adapters. |
| `snapshot_validation.py` | Stateless one-pull validation (`validate_full_snapshot`). |
| `report_builder.py` | Validated rank tuple → `pb.LoadReport` with status + sequence id. |
| `config.py` | `LoadReporterConfig` / `WorkerMetadata` from `ServerArgs`; internal constants. |
| `proto/` | Generated `sglang.router.loadmonitor.v1` bindings. |

## Threading and async model

- Reporter components use the serving loop owned by HTTP, the
  `MultiTokenizerRouter`, or standalone SMG RPC.
- Single fire loop: at most one `get_loads()` in flight; the loop idles when
  no sessions remain and starts with the first registration.
- Pull, connection, write, and shutdown failures never propagate into
  inference requests. Shutdown cancels the fire task directly, so an in-flight
  pull is aborted immediately.

## Tests and validation

- Unit / integration (CPU, real in-process `grpc.aio`): proto contract, runtime
  fire loop (coalesced initial broadcast, shared periodic broadcast, per-session
  deadline gating, pull retry on rank-set change, interval re-anchoring,
  stale/error reports, in-flight pulls, re-registration, leases, shutdown),
  validation and report-builder contracts, service handshake/reporting,
  composition-root lifecycle (disabled, owner startup, cleanup, port conflicts,
  expected-rank updates), and standalone SMG wiring (capability guard,
  `_serve_grpc` failure/exit/cancel cleanup) via a faked `_serve_grpc` import
  boundary (no smg install required).
- Negative invariant tests verify that request activity does not trigger
  snapshot pulls.
- Topology-change tests verify that `update_expected_dp_ranks` updates the
  expected rank set without an immediate pull and that the next fire observes
  the new set.
- E2E (GPU + model, CUDA CI) under `test/registered/tokenizer/`: a real
  `grpc.aio` fake Router dials in and verifies that periodic reports continue
  through inference activity. The standalone SMG test verifies coexistence of
  inference and reporting without a request-end pull path.

## Known limitations

- No TLS/mTLS/gRPC auth, acknowledgements, replay, exactly-once delivery, or
  persistence.
- No custom gRPC keepalive/message-size tuning; grpcio defaults are used.
- Reports are periodic snapshots, not per-request events.
- The gRPC report interval controls delivery cadence; end-to-end snapshot
  freshness is bounded by the Scheduler's snapshot publication interval
  (`load_snapshot_publish_interval`), not by the gRPC delivery cadence.
- A topology update concurrent with a pull can make that one attempt fail
  (`UNREACHABLE`); the next fire self-heals. Publication ordering with the
  expected-rank-set update is Scheduler-side and not coordinated here.
- The external Router client (discovery, retry, registry reconciliation,
  load-aware policy) is a separate prerequisite; until it ships, keep the
  reporter disabled and do not claim end-to-end load-aware routing.
