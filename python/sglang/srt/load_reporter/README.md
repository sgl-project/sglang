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
Worker replies with an ack immediately, waits up to one second for the initial
sampling attempt, and then sends the first `LoadReport`. A successful attempt
therefore makes the first report a completed current snapshot; a hung attempt
produces an explicit `UNREACHABLE` report after the bound. Periodic reports are
then streamed on the negotiated interval, anchored from that first report. This removes the old
FastAPI-only `POST /v1/start_reporting` control plane, so **every** serving mode
— including the ones that never start FastAPI — is reachable.

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

Every mode shares one reporter service / runtime / proto / sampler and one
`enable_load_monitor("request_lifecycle")` decorator. Only the startup site and
the snapshot source differ.

| Serving mode | Reporter start site | Snapshot source | Request-end hint | Sampling |
|---|---|---|---|---|
| HTTP | FastAPI lifespan (`http_load_reporter_lifespan`) | `TokenizerManager` | static `@enable_load_monitor` on `generate_request` | initial + periodic + request-end wake |
| native gRPC (`--grpc-port`) | reuses the same FastAPI lifespan (no second listener) | `TokenizerManager` | same static decorator | initial + periodic + request-end wake |
| embedded Engine | `Engine.__init__` (`start_load_reporter_in_background`) | `TokenizerManager` snapshot reader | same static decorator | initial + periodic + request-end wake |
| multi-tokenizer HTTP (`--tokenizer-worker-num > 1`) | sole `MultiTokenizerRouter` owns the port; HTTP workers bind an IPC notifier | Router shared-memory snapshot reader | HTTP workers coalesce refresh over IPC to the sole owner | initial + periodic + request-end wake |
| standalone SMG RPC (`--smg-grpc-mode`) | `grpc_server.py::_on_request_manager_ready` (`start_load_reporter`) | `GrpcRequestManager.get_loads(include=["core"])` | same decorator applied at runtime to the current instance's bound `generate_request` | initial + periodic + request-end wake |

> **Multi-tokenizer native gRPC is not supported.** `ServerArgs` rejects
> `--grpc-port` together with `--tokenizer-worker-num > 1`, so the reporter does
> not claim that combination. Multi-tokenizer applies to HTTP only.

**HTTP and standalone SMG RPC are symmetric**: both do register-time initial
sampling, periodic interval sampling, and request-end active wake-up.

Standalone SMG RPC does **not** require a separately deployed SMG process:
`smg-grpc-servicer` is a Python package that SGLang imports in-process under
`--smg-grpc-mode`. SGLang attaches the reporter through the existing
`on_request_manager_ready(request_manager, server_args, scheduler_info)`
callback before the gRPC server accepts requests. Capability is detected with
`inspect.signature(_serve_grpc)` (no version sniffing): if the reporter is
enabled but that hook is missing, startup fails loudly; if the reporter is
disabled, the existing compatibility path is preserved.

## Request-end semantics

A request-end (`COMPLETION`) is a **synchronous, non-blocking hint**, not an
accurate per-request counter:

```text
request end ── decorator finally ──▶ notify (sync) ──▶ sampler coalesce ──▶ snapshot ──▶ same bidi stream
```

The decorator never samples, awaits, or writes the gRPC stream on the request
path. It only wakes the single-flight sampler. Concurrent completions coalesce
into at most one follow-up refresh, so **high-throughput completion does not
imply one report per request**. Load values always come from the snapshot
source; reports converge, they are not one-to-one with hints.

The same rule holds across modes:

- HTTP / native gRPC / embedded Engine: `TokenizerManager.generate_request`
  carries the static `@enable_load_monitor("request_lifecycle")`; multi-worker
  HTTP forwards a coalesced IPC hint to the sole router-owned sampler.
- standalone SMG RPC: SGLang wraps the *current* `GrpcRequestManager`
  instance's bound `generate_request` with the same decorator at runtime and
  restores it on shutdown. The class, other instances, and
  `shutdown`/dispatch/abort methods are never modified.

## Composition root

`start_load_reporter(server_args, snapshot_source, *, event_owner=None,
request_lifecycle_method=None) -> Optional[LoadReporterHandle]` is the single
serving-mode-agnostic entry point. Serving entrypoints only ever see the
returned handle's `close()`; no reporter-internal type leaks into them.

- `load_reporter_port is None` → returns `None` before importing grpc/protobuf.
- `snapshot_source is None` (multi-tokenizer HTTP worker) → installs a coalescing
  refresh notifier bound to `event_owner` that forwards hints to the sole owner
  over IPC. No gRPC server, no port bound.
- otherwise → owns a `LoadReporterRuntime` + a `grpc.aio` server on
  `host:load_reporter_port`, binds `event_owner` (if any) so decorator events
  wake the sampler, and — when `request_lifecycle_method` is set — installs the
  bound-method decorator on that one instance.

`LoadReporterHandle.close()` is idempotent and tears down in order: stop the
gRPC server, close the runtime, close the IPC notifier, unbind the registry
callback, restore any shadowed bound method (identity-safe: it only removes its
own shadow, never a later replacement).

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
  sends the ack immediately, then a bounded sampled-first report, followed by
  periodic `LoadReport`s.
- Same `router_id` re-registering on a new stream replaces the old session;
  different `router_id`s coexist. Each session's response queue is capacity-1,
  latest-wins, so a slow Router never accumulates historical reports.
- EOF, cancel, `stop`, lease timeout, and server shutdown all run the same
  idempotent cleanup.
- `LoadReport.worker` remains compatibility metadata; the Router should
  associate a report with the Worker identity of the outbound task, not trust
  `worker_addr`.

**Report status:** `HEALTHY` when every rank satisfies
`report_time - snapshot_time <= load_reporter_snapshot_stale_after_ms`; `STALE`
when at least one rank exceeds it (ranks still included); `UNREACHABLE` when the
store has never completed a full snapshot.

## Configuration

| `ServerArgs` field | Default | Description |
|---|---|---|
| `load_reporter_port` | `None` | Fixed port for the Worker reporter gRPC service. `None` fully disables the reporter (no socket/task/binding). Valid range `1..65535`. |
| `load_reporter_snapshot_stale_after_ms` | `3000` | Emit `REPORT_STATUS_STALE` past this age. |
| `load_reporter_zone` | `None` | Optional zone metadata; empty string normalized to `None`. |

The external Router's paired reporter-port configuration is a delivery contract
only; its parameter name is chosen by the Router owner and is not defined here.

Reporter-internal lifecycle constants live in `config.py` and are intentionally
not CLI arguments.

## Module layout

| File | Responsibility |
|---|---|
| `lifecycle.py` | Composition root plus HTTP-lifespan and background-loop ownership helpers. |
| `decorator.py` | `enable_load_monitor(kind)` / `bind_load_monitor(owner, notify)`; one shared async-generator finalization helper for both the static and bound-method `request_lifecycle` paths. |
| `service.py` | `LoadMonitorService.Monitor` bidi handler (depends only on runtime + proto). |
| `runtime.py` | `LoadReporterRuntime`: inbound Router session table, sampler wiring, bounded shutdown. |
| `sampler.py` | `LoadSampler` single-flight loop; `ManagerLoadSnapshotSource` / `RouterLoadSnapshotSource`. |
| `store.py` | `LatestSnapshotStore` latest-wins view. |
| `report_builder.py` | `SnapshotView` → `pb.LoadReport` with status + sequence id. |
| `ipc.py` | `LoadReporterRefreshNotifier`: multi-tokenizer worker refresh coalescer. |
| `config.py` | `LoadReporterConfig` / `WorkerMetadata` from `ServerArgs`; internal constants. |
| `proto/` | Generated `sglang.router.loadmonitor.v1` bindings. |

## Threading and async model

- Reporter components share one owned asyncio loop: the serving loop for HTTP,
  router, and standalone modes, or a dedicated background loop for Engine.
- Single-flight sampler: at most one `get_loads()` in flight; hints only set a
  wake event.
- Request-end hooks are synchronous and non-throwing; a callback exception is
  logged and never alters the wrapped function's result or exception.
- Sampling, connection, write, and shutdown failures never propagate into
  inference requests.

## Tests and validation

- Unit / integration (CPU, real in-process `grpc.aio`): decorator contract
  (static + bound method; normal exhaustion, business exception, `aclose()`,
  cancellation, callback isolation, identity-safe restore), proto contract,
  runtime sessions, service handshake/reporting, composition-root lifecycle, and
  standalone SMG wiring (capability guard, `_serve_grpc` failure/exit/cancel
  cleanup) via a faked `_serve_grpc` import boundary (no smg install required).
- E2E (GPU + model, CUDA CI) under `test/registered/tokenizer/`: a real
  `grpc.aio` fake Router dials in for single-owner, multi-owner, and standalone
  SMG modes. These require a GPU/model (and `smg-grpc-servicer` for standalone),
  so they do not run on CPU-only hosts. The standalone test uses the SMG
  inference stub to perform a real generation and verifies that its request-end
  wake supplies the snapshot used at the next report deadline.

## Known limitations

- No TLS/mTLS/gRPC auth, acknowledgements, replay, exactly-once delivery, or
  persistence.
- No custom gRPC keepalive/message-size tuning; grpcio defaults are used.
- Reports are convergent hints, not per-request events.
- The external Router client (discovery, retry, registry reconciliation,
  load-aware policy) is a separate prerequisite; until it ships, keep the
  reporter disabled and do not claim end-to-end load-aware routing.
