# SGLang Embedded Load Reporter

## Overview

The per-worker load reporter runs inside the SGLang Python HTTP/TokenizerManager
process and continuously streams scheduler load snapshots to multiple Routers by
using gRPC client streaming.

**Runtime constraints:**

- In single-tokenizer mode, snapshots come from
  `TokenizerManager.get_loads(include=["core"])`.
- In multi-tokenizer mode, the single `MultiTokenizerRouter` owns the runtime;
  HTTP workers register through IPC and coalesce refresh notifications.
- The reporter uses `grpc.aio.insecure_channel` (h2c, without TLS or gRPC
  authentication).
- `POST /v1/start_reporting` is a strictly internal Router-to-Engine control
  endpoint and does not require `--admin-api-key`.

### Runtime dependencies

Load-reporting dependencies are provided by the shared `load-reporter` optional
extra in each platform pyproject. They do not increase the default dependency
set of a regular SGLang wheel. Install the extra in environments that use this
feature:

```bash
pip install "sglang[load-reporter]"
```

The extra requires `grpcio>=1.78.0` and `protobuf>=6.31.1,<7`. The `test` and
`dev` extras also install these dependencies so that community CI can run the
load-reporter tests. A normal server can still start when the dependencies are
missing or incompatible, but the registration endpoint returns HTTP 501.

## Architecture

```text
FastAPI lifespan
 ├─ Single tokenizer: LoadReporterRuntime (composition root)
 └─ Multiple tokenizers: HTTP worker proxy/notifier ─IPC→ MultiTokenizerRouter
                                                └─ LoadReporterRuntime (sole owner)
     ├─ LatestSnapshotStore (atomic latest-wins view)
     ├─ ReportBuilder (SnapshotView → protobuf)
     ├─ LoadSampler (single-flight get_loads loop)
     └─ MonitorManager (MonitorKey → MonitorTask map)
         └─ MonitorTask × N (one independent gRPC stream per Router target)
```

**Timing boundaries:**

- A request-end notification only refreshes the store; it is synchronous and
  non-blocking.
- A gRPC write occurs only when a stream first connects or when that Monitor's
  `report_interval_ms` deadline expires.
- Timer and request-end refreshes share one sampler state machine, so at most
  one `get_loads()` call is in flight at any time.

## Module layout

| File | Responsibility |
|------|----------------|
| `config.py` | Freezes `LoadReporterConfig` and `WorkerMetadata` from `ServerArgs`; defines internal transport constants. |
| `store.py` | `LatestSnapshotStore`: validates `LoadSnapshot`, applies timestamp fallback and latest-wins merging, and publishes immutable `SnapshotView` values. |
| `report_builder.py` | `ReportBuilder`: converts `SnapshotView` to `pb.LoadReport` and adds status and a process-global sequence number. |
| `sampler.py` | `LoadSampler`: the only task that calls `get_loads()`; coalesces refresh notifications in a single-flight background loop. |
| `registration.py` | Strict Pydantic schemas, `MonitorKey` and `MonitorRegistration` value objects, origin normalization, and the `POST /v1/start_reporting` route. |
| `monitor.py` | `MonitorManager` owns the target map and performs identity-safe upserts; each `MonitorTask` owns one gRPC stream and its fixed-rate lease/reconnect state machine. |
| `runtime.py` | `LoadReporterRuntime`: top-level composition, the `start_reporting` control plane, the synchronous `notify_request_finished` hook, and bounded shutdown. |
| `ipc.py` | Correlates multi-tokenizer control requests and responses, coalesces refresh events, and maps stable errors. |
| `proto/load_monitor.proto` | Embedded `router.loadmonitor.v1` IDL shared with the Router load-monitor service. |

### Regenerating the Python protobuf code

Run the pinned toolchain from the repository root so the generated files remain
compatible with the project's minimum supported runtime versions:

```bash
codegen_dir=$(mktemp -d /tmp/sglang-load-reporter-codegen.XXXXXX)
python3 -m venv "$codegen_dir/venv"
"$codegen_dir/venv/bin/python" -m pip install \
  grpcio==1.78.0 grpcio-tools==1.78.0 protobuf==6.31.1
cd python/sglang/srt/load_reporter/proto
"$codegen_dir/venv/bin/python" -m grpc_tools.protoc \
  -I. --python_out=. --grpc_python_out=. load_monitor.proto
```

`grpc_tools.protoc` generates an absolute import for the sibling module. Before
committing, replace `import load_monitor_pb2 as load__monitor__pb2` with the
package-relative import
`from . import load_monitor_pb2 as load__monitor__pb2`. Do not remove or modify
the protobuf or gRPC runtime-version checks emitted by the generator.

## Control flow

### Startup

1. **Lifespan setup**
   - Single tokenizer: construct
     `LoadReporterRuntime(snapshot_source, server_args)` and install the
     request-finished hook.
   - Multiple tokenizers: each HTTP worker constructs a control proxy and
     refresh notifier; the Router lazily constructs the single runtime on the
     first registration.
   - Store the runtime or unsupported reason in
     `app.state.load_reporter_runtime` and
     `app.state.load_reporter_unsupported_reason`.

### Registration and reporting

2. **Router registration**
   - `POST /v1/start_reporting` calls
     `runtime.start_reporting(payload, worker_addr)`, then
     `MonitorManager.upsert`.
   - The first registration creates a `MonitorTask`, starts its `run()` task,
     and calls `sampler.activate()`.
   - Re-registering from the same origin updates `MonitorRegistration`
     (`revision++`), the lease, and the interval, then wakes the task so it can
     recompute its deadline.
   - Re-registering from a different origin returns HTTP 409.

3. **Sampling loop (`LoadSampler`)**
   - Refresh immediately after activation, then wait for either the wake event
     or a `min_interval_ms` timeout.
   - The request-end hook calls `notify_refresh()` to set the wake event.
   - `MonitorManager` calls `notify_schedule_changed()` when an interval
     changes.
   - After each `get_loads()` call, `LatestSnapshotStore.apply_full_snapshot`
     atomically publishes a new view; failures call `record_error` instead.
   - Notifications received while sampling cause at most one additional
     refresh.

4. **Stream writes (`MonitorTask`)**
   - Each target owns a `grpc.aio` channel and client stream
     (`LoadMonitorServiceStub.Report`).
   - Send the current snapshot immediately after each connection, then use a
     fixed `report_interval_ms` cadence.
   - Concurrent waits cover the stop event, registration updates, lease
     expiry, call completion, and the report deadline.
   - An update re-anchors the deadline at `updated_at + interval`.
   - After write backpressure clears, skip missed periods instead of replaying
     historical reports.

5. **Reconnect and error classification**
   - **Retryable** (`UNAVAILABLE`, `DEADLINE_EXCEEDED`, or
     `RESOURCE_EXHAUSTED`): exponential backoff from 0.25 to 5 seconds with
     20 percent jitter.
   - **Wait for renewal** (`INVALID_ARGUMENT`, `UNAUTHENTICATED`,
     `PERMISSION_DENIED`, or `UNIMPLEMENTED`): record the error and wait for a
     registration update with a larger revision.
   - A successful epoch, defined as sending at least one report, resets the
     backoff to its initial value.
   - On lease expiry the task exits and `on_stopped` removes it from the
     manager map.

### Shutdown

6. **Shutdown**
   - During the HTTP worker lifespan, detach hooks and IPC components before
     closing the proxy and notifier.
   - In the parent-process `finally` block, close the sole runtime on the
     Router event loop before removing the Router socket.
   - `close()` orders shutdown as `sampler.close()`, then `manager.close()` to
     stop every task and await convergence.
   - After the timeout, `cancel_remaining()` force-cancels tasks that have not
     converged.

## Configuration

| `ServerArgs` field | Default | Description |
|--------------------|---------|-------------|
| `load_reporter_snapshot_stale_after_ms` | `3000` | Reports `REPORT_STATUS_STALE` after this threshold. |
| `load_reporter_zone` | `None` | Optional zone metadata; an empty string is normalized to `None`. |

**Internal constants** (`config.py`; they do not create CLI arguments):

- `GRPC_CONNECT_TIMEOUT_SECONDS = 3.0`
- `RECONNECT_INITIAL_SECONDS = 0.25`
- `RECONNECT_MAX_SECONDS = 5.0`
- `SHUTDOWN_TIMEOUT_SECONDS = 5.0`

## Protocol constraints

- **Wire contract:** fields 1 through 13 and all enum values in
  `proto/load_monitor.proto` match the canonical Router IDL.
- **`Worker.worker_addr`:** normalize the registration HTTP request origin to
  `scheme://host:port`; never read `Forwarded` or `X-Forwarded-*`.
- **`RankLoad.snapshot_time_unix_ms`:** prefer
  `LoadSnapshot.timestamp * 1000`; fall back to `collected_at_unix_ms` when the
  source timestamp is invalid.
- **Latest-wins merging:** for repeated snapshots of the same DP rank, keep the
  newer timestamp. When timestamps are equal, use the complete raw metrics
  from the current sample.
- **Status logic:**
  - `HEALTHY`: every rank satisfies
    `report_time - snapshot_time <= snapshot_stale_after_ms`.
  - `STALE`: at least one rank exceeds the threshold; the report still includes
    its ranks.
  - `UNREACHABLE`: there is no authoritative rank snapshot because the store
    has never completed `apply_full_snapshot` successfully.

## Threading and asynchronous model

- **Single event loop:** every reporter component shares the FastAPI and
  TokenizerManager asyncio event loop.
- **Single-flight sampler:** at most one `get_loads()` call is in flight;
  notifications set a wake event instead of creating tasks.
- **Per-target task:** every Monitor owns one independent `asyncio.Task`; the
  manager has no coordinator, reconcile loop, or session generation.
- **Request-end hook:** synchronously call `sampler.notify_refresh()`; do not
  await, create a task, or call `get_loads()` there.
- **Error isolation:** sampling, validation, connection, write, background-task,
  and shutdown failures never propagate into inference requests or the main
  FastAPI lifespan.

## Tests and validation

Unit tests cover the store, builder, sampler, monitor deadlines, internal
control behavior, optional-dependency boundary, IPC correlation and coalescing,
single ownership across multiple workers, shutdown cleanup, and msgpack
round-trips. GPU end-to-end validation checks that two tokenizer/HTTP workers
establish only one Router gRPC stream.

## Known limitations

- No TLS, mTLS, gRPC authentication, acknowledgements, replay, exactly-once
  delivery, or persistence.
- No custom gRPC keepalive or message-size configuration; grpcio defaults are
  used.
- The Router protocol has no SDK metadata, normalized load, `worker_id`, or
  similar extensions.
