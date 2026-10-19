# Engine shutdown RPC

`sglang.runtime.v1.SglangService/Shutdown` takes an empty `ShutdownRequest`
and returns an empty `ShutdownResponse`. It acknowledges acceptance, not
completion. The connection can close before the response arrives. The caller
must observe connection closure or process replacement to establish completion.

The caller verifies engine identity before invoking this RPC on its pinned
connection. There is no engine ID in the request, and clients must not
transparently reconnect and replay shutdown against a replacement engine.

Acceptance sets `TokenizerManager.gracefully_exit` on the manager event loop.
Readiness becomes false and new inference admission returns unavailable.
Requests already registered in `rid_to_state`, including tokenization and other
pre-dispatch work, remain visible to the existing drain loop. The existing
`sigterm_watchdog` drains them under the configured shutdown policy, stops the
subprocess watchdog, broadcasts `ShutdownReq`, waits for schedulers, and runs
its existing process/server cleanup. The RPC adds no separate termination path.
Concurrent/repeated calls share that one state transition and watchdog.
Listener cleanup releases the Python GIL while joining the Rust runtime, so
pending bridge calls can finish rather than deadlock teardown.

`WatchEngineState` publishes `healthy=false` after acceptance and preserves a
pending final unhealthy snapshot when its shutdown signal races the update.
Delivery is best effort: an unhealthy snapshot does not acknowledge process
exit, and shutdown does not wait for a client to receive it. Follower metadata
services return `UNIMPLEMENTED` for Shutdown.

## Native API limitation

`sglang.api.v1` declares the same wire contract but currently returns
`UNIMPLEMENTED`, without changing readiness or stopping any threads.

In native mode, `Engine._launch_subprocesses` skips the Python tokenizer manager
and starts the Rust frontend inside scheduler processes. `GrpcService` owns a
`CoreHandle`, which has no engine-owner lifecycle signal. Rust `Server::shutdown`
and `Runtime::request_shutdown` only stop frontend/runtime threads. Dispatching
a local `ShutdownReq` would not establish coordinated draining, fan-out to all
DP groups, suppression of the parent subprocess watchdog during intentional
rank exits, and parent-process exit. An owner hook coordinating those steps is
required before this API can acknowledge engine shutdown. No frontend-only
substitute is implemented.

## Validation

Run with this checkout on `PYTHONPATH=python`. Run the Rust API types, gRPC,
and native server tests, plus:

```bash
python -m unittest test.registered.unit.entrypoints.test_grpc_bridge
python -m unittest test.registered.unit.entrypoints.test_grpc_shutdown_rpc
python -m unittest test.registered.unit.managers.test_tokenizer_manager_rid_cleanup
python -m unittest test.registered.unit.api.v1.test_json_contract test.registered.unit.api.v1.test_parity
python test/registered/e2e/grpc/test_shutdown.py
```

The wire unit test uses a real Rust listener and a mock owner that never exits
the test process. It needs a built extension and grpcio-tools. The engine test
launches a standalone Python-backed engine, verifies serving, invokes Shutdown,
checks readiness/admission, and asserts engine and scheduler exit without using
Dynamo. It requires a GPU and a built extension. Set `SGLANG_TEST_GRPC_EXTENSION`
to a cargo-built `libsglang_grpc_core.so` to test that artifact directly.

For multi-rank runs, set `SGLANG_TEST_SHUTDOWN_TP=2` or
`SGLANG_TEST_SHUTDOWN_DP=2` on a host with enough GPUs. These exercise ordinary
TP/DP on one node; multi-node and attention-DP deployments need separate
validation. `SGLANG_TEST_SHUTDOWN_MODEL` selects the model (default:
`Qwen/Qwen3-0.6B`). Emergency test cleanup is not counted as successful shutdown.

## Validation performed on 2026-10-19

Base: upstream `678913a0f029394ad1f4ee64d1439421e9ea56ef`.
Rust checks used installed toolchain 1.93.1 (the workspace pins 1.92).
Python checks used Python 3.12.14 and the gRPC extension built from this change.

| Check | Result |
| --- | --- |
| `cargo +1.93.1 test --workspace -j 4` after updating the benchmark gRPC mock | Passed: 538 tests; no failures or ignored tests |
| `cargo +1.93.1 test -p sglang-api-types -p sglang-grpc -p sglang-server -j 4` | Passed: 23 API-types, 21 gRPC, 365 native-server tests |
| `cargo +1.93.1 test -p sglang-grpc -j 4` after listener teardown fix | Passed: 21 tests |
| Bridge, tokenizer request cleanup, Python API JSON and parity suites listed above | Passed: 95 tests |
| `python -m unittest test.registered.unit.entrypoints.test_grpc_shutdown_rpc -v` with the built extension | Passed: 3 tests; no skips |
| Second `cargo +1.93.1 run -p sglang-api-codegen` | All checked-in output hashes unchanged |
| `cargo +1.93.1 fmt --all -- --check`, isort 7.0.0, Ruff 0.15.1 formatting/lint, `git diff --check` | Passed |
| Standalone GPU engine test | Blocked before Shutdown: engine warmup failed in CUDA JIT kernels |
| TP/DP multi-rank and multi-node engine tests | Not run: only one GPU available |

The wire tests verify concurrent empty acknowledgments, immediate health and
admission changes, unhealthy state observation, listener closure with an open
watch, rejection of stale healthy snapshots, and teardown with a pending Python
snapshot. The last case timed out before the GIL-release fix and passed afterward.

The GPU attempts used an RTX 6000 Ada and cached `Qwen/Qwen3-0.6B` and
`Qwen/Qwen2.5-0.5B-Instruct`. The initial compiler was CUDA 11.5, which rejected
C++20. Retrying with installed CUDA 13.0 reached warmup but failed with
`CUDA error: invalid resource handle` in QK normalization / RoPE. Using
`SGLANG_FORCE_FUSED_OP_BACKEND=torch` and PyTorch sampling advanced to the
KV-cache kernel, which failed with the same error. None of these runs reached
the Shutdown call; they do not validate engine or scheduler exit. The GPU test
must pass in a working engine environment before claiming that validation.
