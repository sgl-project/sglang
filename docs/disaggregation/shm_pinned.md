# shm_pinned Backend

## Status

`shm_pinned` is the single-node PD-disaggregation backend for hosts where prefill and decode run as separate processes on PCIe-connected GPUs without NVLink. PR1 intentionally keeps the scope small:

- Sync `KV + aux` transfer only
- Single-node, multi-process deployment
- POSIX shared memory + pinned host memory

Not in PR1:

- Async send workers
- Mamba/SWA/NSA state transfer
- `state_item_lens` handshake extensions

## Why another backend

`mooncake` and `nixl` are good defaults when RDMA or a broader transport stack is available. `shm_pinned` targets a narrower case:

- the workload is confined to one machine
- GPU-to-GPU P2P is unavailable or not desirable
- the dominant transfer path is PCIe D2H/H2D
- operational simplicity is preferred over a larger networking dependency surface

In that setting, a shared-memory ring buffer is enough to move KV pages between prefill and decode workers with a predictable control path.

## Design

PR1 uses one decode-owned shared-memory session per TP rank:

1. Decode creates a ring buffer in POSIX shared memory and registers it as pinned host memory.
2. Decode sends `ShmPinnedInfo` to prefill, including the decode control endpoint.
3. Decode sends a `TRANSFER_REQ` containing destination KV page indices.
4. Prefill waits for a free slot, performs D2H copies for the requested KV pages, appends aux metadata on the last chunk, and marks the slot ready.
5. Decode waits for ready slots, performs H2D copies into the destination KV buffers, copies aux metadata on the last chunk, and then frees the slot.

The request key is `(session_id, room)` rather than `room` alone. This avoids cross-session overwrite when multiple decode sessions reuse the same room value.

## Failure semantics

PR1 keeps the timeout-based fallback from the existing disaggregation stack, but it no longer relies on timeout as the only failure signal:

- Decode -> prefill abort still exists for request cancellation.
- Prefill -> decode now sends an explicit `FAIL` control message when transfer setup or copy fails.
- Decode removes the pending request on explicit failure and reports the stored error immediately.

This keeps the 300-second timeout as a safety net instead of the primary failure path.

## Current limitations

- Only `state_type=none` is supported.
- The backend assumes a single-node deployment.
- The implementation is optimized for contiguous KV-page groups; fragmented transfers still work but use more small copies.
- Benchmarks for the upstream PR should focus on PCIe / no-NVLink hosts.

## Backend boundaries

Use `shm_pinned` when:

- prefill and decode are on the same host
- you want a transport with minimal external dependencies
- RDMA and NVLink are not the main optimization target

Prefer `mooncake` or `nixl` when:

- multi-node deployment is required
- transport abstraction across NIC/storage fabrics matters
- NVLink/RDMA-specific paths are available and beneficial

## Configuration

CLI arguments introduced for this backend:

- `--disaggregation-transfer-backend shm_pinned`
- `--disaggregation-shm-slot-count`
- `--disaggregation-shm-chunk-tokens`

Example:

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend shm_pinned \
  --disaggregation-bootstrap-port 8998

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --base-gpu-id 1 \
  --disaggregation-transfer-backend shm_pinned \
  --disaggregation-bootstrap-port 8998
```

## Benchmark plan

The upstream PR should include at least one benchmark table for the target environment:

- single host
- PCIe-only GPU interconnect
- no NVLink

Recommended comparison:

- `mooncake` baseline on the same host
- `shm_pinned` sync KV+aux

Metrics:

- end-to-end throughput
- TTFT
- TPOT
- P90 latency
