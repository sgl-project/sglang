# UniFlow KV transfer backend

`--disaggregation-transfer-backend uniflow` selects a native KV transfer backend
for prefill/decode disaggregation built on the `uniflow._core` binding. Transfers
are prefill-driven: decode publishes where each request's KV pages, aux data, and
state must land, and prefill writes them into decode memory with one-sided
UniFlow `put()`. Control messages use the `CommonKVManager` ZMQ side channel, and
rank discovery reuses `CommonKVBootstrapServer` unchanged.

## Scope

The backend transfers paged KV, aux data, and mamba, SWA, or DSA state over
one-to-one prefill/decode rank routing: each prefill rank serves one decode rank
and each decode rank waits for one prefill response. This covers equal attention
TP sizes, and MLA models without linear-attention layers with a larger prefill TP
whose extra prefill ranks are dummy ranks: they get METADATA with empty KV and aux
frames, move no data, and send no STATUS.

The manager raises `ValueError` at construction for:

- attention CP != 1, PP != 1, or decode context parallelism (`--dcp-size` > 1);
- hisparse or `SGLANG_MOONCAKE_CUSTOM_MEM_POOL`;
- a state type other than mamba, SWA, or DSA, or a state type count that differs
  from the number of state components;
- a non-positive `SGLANG_DISAGGREGATION_UNIFLOW_TRANSFER_TIMEOUT_MS`;
- mismatched pointer and length counts for KV, aux, or state buffers.

If `uniflow._core` cannot be imported, it raises `ImportError` before
`CommonKVManager.__init__`, so no socket or thread is created. Routing that is
not one-to-one is rejected per request at receiver init: the room fails with a
reason and decode sends ABORT, so prefill does not hold the KV until its bootstrap
timeout.

## Design principles

- Build on the common manager, sender, receiver, and bootstrap server; override
  only the transfer path, the control messages, and status and failure handling.
- Keep the object model small. `UniflowKVManager` (one per engine rank) owns the
  UniFlow agent, registered segments and export ids, side-channel threads, the
  decode peer cache, and per-room transfer state. `UniflowKVSender` enqueues
  prefill chunks; `UniflowKVReceiver` registers decode segments, publishes
  per-request metadata, and notifies prefill of aborts.
- Fail fast at init on an unsupported configuration or a missing binding, and
  surface every later failure as `KVPoll.Failed` with a reason, raised as
  `KVTransferError`, rather than hanging or silently falling back.

## Transfer contract

1. When decode resolves a prefill peer it has not cached, it sends REGISTER with
   its agent id and the export ids of its KV, aux, and state segments.
2. Per request, decode sends METADATA to each prefill peer: destination KV page
   indices, aux index, state indices, and decode prefix length. Decode's waiting
   timeout starts here.
3. The prefill transfer worker waits, up to the bootstrap timeout, until the
   room's METADATA and its decode peer's REGISTER have both arrived, then
   connects to the decode agent and imports its segments on first use.
4. For each chunk, the worker synchronizes the chunk's ready event when the
   scheduler attached one for an early send, then puts the KV pages in one batch
   with one request per KV segment per run of pages contiguous on both sides.
5. On the last chunk, the worker puts aux (required when aux segments are
   registered) and state, even when the chunk has no KV pages.
6. Prefill reports only the terminal status: STATUS with Success, or with Failed
   and its reason. Until it arrives, the receiver's `poll()` returns
   `WaitingForInput`.

Prefill pushes because it knows when each chunk's KV is ready. KV, aux, and state
are separate put batches; each must complete within
`SGLANG_DISAGGREGATION_UNIFLOW_TRANSFER_TIMEOUT_MS` (milliseconds, default 30000).

## Wire protocol

| Message | Direction | Frames | Notes |
|---|---|---|---|
| REGISTER | decode to prefill | 3 | JSON agent id and base64 export ids |
| METADATA | decode to prefill | 10 | empty KV and aux frames: dummy rank; aux index -1: no aux |
| ABORT | decode to prefill | 6 | room, reason, and the decode address for the ack |
| STATUS | prefill to decode | 5 | common status message, `GUARD` tag, reason frame |
| ABORT_ACK | prefill to decode | 3 | common deferred-release ack, not `GUARD`-framed |

Every message except ABORT_ACK starts with `GUARD`; the `conn.py` module
docstring gives the layout. Prefill logs and drops messages without it, and
REGISTER, ABORT, or METADATA messages with fewer than 3, 3, or 8 frames. An
ABORT's reason and decode address, and a METADATA's state indices and decode
prefix length, are optional. Prefill ignores unknown message types without
logging.

With `SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE` (on by default), decode
waits for ABORT_ACK only for rooms aborted by a scheduler abort request; a room
it aborts on its waiting timeout is held until the deferred-release timeout.
Prefill answers an ABORT that carries the decode address, including one for a
concluded room, with ABORT_ACK once every put queued before the ABORT has
returned, unless a failed put for the room did not shut down cleanly (see
[Failure and abort semantics](#failure-and-abort-semantics)).

## Threads and ordering

Both roles run an accept thread. Prefill runs a message thread (REGISTER,
METADATA, ABORT) and one FIFO transfer worker that runs chunks and ABORT_ACKs in
enqueue order, so the scheduler never blocks on put completion. Decode runs a
message thread (STATUS, ABORT_ACK) and the common heartbeat checker, which fails
in-flight rooms of a prefill whose bootstrap server stops answering `/health`.

One worker keeps per-room chunk order, confines connection use to one thread, and
orders each ABORT_ACK after every put queued before the ABORT. The cost is
head-of-line blocking: a chunk waiting for metadata (up to the bootstrap
timeout), a put waiting for completion (up to the transfer timeout), or an RDMA
shutdown after a failed put (no deadline) delays every room queued behind it. The
worker skips queued chunks of rooms that already concluded.

## Failure and abort semantics

- Failed is terminal: status never leaves Failed, and a last chunk that completes
  after its room failed reports Failed to decode, not Success.
- A failed put keeps a reason already recorded for its room, such as a decode
  abort that raced it, and sender and receiver `abort()` keep an existing reason
  over the generic one.
- Timeouts (seconds, default 300): prefill fails a room still bootstrapping
  `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` after its sender was created, and the
  worker's metadata wait uses the same bound; decode fails a room
  `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` after METADATA and sends ABORT.
- A decode ABORT for a live room clears its transfer info and fails it with the
  decode's reason. One that arrives before the prefill sender exists is parked (at
  most 4096, oldest evicted) and the sender's constructor fails the room. Before
  the KV put, and before the aux and state puts of a last chunk, the worker skips
  a room that was stopped meanwhile.
- A failed put (error, timeout, or binding exception) drops the cached decode peer
  and calls the connection's `shutdown()` before anything is reported. If it
  returns, decode is told Failed. If it raises, prefill sends neither Failed nor
  that room's ABORT_ACK (tracked for at most 4096 rooms) and leaves decode to its
  timeouts, which release the pages without proof that the writes stopped. The
  next transfer to that decode agent reconnects.
- Other errors, such as a segment count mismatch or a missing aux index, fail the
  room and notify decode at once.

## Transport tiers and failed-put shutdown

UniFlow picks one transport per put batch, using a tier only if every request in
the batch has a handle for it on both sides; the backend sets no transport
override. A VRAM-to-VRAM batch on the agent's device tries the intra-host GPU peer
tier (P2P/XGMI on AMD, NVLink on NVIDIA), then RDMA, then TCP; any other batch,
including aux in host memory, tries RDMA, then TCP. TCP is compiled in only when
UniFlow is built with TCP enabled, as its HIP build is, and registered only when a
non-loopback bind address resolves.

| Tier | What `shutdown()` waits for |
|---|---|
| Intra-host GPU peer | nothing; queued GPU copies still complete |
| RDMA | pending and in-flight work to drain, with no deadline |
| TCP | local lanes and threads; bytes handed to the kernel may still land |

So decode may safely free a request's pages on Failed or ABORT_ACK after a failed
put only on the RDMA tier.

Decode does not always wait for either message. With
`SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE`, it releases an aborted
request's device pages without an ack once
`SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT` (default 30 s) expires.
That equals the default transfer timeout, and RDMA shutdown has no deadline, so on
any tier a put that stalls to its deadline can still be writing when decode
releases the pages. A deferred-release timeout well above the transfer timeout
narrows this window but does not close it.

## Correctness requirements

- KV pages, aux data, and state transfer bit-exactly; the last chunk carries aux
  and state even when it has no KV pages.
- The i-th source page index of a chunk writes the i-th destination index of the
  same slice; a slice size mismatch fails the room.
- Missing metadata, put failure or timeout, decode abort, and decode timeout reach
  Failed on both sides without deadlocking the transfer worker.

## Testing

- `test/registered/unit/disaggregation/test_uniflow_backend.py`: CPU tests of
  the connector with a fake `uniflow._core`.
- `test/registered/amd/disaggregation/test_uniflow_transfer_engine_gpu.py`: KV,
  aux, and state transfer between two GPUs on the real binding.
- `test/registered/amd/disaggregation/test_uniflow_transfer_engine_e2e.py`:
  prefill and decode servers on separate GPUs, with output compared against a
  non-disaggregated server.
