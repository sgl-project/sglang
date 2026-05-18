# HiCache L3 over UDS and shared memory

This package implements the connection of HiCache to standalone L3 storage server process. SGLang processes talk to a standalone **StorageServer** process via **UNIX domain sockets (UDS)**. Data moves through **shared memory**, so the hot path avoids copying through the gRPC payload.

```
  ┌──────────────────┐                  ┌──────────────────┐
  │ SGLang tp_rank=0 │ ── uds & shm ───>│                  │
  ├──────────────────┤                  │  Standalone L3   │
  │ SGlang tp_rank=1 │ ── uds & shm ───>│                  │
  ├──────────────────┤                  │  Storage Server  │
  │       ...        │ ── uds & shm ───>│                  │
  └──────────────────┘                  └──────────────────┘
```

## Goals

- **Standalone out-of-tree storage server**: Allow the storage implementation outside the SGLang codebase.
- **High Performance**: Use shared memory on the data path; gRPC carries metadata and iovec descriptions (offsets/lengths into a registered pool).
- **Vendor-neutral**: The wire-protocol is defined in protobuf and not tied with any particular cache systems such as LMCache or Mooncake.

## Usage

gRPC and mempool registration use two separate Unix domain sockets.

**SGLang server** (RPC backend + extra config):

```sh
python -m sglang.launch_server ... \
    --hicache-storage-backend rpc \
    --hicache-storage-backend-extra-config \
    '{"mempool_socket": "unix:///path/to/mempool.sock", "grpc_socket": "unix:///path/to/grpc.sock"}'
```

**Demo storage** (file-backed `StorageServer` included in this tree):

```sh
python -m sglang.srt.mem_cache.storage.rpc.demo_server \
    --data-dir /tmp/hicache_demo_data \
    --mempool-socket unix:///path/to/mempool.sock \
    --grpc-socket unix:///path/to/grpc.sock
```

Paths may be written as `unix:///path/to/socket` or as a bare absolute filesystem path; both are normalized by the client helpers.

## Design

### Memory pool sharing

Used only for **RegisterPool**: the host sends a **custom binary preamble** plus **one FD** via **`SCM_RIGHTS`** (e.g. a `memfd` holding KV tensors). The storage side receives the FD, `mmap`s it, and assigns a **`pool_id`**.

### RegisterPool flow (mempool socket)

After a successful registration, **keep the connection open** for the lifetime of that pool registration on the storage side.

```
  sglang                    storage
     |   RegisterPool      |
     | (magic + name + FD) |
     |-------------------->|
     |   pool_id (u32 LE)  |
     |<--------------------|
     |                     |
     |  keep connection    |
```

**Wire format** (same as `rpc_fd.send_pool_registration` / `recv_pool_registration`):

1. **8 bytes** little-endian magic: `0x8DF6FBFD` (distinct from an HTTP/2 client preface starting with `PRI `).
2. **`uint32`** name length (little-endian), then pool name.
3. **Ancillary data**: exactly ONE file descriptor (`SCM_RIGHTS`).

On success, the server replies with **4 bytes** little-endian **`pool_id`**. IDs are allocated starting at **1** and increase monotonically.

If the connection closes, the server unmaps the region for that `pool_id`. Subsequent gRPC `Put`/`Get` with the same `pool_id` will fail. The client should NOT Close the registration connection while the process still relies on that pool.

### RPC

Carries the gRPC service **`StorageServer`** with:

- `Exist` — batch key existence check.
- `BatchPut` / `BatchGet` — read/write L3 using `Iovec` entries (offset + length) into a registered memory pool.

`BatchPutRequest` and `BatchGetRequest` must include `pool_id`. The ID is returned by the storage process when the host registers a pool on the mempool socket.

Refer to the message definiation in [`hicache.proto`](../../../../../../proto/sglang/runtime/hicache.proto) for more information.

## TODOs

- Handle StorageServer restarts.
