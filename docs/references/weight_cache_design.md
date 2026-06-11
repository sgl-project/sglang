# Weight Cache Daemon: CUDA IPC Shared GPU Memory

## Problem

SGLang loads model weights from storage into GPU memory during engine startup. For large models (e.g., 70B FP16 ~140GB), this takes minutes. When the engine crashes or restarts, it must reload all weights from storage, even though the same weights may already exist in GPU memory from a previous instance.

## Solution

A persistent **Weight Cache Daemon** process that holds post-quantized, TP-sharded weights in GPU memory. On engine restart, the new engine process maps weights from the daemon via CUDA IPC handles, reducing restart time from minutes to sub-second.

## Architecture

```
┌─ GPU i ──────────────────────────────────────────────────┐
│                                                          │
│  ┌───────────────────┐   cudaIpcMemHandle   ┌─────────┐ │
│  │ Weight Cache      │─────────────────────►│ Engine  │ │
│  │ Daemon (rank i)   │   (zero-copy)        │ Rank i  │ │
│  │                   │                      │         │ │
│  │ Holds:            │                      │         │ │
│  │ - TP-sharded      │                      │         │ │
│  │   weights (fp8)   │                      │         │ │
│  │ - weight_scale    │                      │         │ │
│  │ - workspace       │                      │         │ │
│  │ - all post-quant  │                      │         │ │
│  │   params/buffers  │                      │         │ │
│  └───────────────────┘                      └─────────┘ │
│                                                          │
└──────────────────────────────────────────────────────────┘

Coordination: Unix Socket /tmp/sglang_weight_cache_gpu{i}.sock
```

Each GPU runs one daemon process holding only its own TP rank's shard. The daemon caches the **complete post-quantization state** (after `process_weights_after_loading()`), including new parameters like `weight_scale`, `workspace`, repacked weights, etc.

The engine and daemon share the same physical GPU memory via CUDA IPC. Only 1x model memory is needed on each GPU.

## Design Details

### 1. Weight Cache Daemon

Each daemon process:
1. Initializes CUDA context on its assigned GPU
2. Loads model via `DefaultModelLoader` (full pipeline: disk → TP shard → quantize → repack)
3. Exports every parameter and buffer in `model.state_dict()` as CUDA IPC handles via `MultiprocessingSerializer`
4. Records a `CacheConfig` fingerprint (model path, TP/DP size, quant config hash, dtype)
5. Serves IPC handles over a Unix socket

### 2. IPC Model Loader

A `BaseModelLoader` subclass that:
1. Connects to the daemon's Unix socket
2. Validates `CacheConfig` compatibility
3. On match: imports IPC tensors, replaces model parameters (zero-copy)
4. On mismatch: falls back to `DefaultModelLoader` (disk load) in client mode; raises error in daemon mode

### 3. Config Validation (Critical)

Any mismatch in these fields triggers a full disk reload:

- `model_path` + `model_arch` — different model
- `tp_size` + `tp_rank` — different TP sharding
- `dp_size` — different DP strategy
- `quant_method` + `quant_config_hash` — different quantization
- `dtype` — different precision

This ensures correctness when engine restart configuration changes.

### 4. Process Lifecycle

```
First Start:
  Engine → launch daemons (rank 0..N) → daemons load from disk (~3min for 70B)
         → engine loads from daemon via IPC (~0.3s)
         → engine runs normally

Engine Crash/Restart (daemon alive):
  Engine → connect to daemon socket
         → validate CacheConfig → match → IPC load (~0.3s)
         → mismatch → fallback to disk load (~3min)

Daemon Restart:
  Daemon → reload from disk → re-export IPC handles
         → subsequent engine restarts can use cache
```

### 5. Integration Points

- `LoadFormat.IPC_CACHE` — load format enum value
- `--weight-cache-mode` — server arg: `off` | `daemon` | `client`
- `--weight-cache-socket` — server arg: path to daemon socket (for client mode)
- `IpcModelLoader` — `BaseModelLoader` subclass
- `ModelRunner.load_model()` — dispatches to `IpcModelLoader` when cache mode is set
- `WeightCacheDaemon` — standalone process, launched by engine or independently

### 6. Relationship to Existing Mechanisms

| Mechanism | Relationship |
|-----------|-------------|
| `ShardedStateLoader` | Similar pattern (per-rank pre-processed state), but reads from disk instead of GPU IPC |
| `_ShardedGpuParamOffloader` | Reuses `MultiprocessingSerializer` for CUDA IPC; offloader shares within server, cache daemon shares across restarts |
| `RemoteInstanceModelLoader` | Copies from running instance via NCCL/RDMA; IPC is faster for same-GPU mapping |
| `torch-memory-saver` | Pause/resume within single process; cache daemon persists across process lifetimes |
| `HostSharedMemoryManager` | CPU-side shared memory; cache daemon is GPU-side shared memory |

## Launching Daemons

### Single command (all TP ranks)

```bash
python3 -m sglang.srt.weight_cache.daemon \
    --model-path /path/to/model \
    --tp-size 4 \
    --load-format auto --dtype auto
```

This automatically:
- Spawns one daemon process per GPU (gpu 0..3)
- Allocates a free port for NCCL distributed init
- Waits for all daemons to become ready
- Monitors processes — if any exits, terminates all

### Single rank (manual)

```bash
DIST_PORT=29500

python3 -m sglang.srt.weight_cache.daemon \
    --model-path /path/to/model \
    --gpu-id 0 --tp-size 4 --tp-rank 0 \
    --dist-init-method tcp://127.0.0.1:$DIST_PORT &
```

## File Layout

```
python/sglang/srt/weight_cache/
├── __init__.py
├── daemon.py          # WeightCacheDaemon process + launch_weight_cache_daemons
├── ipc_loader.py      # IpcModelLoader (BaseModelLoader subclass)
└── protocol.py        # CacheConfig, socket protocol, serialization helpers
```

## Performance Expectations

| Model | Weight Size | Disk Load | IPC Zero-copy |
|-------|-------------|-----------|---------------|
| 7B FP16 | ~14 GB | ~30s | < 1s |
| 70B FP16 | ~140 GB | ~3-5min | < 1s |
| 235B FP8 | ~235 GB | ~5-10min | < 1s |

IPC handle mapping: ~10k handles/ms.