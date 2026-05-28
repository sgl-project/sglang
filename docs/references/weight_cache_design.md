# Weight Cache Daemon: CUDA IPC Shared GPU Memory

## Problem

SGLang loads model weights from storage into GPU memory during engine startup. For large models (e.g., 70B FP16 ~140GB), this takes minutes. When the engine crashes or restarts, it must reload all weights from storage, even though the same weights may already exist in GPU memory from a previous instance.

## Solution

A persistent **Weight Cache Daemon** process that holds post-quantized, TP-sharded weights in GPU memory. On engine restart, the new engine process maps or copies weights from the daemon via CUDA IPC handles, reducing restart time from minutes to sub-second.

## Architecture

```
┌─ GPU i ──────────────────────────────────────────────────┐
│                                                          │
│  ┌───────────────────┐   cudaIpcMemHandle   ┌─────────┐ │
│  │ Weight Cache      │─────────────────────►│ Engine  │ │
│  │ Daemon (rank i)   │   (zero-copy)        │ Rank i  │ │
│  │                   │                      │         │ │
│  │ Holds:            │   or copy_()         │         │ │
│  │ - TP-sharded      │────── ~0.3s ────────►│         │ │
│  │   weights (fp8)   │   (copy mode)        │         │ │
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

## Two Mapping Modes

| Mode | Flow | Restart Time (70B FP16) | GPU Memory | Use Case |
|------|------|-------------------------|------------|----------|
| **Zero-copy** | import IPC handle → use as param.data | < 0.1s | 1x (shared) | Multi-instance, memory-constrained |
| **Copy** | import IPC handle → copy_() → release | ~0.3s | 2x (daemon+engine) | General, engine-independent |

## Design Details

### 1. Weight Cache Daemon

Each daemon process:
1. Initializes CUDA context on its assigned GPU
2. Loads model via `DefaultModelLoader` (full pipeline: disk → TP shard → quantize → repack)
3. Exports every parameter and buffer in `model.state_dict()` as CUDA IPC handles via `MultiprocessingSerializer`
4. Records a `CacheConfig` fingerprint (model path, TP/DP size, quant config hash, dtype)
5. Serves IPC handles over a Unix socket

### 2. IPC Model Loader

A new `BaseModelLoader` subclass that:
1. Connects to the daemon's Unix socket
2. Validates `CacheConfig` compatibility
3. On match: imports IPC tensors, replaces model parameters
4. On mismatch: falls back to `DefaultModelLoader` (disk load)

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

- `LoadFormat.IPC_CACHE` — new load format enum value
- `--weight-cache-mode` — server arg: `off` | `daemon` | `client` | `copy`
- `--weight-cache-socket` — server arg: path to daemon socket (for client mode)
- `IpcModelLoader` — new `BaseModelLoader` subclass
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

## File Layout

```
python/sglang/srt/weight_cache/
├── __init__.py
├── daemon.py          # WeightCacheDaemon process
├── ipc_loader.py      # IpcModelLoader (BaseModelLoader subclass)
└── protocol.py        # CacheConfig, socket protocol, serialization helpers
```

## Performance Expectations

| Model | Weight Size | Disk Load | IPC Copy | IPC Zero-copy |
|-------|-------------|-----------|----------|---------------|
| 7B FP16 | ~14 GB | ~30s | ~0.03s | < 0.01s |
| 70B FP16 | ~140 GB | ~3-5min | ~0.3s | < 0.1s |
| 235B FP8 | ~235 GB | ~5-10min | ~0.25s | < 0.1s |

GPU-internal copy bandwidth: ~500-900 GB/s (H20 HBM). IPC handle mapping: ~10k handles/ms.
