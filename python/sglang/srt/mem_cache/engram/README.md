# Engram — N-gram Embedding Module for SGLang

Engram is an experimental module that augments Transformer models with
n-gram hash-based embedding lookups. It runs **alongside** the existing
KV cache system (HiCache) rather than replacing any backend — the KV cache
continues to manage paged attention, while Engram independently manages its
own embedding tables through a dedicated storage layer.

> **Prototype status**: The current implementation is a simulation prototype.
> We support Qwen2 and Qwen3 model integration as a proof-of-concept
> (the DeepSeek model variant is not publicly released at this time).
> The module is used for simulating the Engram architecture and is not yet
> optimised for production workloads.

## Architecture Overview

Engram is split into two packages with a clear separation of concerns:

```
sglang/srt/
├── models/engram/              # Computation side
│   ├── engram.py               # Core module (hashing, gating, projections)
│   └── triton_ops/             # Optional Triton kernel implementations
│       ├── __init__.py
│       └── engram_triton.py
│
└── mem_cache/engram/           # Storage side (this directory)
    ├── __init__.py             # Package exports
    ├── engram_store.py         # EngramStore abstract base & EngramStoreConfig
    ├── engram_store_manager.py # EngramStoreManager + global accessors
    └── local_engram_store.py   # CPU DRAM-backed store

test/srt/engram/                # Integration tests
    ├── run_all_tests.py
    ├── test_import_paths.py
    ├── test_local_store_manager.py
    └── test_engram_e2e.py
```

### Computation side (`models/engram/`)

| Component | Description |
|---|---|
| `NgramHashMapping` | Builds n-gram hash keys from token IDs using prime-modulo hashing. Supports both NumPy (offline) and Torch (online) paths. |
| `MultiHeadEmbedding` | Looks up embeddings from an `EngramStore` given hashed indices. Each n-gram order gets multiple heads with prime-sized tables. |
| `Engram` | Top-level module that combines hashing, embedding lookup, gating, value projection, and a short depthwise convolution. Supports async prefetch on a separate CUDA stream. |
| `ShortConv` | Grouped depthwise 1-D convolution with RMSNorm and SiLU activation. |
| Triton ops | Optional fused kernels for gate-value computation, grouped RMSNorm, and short-conv preprocessing. Activated via `ENGRAM_TRITON=1`. |

### Storage side (`mem_cache/engram/`)

| Component | Description |
|---|---|
| `EngramStore` (ABC) | Abstract interface: `put_sharded`, `get_one`, `get_many`, `close`. |
| `LocalEngramStore` | Default backend — stores embedding tables as CPU tensors (`torch.nn.Embedding`) in process DRAM. No external dependencies. |
| `EngramStoreManager` | Registry and factory for per-layer `EngramStore` instances. Provides global accessor functions for singleton lifecycle management. |

## Supported Models

| Model | Entry Class | How to Enable |
|---|---|---|
| Qwen2 | `Qwen2ForCausalLM` | Requires swapping `Qwen2Model` to `Qwen2MoelEngram` in the model class |
| Qwen3 | `Qwen3ForCausalLM` | Set environment variable `ENABLE_ENGRAM=1` |

In Qwen3, enabling Engram is controlled by the `ENABLE_ENGRAM` environment
variable. When set, `Qwen3ForCausalLM` instantiates `Qwen3ModelEngram`
(which inherits from `Qwen2MoelEngram`) instead of the standard
`Qwen3Model`.

## Storage Backend

Engram uses a single storage backend: **Local DRAM**.

### Local (default)

Stores all embedding tables as CPU tensors (`torch.nn.Embedding`) in the
current process's DRAM. No external dependencies; suitable for single-node
development and testing.

**How it works:**

- `put_sharded(vocab_table)` — validates shape, casts to the configured
  dtype, and copies the table into an `nn.Embedding` weight buffer.
- `get_one(index, layer_id, device)` — returns a single embedding vector;
  returns a zero vector for out-of-range indices.
- `get_many(indices, layer_id, device)` — batch lookup via `nn.Embedding`,
  returns a tensor of shape `(*indices.shape, embedding_dim)`.
- `close()` — no-op (memory is released when the Python object is garbage
  collected).

**Configuration:**

The backend is selected via `EngramConfig.store_backend` (default `"local"`).
No environment variables are required.

## Quick Start

### Running integration tests

```bash
python test/srt/engram/run_all_tests.py
```

This runs three test suites in sequence:

1. **Import path verification** — checks all modules are importable
2. **LocalEngramStore + Manager lifecycle** — put/get correctness, global accessor lifecycle
3. **End-to-end Engram module** — forward pass with `EngramStoreManager`, backward compatibility, multi-layer scenarios

### Launching SGLang with Engram (Qwen3)

> **Note:** CUDA graph capture is not yet supported with Engram. You must
> pass `--disable-cuda-graph` when launching the server.

```bash
ENABLE_ENGRAM=1 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-XXX \
    --disable-cuda-graph \
    --port 30000
```

## Configuration

Global defaults are defined in `models/engram/engram.py` via the
`EngramConfig` and `BackBoneConfig` dataclasses. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `store_backend` | `"local"` | Storage backend (currently only `"local"` is supported) |
| `engram_vocab_size` | `[1024, 1024]` | Per-ngram-order hash table sizes |
| `max_ngram_size` | `3` | Maximum n-gram order (bigram + trigram) |
| `n_embed_per_ngram` | `512` | Total embedding dimension per n-gram |
| `n_head_per_ngram` | `8` | Number of hash heads per n-gram order |
| `layer_ids` | `[1, 15]` | Transformer layer indices where Engram is injected |
| `enable_prefetch` | `True` | Async embedding prefetch on a separate CUDA stream |

## Engram Compute Pipeline

Engram's compute path consists of three stages: **token hashing**, **embedding
lookup**, and **gated projection**. When prefetch is enabled a fourth
overlapping stage runs asynchronously on a side CUDA stream.

### Stage 1 — Token compression & n-gram hashing

Before any embedding lookup, raw token IDs are normalised by
`CompressedTokenizer` (NFKC/NFD normalisation, accent stripping, case folding,
whitespace collapsing) to collapse semantically equivalent tokens to the same
compressed ID. This reduces the effective vocabulary and improves hash
collision quality.

`NgramHashMapping.hash_torch` then computes, for every token position and
every Engram-injected layer, a set of hash indices using a prime-modulo XOR
scheme:

```
mix_n = token[t] * m[0]  XOR  token[t-1] * m[1]  XOR  ...  XOR  token[t-n+1] * m[n-1]
hash_n_head_j = mix_n  %  prime_j          # prime_j is unique per (layer, ngram, head)
```

Multipliers `m[k]` are seeded deterministically per layer so each layer
produces an independent hash space. The output is
`hash_input_ids: [B, T, num_heads_total]` where
`num_heads_total = (max_ngram_size - 1) * n_head_per_ngram`.

### Stage 2 — Multi-head embedding lookup

`MultiHeadEmbedding` maps each hash index to a `head_dim`-dimensional vector
by calling `EngramStore.get_many`. The per-head offsets are added to the raw
hash indices so that all heads share a single flat embedding table in the
store. The results are concatenated:

```
# Tensor shapes with default config (max_ngram_size=3, n_head=8, head_dim=64):
raw_embeddings : [B, T, 16, 64]   # 16 heads × 64-dim
embeddings     : [B, T, 1024]     # flattened  (engram_hidden_size)
```

### Stage 3 — Gated projection & short convolution

The flattened embeddings are projected into the backbone's hidden space and
combined with the current `hidden_states` via a learned scalar gate:

```
input_ids [B, T]
    │
    ▼
CompressedTokenizer.compress_torch()       # token-space normalisation
    │
    ▼
NgramHashMapping.hash_torch()              # per-layer prime-modulo XOR hashing
    │  → hash_input_ids [B, T, num_heads_total]
    ▼
MultiHeadEmbedding.forward()               # store.get_many  (CPU DRAM → GPU)
    │  → raw_embeddings [B, T, num_heads_total, head_dim]
    │  .flatten(start_dim=-2)
    │  → embeddings [B, T, engram_hidden_size]
    ▼
key_projs_all (Linear)                     # project to [B, T, hc_mult, D]
    │
    ├─► parallel_rms_norm (norm1_weight)   # normalise keys
    │
hidden_states [B, T, hc_mult, D]
    │
    └─► parallel_rms_norm (norm2_weight)   # normalise queries
    │
    ▼
gate = sigmoid( (normed_key · normed_query) / √D )   # scalar gate per head
    │  shape: [B, T, hc_mult, 1]
    ▼
value_proj (Linear)  →  v [B, T, 1, D]
    │
    ▼
value = gate * v                           # gated value
    │
    ▼
output = value + ShortConv(value)          # residual depthwise conv
    │  shape: [B, T, hc_mult, D]
    ▼
added back to hidden_states (hyper-connection residual)
```

`ShortConv` applies a grouped depthwise 1-D convolution (kernel size 4,
dilation = `max_ngram_size`) with RMSNorm and SiLU, capturing local sequential
context on top of the retrieved embeddings.

### Tensor size summary

```
num_ngram_orders  = max_ngram_size - 1              # e.g. 2  (bigram + trigram)
num_heads         = n_head_per_ngram                # e.g. 8
head_dim          = n_embed_per_ngram // num_heads  # e.g. 512 // 8 = 64
total_heads       = num_ngram_orders * num_heads    # e.g. 16
engram_hidden_size = num_ngram_orders * n_embed_per_ngram  # e.g. 1024
```

### Stage 4 — Async prefetch (optional)

When `enable_prefetch=True` (default), Stages 1–2 (hashing + embedding
lookup) are kicked off on a dedicated side CUDA stream at the very beginning
of each decode-step forward pass — **before any Transformer layer runs** —
so that the CPU→GPU embedding transfer overlaps with the entire layer stack:

```
Qwen2MoelEngram.forward()
    │
    ├── embed_tokens(input_ids)                      # (main stream)
    │
    ├──[side stream]──► _start_engram_prefetch()     # fired once per forward
    │                       for each engram layer_id:
    │                           compress_torch()
    │                           hash_torch()
    │                           get_many()  ← CPU DRAM → GPU transfer
    │                           cuda.Event.record(prefetch_event)
    │
    ├── layer 0  (main stream) ◄─── overlaps with side stream above
    ├── layer 1
    ├── ...
    ├── layer N  (Engram-injected)
    │       └─► Engram.forward()
    │               └─► _consume_prefetch()
    │                       main_stream.wait_event(prefetch_event)  # sync
    │                       return cached embeddings  # Stages 1-2 done
    │                       → Stage 3: gate + proj + conv
    ├── ...
    └── layer M  (next Engram-injected, if any)
            └─► Engram.forward() → _consume_prefetch() → ...
```

`_start_engram_prefetch` iterates over all Engram-injected `layer_id`s and
calls `start_prefetch` on each, so every Engram layer's embeddings are
prefetched in parallel on the same side stream before the first Transformer
layer executes. By the time execution reaches an Engram-injected layer, the
transfer is typically already complete and `_consume_prefetch` only inserts a
lightweight `wait_event` barrier.


## TODO

- [ ] Support CUDA graph capture with Engram enabled (currently requires `--disable-cuda-graph`)
- [ ] Production-ready DeepSeek model integration
- [ ] **Custom CUDA kernel for Engram** — replace the current multi-step Python/Triton pipeline with a single fused CUDA kernel covering: token compression, prime-modulo XOR hash computation, multi-head embedding gather (coalesced global-memory reads from CPU-pinned or GPU-resident tables), gate-value computation, and the short depthwise convolution. A fused kernel would eliminate intermediate tensor allocations, reduce kernel-launch overhead, and enable warp-level optimisations for the irregular gather pattern inherent to n-gram hash lookups.
- [ ] Distributed TP (tensor parallel) support for Engram embeddings across ranks
