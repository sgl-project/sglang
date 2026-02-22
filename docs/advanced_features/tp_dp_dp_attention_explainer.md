# Understanding TP, DP, and DP-Attention in SGLang

This guide explains how SGLang distributes model inference across multiple GPUs using tensor parallelism (TP), data parallelism (DP), and the specialized DP-attention mode. All examples use an 8-GPU machine.

## Background: Why Parallelize at All?

Large language models often don't fit on a single GPU. Even when they do, a single GPU may not deliver enough throughput. Parallelism strategies let you either **split a model** across GPUs or **run multiple copies** of it.

## Tensor Parallelism (TP)

**What it does:** Splits a single model across `N` GPUs. Each GPU holds a slice of every layer's weight matrices. During inference, the GPUs compute partial results and communicate (via all-reduce or all-gather) to reconstruct the full output.

**When to use it:** When the model is too large for one GPU, or when you want to reduce per-request latency by distributing the computation.

**Trade-off:** More TP means more inter-GPU communication. On a single node with fast NVLink, TP=8 works well. Across nodes with slower interconnects, high TP can bottleneck on communication.

```
--tp-size 8 on 8 GPUs:

  One model, sharded across all 8 GPUs
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │GPU 0 │GPU 1 │GPU 2 │GPU 3 │GPU 4 │GPU 5 │GPU 6 │GPU 7 │
  │shard │shard │shard │shard │shard │shard │shard │shard │
  │ 0/8  │ 1/8  │ 2/8  │ 3/8  │ 4/8  │ 5/8  │ 6/8  │ 7/8  │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
        ←───── all-reduce / all-gather between all 8 ─────→
```

**KV cache:** The KV cache is **sharded by attention heads** across the TP group. Each GPU stores `total_num_kv_heads ÷ tp_size` heads worth of K and V tensors. All GPUs share the same token index space — when the attention backend reads the KV cache for token index `i`, each GPU reads only its shard of heads at that index.

```
KV cache with 8 KV heads, TP=8:

  GPU 0     GPU 1     GPU 2     ...     GPU 7
  ┌──────┐  ┌──────┐  ┌──────┐         ┌──────┐
  │head 0│  │head 1│  │head 2│   ...   │head 7│
  │ K,V  │  │ K,V  │  │ K,V  │         │ K,V  │
  │tokens │  │tokens │  │tokens │         │tokens │
  │ 0..N  │  │ 0..N  │  │ 0..N  │         │ 0..N  │
  └──────┘  └──────┘  └──────┘         └──────┘

Each GPU stores 1 head × max_total_num_tokens entries.
All GPUs store the SAME set of tokens (same requests), just different head shards.
```

The per-token KV cache cost per GPU is small (few heads), so `max_total_num_tokens` is large — more tokens fit in memory. However, all GPUs must store KV entries for every request since attention requires an all-reduce across the full TP group.

**Flag:** `--tp-size N`

## Data Parallelism (DP) — Normal Mode

**What it does:** Creates `N` independent copies (replicas) of the model. A dispatcher distributes incoming requests across replicas using a load-balancing strategy (round-robin, least-requests, or least-tokens).

**When to use it:** When the model fits at a lower TP and you want to maximize throughput by serving requests in parallel across replicas.

**GPU math:**

```
total_gpus = tp_size × dp_size    (assuming pp_size=1)
```

So on 8 GPUs, valid configurations are:

| `tp_size` | `dp_size` | Description |
|-----------|-----------|-------------|
| 8 | 1 | One replica sharded across all 8 GPUs |
| 4 | 2 | Two replicas, each sharded across 4 GPUs |
| 2 | 4 | Four replicas, each sharded across 2 GPUs |
| 1 | 8 | Eight fully independent replicas |

```
--tp-size 4 --dp-size 2 on 8 GPUs:

  Replica 0 (TP group)         Replica 1 (TP group)
  ┌──────┬──────┬──────┬──────┐ ┌──────┬──────┬──────┬──────┐
  │GPU 0 │GPU 1 │GPU 2 │GPU 3│ │GPU 4 │GPU 5 │GPU 6 │GPU 7│
  │shard │shard │shard │shard│ │shard │shard │shard │shard│
  │ 0/4  │ 1/4  │ 2/4  │ 3/4│ │ 0/4  │ 1/4  │ 2/4  │ 3/4│
  └──────┴──────┴──────┴──────┘ └──────┴──────┴──────┴──────┘
        ← all-reduce →               ← all-reduce →

  Dispatcher routes Request A → Replica 0
                     Request B → Replica 1
```

Each replica is a fully independent serving instance with its own scheduler, KV cache, and model weights. The `DataParallelController` sits in front and dispatches requests.

**KV cache:** Each DP replica has a **completely independent KV cache** — there is zero sharing between replicas. Within each replica, KV heads are sharded across the replica's TP group exactly as in pure TP mode (`total_num_kv_heads ÷ tp_size` heads per GPU). Each replica's GPUs independently run `init_memory_pool()` and allocate their own `ReqToTokenPool` and `TokenToKVPool`.

```
--tp-size 2 --dp-size 4 on 8 GPUs (model with 8 KV heads):

  Replica 0          Replica 1          Replica 2          Replica 3
  GPU 0 │ GPU 1      GPU 2 │ GPU 3      GPU 4 │ GPU 5      GPU 6 │ GPU 7
  4 heads│ 4 heads    4 heads│ 4 heads    4 heads│ 4 heads    4 heads│ 4 heads

  KV cache A ───┘     KV cache B ───┘     KV cache C ───┘     KV cache D ───┘
  (independent)        (independent)        (independent)        (independent)

  Request X served by Replica 0 → only GPUs 0,1 store its KV cache
  Request Y served by Replica 2 → only GPUs 4,5 store its KV cache
```

Because replicas don't share KV cache, a request's cached KV state is only available on the replica that served it. The dispatcher's load-balancing strategy determines which replica handles each request.

**Flags:** `--tp-size N --dp-size M`

**Key code paths:**
- Engine launch split: `python/sglang/srt/entrypoints/engine.py` (dp_size==1 launches a single TP group; dp_size>1 launches the `DataParallelController`)
- DP controller: `python/sglang/srt/managers/data_parallel_controller.py`

## DP-Attention — A Different Kind of Parallelism

DP-attention (`--enable-dp-attention`) is **not** normal DP. It does not create separate model replicas. Instead, it takes a single TP group and **splits the attention computation differently from the FFN/MoE computation** within that same group of GPUs.

### The Problem It Solves

Models like DeepSeek-V2/V3 and Qwen use Mixture-of-Experts (MoE). In MoE models:

- **FFN/MoE layers** benefit from expert parallelism (EP) — distributing different experts across GPUs. This needs all GPUs to coordinate on expert routing.
- **Attention layers** are relatively cheap per-GPU and have independent KV caches. Sharding attention across many GPUs (high TP) wastes communication bandwidth for little gain.

DP-attention solves this mismatch: let the **attention layers use fewer GPUs** (higher data parallelism), while the **FFN/MoE layers still use all GPUs** (full TP/EP).

### How It Works

In DP-attention mode, all GPUs still belong to a single TP group (they share one NCCL communicator, one set of weights). But the attention layers see a **remapped view** of the parallelism:

```
attn_tp_size = tp_size ÷ dp_size      # how many GPUs share one attention computation
attn_dp_rank = tp_rank ÷ attn_tp_size  # which attention-DP group this GPU belongs to
attn_tp_rank = tp_rank % attn_tp_size   # rank within that attention-DP group
```

This remapping is computed in `compute_dp_attention_world_info` (`python/sglang/srt/layers/dp_attention.py`).

### Concrete Example: `--tp 8 --dp 8 --ep 8 --enable-dp-attention`

This is a typical configuration for DeepSeek-V3 on 8 GPUs.

```
attn_tp_size = 8 ÷ 8 = 1   →  each GPU does attention independently
```

Since `attn_tp_size=1`, **every GPU runs attention on its own subset of requests with no attention-related communication**.

Meanwhile, for FFN/MoE, all 8 GPUs still form one TP/EP group and communicate via all-to-all for expert routing.

The per-GPU rank assignments:

```
┌─────┬─────────┬──────────────┬──────────────┬─────────────┐
│ GPU │ tp_rank │ attn_tp_rank │ attn_dp_rank │ moe_ep_rank │
├─────┼─────────┼──────────────┼──────────────┼─────────────┤
│  0  │    0    │      0       │      0       │      0      │
│  1  │    1    │      0       │      1       │      1      │
│  2  │    2    │      0       │      2       │      2      │
│  3  │    3    │      0       │      3       │      3      │
│  4  │    4    │      0       │      4       │      4      │
│  5  │    5    │      0       │      5       │      5      │
│  6  │    6    │      0       │      6       │      6      │
│  7  │    7    │      0       │      7       │      7      │
└─────┴─────────┴──────────────┴──────────────┴─────────────┘
```

Reading this table:
- **attn_tp_rank = 0 for everyone**: No GPU shares attention work with another. Each GPU independently computes attention for its own batch of requests.
- **attn_dp_rank differs per GPU**: Each GPU handles different requests for the attention phase — this is the "data parallel" part.
- **moe_ep_rank differs per GPU**: Each GPU holds different experts for the MoE phase — this is the expert parallel part. All GPUs cooperate on every request during FFN/MoE.

### What Happens During a Forward Pass

Here's the flow for a single transformer layer in DP-attention mode:

```
Incoming tokens (global batch)
         │
    ┌────▼────┐
    │ Scatter │  Each GPU takes its own slice of the batch
    └────┬────┘
         │
   ┌─────▼──────┐
   │  Attention  │  Each GPU runs attention INDEPENDENTLY on its slice
   │  (local)    │  No cross-GPU communication needed
   └─────┬──────┘
         │
    ┌────▼────┐
    │ Gather  │  Reassemble all tokens into the global batch
    └────┬────┘
         │
   ┌─────▼──────┐
   │  FFN / MoE │  All 8 GPUs cooperate (all-to-all for expert routing)
   │  (global)  │  Standard TP/EP communication
   └─────┬──────┘
         │
    next layer...
```

The **gather** and **scatter** operations (`dp_gather` and `dp_scatter` in `dp_attention.py`) are what stitch the two parallelism strategies together. Before attention, tokens are scattered so each GPU works on its own subset. After attention, tokens are gathered back so the FFN/MoE phase can see the full batch.

### KV Cache in DP-Attention Mode

The KV cache behavior in DP-attention is the trickiest to understand. Two key principles:

1. **KV heads are sharded by `attn_tp_size`, not `tp_size`.** The function `get_attention_tp_size()` returns `attn_tp_size`, and all KV cache head counts use this value. Since `attn_tp_size ≤ tp_size`, each GPU stores **more KV heads** than in pure TP mode.

2. **Each GPU stores KV cache only for its own attention-DP subset of requests.** There is no sharing of KV cache across attention-DP ranks. Each attention-DP rank runs its own scheduler with its own `ReqToTokenPool` and `TokenToKVPool`.

The scatter/gather operations move **hidden state activations** between attention and FFN phases — they do NOT touch the KV cache. The KV cache is written locally during each GPU's attention forward pass and stays local.

**Concrete numbers for `--tp 8 --dp 8` with a model having 8 KV heads:**

```
attn_tp_size = 8 ÷ 8 = 1
KV heads per GPU = 8 ÷ 1 = 8  (full copy of all heads!)

  GPU 0          GPU 1          GPU 2          ...    GPU 7
  ┌────────┐     ┌────────┐     ┌────────┐           ┌────────┐
  │8 heads │     │8 heads │     │8 heads │           │8 heads │
  │ K,V    │     │ K,V    │     │ K,V    │    ...    │ K,V    │
  │tokens: │     │tokens: │     │tokens: │           │tokens: │
  │ own    │     │ own    │     │ own    │           │ own    │
  │requests│     │requests│     │requests│           │requests│
  │ only   │     │ only   │     │ only   │           │ only   │
  └────────┘     └────────┘     └────────┘           └────────┘

  GPU 0 handles requests {A, B}  → only GPU 0 stores KV for A, B
  GPU 3 handles requests {G, H}  → only GPU 3 stores KV for G, H
```

Each GPU stores all 8 KV heads (no head sharding) but only for ~1/8th of total requests. This is the opposite trade-off from pure TP=8 where each GPU stores 1 KV head but for ALL requests.

**Memory trade-off:** Because each GPU stores more heads per token, the per-token KV cache cost is larger, so `max_total_num_tokens` per GPU is smaller. But each GPU also handles fewer concurrent requests (`max_running_requests ÷ dp_size`), so fewer token slots are needed. These two effects roughly balance out.

```
KV cache sizing formula:

  cell_size = num_kv_heads_per_gpu × (head_dim_k + head_dim_v) × num_layers × dtype_bytes
            = (total_kv_heads ÷ attn_tp_size) × ...

  max_total_num_tokens = available_memory ÷ cell_size

  Pure TP=8:      cell_size is small  → many tokens fit, shared across all requests
  DP-Attn dp=8:   cell_size is 8× larger → fewer tokens fit, but only local requests need them
```

**Note on MLA (Multi-head Latent Attention):** Models like DeepSeek-V2/V3 use MLA, which compresses the KV cache into a low-rank representation (`kv_lora_rank + qk_rope_head_dim`) with effectively `head_num=1`. This makes the KV cache size per token **independent** of `attn_tp_size` — MLA's compressed cache is the same size regardless of parallelism configuration. This is one reason DP-attention works especially well with MLA models.

### Another Example: `--tp 8 --dp 4 --enable-dp-attention`

```
attn_tp_size = 8 ÷ 4 = 2   →  pairs of GPUs share attention
```

Now GPUs are grouped in pairs for attention:

```
┌─────┬─────────┬──────────────┬──────────────┐
│ GPU │ tp_rank │ attn_tp_rank │ attn_dp_rank │
├─────┼─────────┼──────────────┼──────────────┤
│  0  │    0    │      0       │      0       │
│  1  │    1    │      1       │      0       │ ← GPUs 0,1 share attention group 0
│  2  │    2    │      0       │      1       │
│  3  │    3    │      1       │      1       │ ← GPUs 2,3 share attention group 1
│  4  │    4    │      0       │      2       │
│  5  │    5    │      1       │      2       │ ← GPUs 4,5 share attention group 2
│  6  │    6    │      0       │      3       │
│  7  │    7    │      1       │      3       │ ← GPUs 6,7 share attention group 3
└─────┴─────────┴──────────────┴──────────────┘
```

Here, 4 attention-DP groups each have 2 GPUs. Attention is sharded across 2 GPUs (small all-reduce within each pair), while FFN/MoE still uses all 8.

**KV cache for this config** (model with 8 KV heads):

```
attn_tp_size = 2  →  KV heads per GPU = 8 ÷ 2 = 4

  Attn Group 0      Attn Group 1      Attn Group 2      Attn Group 3
  GPU 0 │ GPU 1     GPU 2 │ GPU 3     GPU 4 │ GPU 5     GPU 6 │ GPU 7
  heads │ heads     heads │ heads     heads │ heads     heads │ heads
  0-3   │ 4-7       0-3   │ 4-7       0-3   │ 4-7       0-3   │ 4-7

  Each pair shares KV cache for the SAME requests (sharded by heads).
  Different pairs store KV cache for DIFFERENT requests.
```

This is a middle ground: 4 heads per GPU (vs 1 in pure TP=8, vs 8 in full dp=8), with each pair handling ~1/4 of total requests.

## Normal DP vs DP-Attention: Key Differences

| | Normal DP | DP-Attention |
|---|---|---|
| **Model copies** | `dp_size` independent replicas | Single model instance on one TP group |
| **GPU count** | `tp_size × dp_size` GPUs total | `tp_size` GPUs total (dp_size ≤ tp_size) |
| **Weight sharing** | No — each replica loads its own copy | Yes — one set of weights, one NCCL group |
| **KV cache** | Separate per replica; heads sharded within each replica's TP group | Partitioned across attention-DP ranks; heads sharded by `attn_tp_size` (not `tp_size`), so each GPU stores more heads but for fewer requests |
| **Attention** | Each replica does full attention on its requests | Attention split across DP-attention groups |
| **FFN/MoE** | Each replica does full FFN on its requests | All GPUs cooperate on FFN/MoE for all requests |
| **Scheduling** | External dispatcher routes whole requests to replicas | Internal scatter/gather moves tokens between phases |

## Expert Parallelism (EP) and TP×EP Hybrid for MoE

The sections above explain how DP-attention splits **attention vs FFN/MoE** across the same GPUs. This section zooms into the MoE layer itself and explains how `--ep-size` controls expert placement — and why the hybrid TP4EP2 configuration is often the best trade-off.

### The Three Extremes for MoE Expert Placement

Consider DeepSeek-V3 with 256 routed experts on 8 GPUs. There are three pure strategies for distributing expert weights:

**Pure TP8 (every expert sharded across all 8 GPUs):**
Each expert's weight matrices (gate_proj, up_proj, down_proj) are split into 8 shards, one per GPU. Every GPU participates in computing every expert. No token routing needed, but an 8-GPU all-reduce after every expert computation.

**Pure EP8 (each GPU holds 32 complete, unsharded experts):**
Each GPU owns a unique set of 32 experts. Tokens must be routed to the GPU that holds their selected expert via an 8-way all-to-all. No all-reduce, but load imbalance — popular experts overload their GPU while others sit idle.

**TP4EP2 — the hybrid** (`--tp-size 8 --ep-size 2`):
Split the 256 experts into 2 EP groups of 128 each. Within each group, shard expert weights across 4 GPUs using tensor parallelism. This is the sweet spot for many MoE models.

### What TP4EP2 Looks Like on 8 GPUs

The key derived quantity:

```
tp_per_ep_group = tp_size ÷ ep_size = 8 ÷ 2 = 4
```

This gives 2 EP groups, each containing 4 GPUs, with expert weights tensor-parallel sharded within each group:

```
EP Group 0 (GPUs 0-3)                    EP Group 1 (GPUs 4-7)
Holds experts 0–127                      Holds experts 128–255
Each expert TP-sharded across 4 GPUs     Each expert TP-sharded across 4 GPUs

┌────────┬────────┬────────┬────────┐    ┌────────┬────────┬────────┬────────┐
│ GPU 0  │ GPU 1  │ GPU 2  │ GPU 3  │    │ GPU 4  │ GPU 5  │ GPU 6  │ GPU 7  │
│        │        │        │        │    │        │        │        │        │
│shard0/4│shard1/4│shard2/4│shard3/4│    │shard0/4│shard1/4│shard2/4│shard3/4│
│of each │of each │of each │of each │    │of each │of each │of each │of each │
│of 128  │of 128  │of 128  │of 128  │    │of 128  │of 128  │of 128  │of 128  │
│experts │experts │experts │experts │    │experts │experts │experts │experts │
└────────┴────────┴────────┴────────┘    └────────┴────────┴────────┴────────┘
```

Every GPU holds `128 experts × (1/4 of each expert's weights)`. Total memory per GPU = `total_expert_params / 8` — identical to pure TP8 or pure EP8. The difference is entirely about communication patterns.

Note that attention and shared experts still use all 8 GPUs in TP8 — the EP grouping only affects the sparse MoE expert layers.

### Forward Pass Through One MoE Layer (TP4EP2)

**Step 1 — Router:** Every GPU runs the gating network on all tokens and computes top-k expert IDs. This is replicated — all 8 GPUs agree on which tokens go to which experts.

```
Token A → experts {3, 45, 200, 130, ...}    (top-8 selected)
Token B → experts {12, 180, 99, 255, ...}
```

**Step 2 — All-to-all dispatch (between EP groups):** Tokens routed to experts 0–127 need to be on GPUs 0–3. Tokens routed to experts 128–255 need to be on GPUs 4–7. A 2-way all-to-all shuffles tokens between groups:

```
Before all-to-all:                     After all-to-all:

All 8 GPUs have all tokens             EP Group 0 (GPUs 0-3):
                                          receives tokens for experts 0-127
  GPUs 0-7: [Token A, B, C, ...]
                                        EP Group 1 (GPUs 4-7):
                                          receives tokens for experts 128-255
```

**Step 3 — Expert computation with TP4 (within each EP group):** Each EP group runs its experts on the tokens it received. Because each expert's weights are TP-sharded across 4 GPUs, this works like standard tensor parallelism:

```
Within EP Group 0 (GPUs 0-3), Token A routed to expert 3:

  GPU 0: matmul with shard 0/4 of expert 3 ──┐
  GPU 1: matmul with shard 1/4 of expert 3 ──┼── all-reduce across 4 GPUs
  GPU 2: matmul with shard 2/4 of expert 3 ──┤   to get full result
  GPU 3: matmul with shard 3/4 of expert 3 ──┘
```

The all-reduce here is across 4 GPUs, not 8.

**Step 4 — All-to-all combine:** Results are sent back to their original GPUs via another 2-way all-to-all so every GPU has results for all its original tokens.

```
Full timeline for one MoE layer:

  [all-to-all dispatch]  →  [TP4 expert compute + all-reduce₄]  →  [all-to-all combine]
       2-way                    within each EP group                    2-way
```

### Why TP4EP2 Wins Over the Extremes

| | Pure TP8 | TP4EP2 | Pure EP8 |
|---|---|---|---|
| **All-to-all width** | none | 2-way | 8-way |
| **All-reduce width** | 8-way | 4-way | none |
| **Experts per EP group** | 256 (all) | 128 | 32 |
| **Load balance** | Perfect | Good (128 experts averages out variance) | Poor (32 experts, high variance) |
| **Communication cost** | High (8-way all-reduce) | Moderate | High (8-way all-to-all) |

- **vs Pure TP8:** TP4EP2 cuts the all-reduce from 8-GPU to 4-GPU. All-reduce cost scales super-linearly with group size, so halving the group is a significant win. The added 2-way all-to-all is cheap.
- **vs Pure EP8:** TP4EP2 replaces an expensive 8-way all-to-all with a cheap 2-way one. Load balance improves dramatically — with 128 experts per group, the law of large numbers smooths out routing hotspots far better than 32 experts per GPU.

### Full Transformer Layer with TP8 Attention + TP4EP2 MoE

```
                        All 8 GPUs (TP8)
                    ┌─────────────────────┐
                    │   Attention (TP8)    │  heads split across 8 GPUs, all-reduce₈
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Shared Experts (TP8) │  standard TP matmul, all-reduce₈
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │         Router (replicated)       │
              └──────┬─────────────────┬─────────┘
                     │ all-to-all₂     │ all-to-all₂
            ┌────────▼────────┐  ┌─────▼───────────┐
            │ Sparse Experts  │  │ Sparse Experts   │
            │ EP Group 0      │  │ EP Group 1       │
            │ GPUs 0-3 (TP4)  │  │ GPUs 4-7 (TP4)  │
            │ all-reduce₄     │  │ all-reduce₄      │
            └────────┬────────┘  └─────┬───────────┘
                     │ all-to-all₂     │ all-to-all₂
              ┌──────▼─────────────────▼─────────┐
              │         Combine results           │
              └──────────────┬───────────────────┘
                             │
                        next layer...
```

Attention and shared experts use all 8 GPUs uniformly. Only the sparse MoE expert computation splits into the EP-group topology — and only for those layers.

## KV Cache Summary Across Configurations

For a model with 8 KV heads on 8 GPUs:

| Configuration | `attn_tp_size` | KV heads/GPU | Requests stored per GPU | `max_total_num_tokens` | Trade-off |
|---|---|---|---|---|---|
| Pure TP=8, DP=1 | 8 | 1 | All | Largest | Thin shard of all requests |
| Normal DP: TP=2, DP=4 | 2 | 4 | 1/4 (own replica) | Medium | Independent caches, no sharing |
| DP-Attn: TP=8, DP=2 | 4 | 2 | 1/2 | Large | Moderate split |
| DP-Attn: TP=8, DP=4 | 2 | 4 | 1/4 | Medium | Balanced |
| DP-Attn: TP=8, DP=8 | 1 | 8 (all) | 1/8 | Smallest | Full heads, fewest requests |

**Key code paths for KV cache:**
- Head count per GPU: `model_config.get_num_kv_heads(get_attention_tp_size())` in `python/sglang/srt/configs/model_config.py`
- Pool allocation: `init_memory_pool()` in `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `get_attention_tp_size()` in `python/sglang/srt/layers/dp_attention.py` — returns `attn_tp_size` when DP-attention is enabled, `tp_size` otherwise
- Memory pool classes: `MHATokenToKVPool` and `MLATokenToKVPool` in `python/sglang/srt/mem_cache/memory_pool.py`

## Constraints

- **Divisibility:** DP-attention requires `tp_size % dp_size == 0` (checked in `python/sglang/srt/server_args.py`).
- **Multi-node DP:** Normal DP across multiple nodes is not supported. Multi-node data parallelism requires `--enable-dp-attention`.
- **dp_size=1 disables DP-attention:** When `dp_size=1`, DP-attention is automatically turned off — there's nothing to split.

## When to Use What

**Normal DP** (without `--enable-dp-attention`):
- Model fits at a low TP (e.g., TP=1 or TP=2) and you want more throughput by running multiple replicas.
- Dense (non-MoE) models where there's no FFN/attention parallelism mismatch.

**DP-Attention**:
- MoE models (DeepSeek-V2/V3, Qwen-MoE) where FFN benefits from expert parallelism across all GPUs but attention does not.
- High-throughput, high-batch-size workloads where splitting the batch for attention gives better efficiency.
- Multi-node setups where you need data parallelism (normal DP doesn't support multi-node).

**Maximize TP** (TP=8, DP=1):
- The model barely fits in memory and you need all GPUs just to hold the weights.
- Latency-sensitive workloads with small batch sizes.
