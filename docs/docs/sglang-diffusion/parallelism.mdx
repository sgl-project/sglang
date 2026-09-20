---
title: "Parallelism Overview"
metatags:
    description: "How CFG, tensor, Ulysses, and ring parallelism compose in SGLang Diffusion: what each axis splits, the divisibility constraints, topology mapping, and how to choose a configuration."
---

SGLang Diffusion ships several parallelism strategies. Each one splits a
different dimension of the DiT forward pass, which is exactly why they can be
combined: the total GPU count is the product of the degrees,

```text
num_gpus = cfg_parallel_degree × tp_size × sp_degree
sp_degree = ulysses_degree × ring_degree
```

This page is the map — what each axis does, which combinations are legal, and
how to pick one. Per-axis depth lives in
[Sequence Parallelism](./ring_sp_performance),
[Encoder Parallelism](./encoder_parallel) (text/image encoders are a separate
axis with their own knob), and the [CLI reference](./api/cli).

## The axes

| Strategy | Splits | Communication | Flag |
| --- | --- | --- | --- |
| CFG parallel | guidance branches | one combine per denoise step | `--cfg-parallel-size` |
| Tensor parallel (TP) | weights and attention heads | all-reduce per transformer block | `--tp-size` |
| Ulysses SP | sequence outside attention ↔ heads inside it | two all-to-alls per attention | `--ulysses-degree` |
| Ring SP | sequence rows inside attention | neighbor-only K/V rotation, overlapped with compute | `--ring-degree` |
| K/V-gather CP | sequence rows inside attention | one K/V all-gather per attention | `--kv-gather-degree` |
| Data parallel | requests | none between replicas | `--dp-size` |

Two strategies compose when they split different dimensions. TP and Ulysses
both touch heads but compose serially — TP splits the projection weights, then
Ulysses splits the activations of the TP-local heads. Ring and Ulysses compose
because ring splits rows while Ulysses splits heads. Ring has no composition
with a K/V-all-gather style of attention parallelism: both answer the same
question (how a rank's query rows see remote K/V), so they are alternatives for
one slot, not complements.

## What happens to the shapes

For `tp_size = T`, `ulysses_degree = U`, `ring_degree = R`, one attention runs:

```text
[B, S/(U·R), H/T, D]                     sequence-sharded activations
   │  Ulysses input all-to-all (inside each Ulysses group)
   ▼
[B, S/R, H/(T·U), D]                     full sequence of this ring block, few heads
   │  ring attention (R−1 neighbor hops; Q never moves, K/V rotate)
   ▼
[B, S/R, H/(T·U), D]
   │  Ulysses output all-to-all (inverse)
   ▼
[B, S/(U·R), H/T, D]
```

The ring merge (online softmax) requires every rank in a ring group to hold the
*same heads* over *different rows*; the group construction guarantees this. A
K/V-gather (CP-style) variant fills the same slot differently: instead of R−1
overlapped hops it all-gathers K/V once and computes the local Q rows against
the full sequence in one shot — fewer, larger transfers, paid for by holding the
whole K/V per rank. Like ring it splits rows, so it adds no head constraint. When no SP degree is
set explicitly, `sp_degree=2` defaults to `kv_gather_degree=2` — its
measured-win zone — and higher degrees default to Ulysses.
Ulysses groups are laid out on contiguous ranks and ring groups on strided
ranks, so with a node-major rank mapping, Ulysses traffic stays on intra-node
NVLink (all-to-all needs full-bisection bandwidth) while ring hops cross the
slower interconnect where neighbor-only transfers overlap with compute. A
mis-mapped layout stays numerically correct and silently loses the performance
— worth checking when a sharded run is unexpectedly slow.

## Constraints

- `num_attention_heads % tp_size == 0` — TP splits heads at the projections.
- `(num_attention_heads / tp_size) % ulysses_degree == 0` — Ulysses splits the
  **TP-local** heads. `H % U == 0` alone is not sufficient: 56 heads pass with
  `tp=2, ulysses=4` (28 % 4) and fail with `tp=4, ulysses=4` (14 % 4).
- Ring adds no head constraint (it splits rows), but the sequence — including
  any model-specific packing alignment — must divide by `ulysses × ring`, since
  ring adds an outer row split on top of Ulysses's inner one.
- Ring requires an attention backend that declares
  `supports_ring_rotation()` — the per-hop merge needs the kernel's softmax
  LSE. `fa` and `sage_attn` declare it; the launcher auto-selects `fa` when
  unset.
- `USPAttention`'s masked/tail-padded text path and its replicated-prefix,
  -suffix, and -kv-prefix paths all support ring: the sharded K/V rotates
  through the ring while the tail-pad or replicated portion is attended
  locally once and combined into the ring result with the same online-softmax
  merge. This covers the joint text+image attention most models use (flux,
  flux_2, qwen_image, zimage, glm_image, ernie_image, and others).
- What still raises `NotImplementedError` under ring: `USPAttention`'s generic
  varlen path (multiple packed segments per row, as used by HunyuanVideo —
  no ring-aware rotation for it yet), and the legacy stacked-QKV
  `UlyssesAttention` layer, now down to one user (Wan's VSA sparse-attention
  variant) that has no softmax LSE to merge and can't gain ring support
  without a different kernel.
- The launcher validates `num_gpus` against the product of the degrees and
  fails fast on any mismatch.

## The Ulysses transport

The all-to-alls normally run over NCCL. On exactly 2 GPUs with peer-to-peer
access, a CUDA-IPC transport replaces them by default: each rank writes its
half directly into the peer's mapped staging buffer, with GPU-side sequence
counters instead of a NCCL rendezvous. An all-to-all is a permutation, never a
reduction, so the transport cannot change results — outputs are bitwise
identical to the NCCL path.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `SGLANG_DIFFUSION_IPC_A2A` | `1` | set `0` to force NCCL |
| `SGLANG_DIFFUSION_IPC_A2A_TIMEOUT_MS` | `10000` | deadlock backstop for the peer wait; on expiry the transport retires on every rank and the request fails rather than returning incomplete data |
| `SGLANG_DIFFUSION_IPC_A2A_MAX_BUFFERS` | `16` | staging pairs kept alive; raise for many-resolution serving |

Independently of the transport, the default path already packs the three
Q/K/V input exchanges into one destination-major collective. A handful of
models (e.g. LTX-2) instead opt into `enable_packed_qkv_input_a2a`, which
pipelines three separate exchanges over a dedicated stream rather than
merging them into one payload — a different trade-off, not a strict
upgrade over the default.

## Which axes tolerate crossing nodes

Each axis has a fixed communication pattern, and the pattern — volume per step,
how often it fires, and whether it can hide behind compute — decides whether the
axis survives the drop from NVLink to the inter-node fabric. Ordered from most
to least cross-node friendly:

| Axis | Pattern | Traffic per denoise step | Cross-node verdict |
| --- | --- | --- | --- |
| Data parallel | none between replicas | zero on the request path | **Best.** Replicas only share startup init and control-op fan-out; no fast interconnect needed at all. |
| CFG parallel | one branch-combine | one latent-sized exchange, once | **Good candidate.** Once per step, small payload, naturally deadline-tolerant. Unmeasured across nodes so far. |
| K/V-gather CP | one K/V all-gather per attention | (R−1)/R of K/V, one unoverlapped burst | **Between Ulysses and ring:** only K/V moves (queries never do), but the burst cannot hide behind compute — prefer ring across nodes, gather at small degrees within one. |
| Ring SP | neighbor-only K/V rotation | K/V ÷ ring_degree, (R−1) hops, overlapped with attention tiles | **Designed for it.** The mechanism now covers most models' joint attention (see Constraints). Actually crossing nodes end-to-end is validated for MiniMax H3 (Ulysses intra-node × ring across, net positive and growing with sequence length); other models' ring support is same-node-validated so far. |
| Ulysses SP | all-to-all | full q/k/v activations, twice per attention, every layer | **Keep intra-node.** All-to-all needs full-bisection bandwidth; across nodes it becomes R² flows with receiver incast. |
| Tensor parallel | all-reduce | hidden-sized reduction per transformer block (×60 blocks for qwen-class DiTs) | **Worst.** Highest frequency, no overlap, already ~70% of sharded kernel time on NVLink. |

Two caveats keep this a map rather than a promise. Cross-node launch
(`--nnodes`/`--node-rank`/`--dist-init-addr`) is merged, and per-model ring
support is broad now (see the constraints above) — but those two facts
together still don't add up to "any model, any node count." The only
configuration actually run end-to-end across nodes is the H3 recipe; other
models' ring support is same-node-validated so far, which means crossing
nodes with them is untested, not disallowed. And the data-parallel row
describes the design: the current
implementation binds each replica's ingress on the local host, so replicas
spanning hosts additionally need per-replica host addressing before `--dp-size`
can place one replica per node.

## Choosing a configuration

Measured guidance rather than rules — the right combination depends on the
model's communication profile and the hardware topology, and legal does not
mean profitable:

- **Multi-branch (true-CFG) models**: CFG parallelism first. Branches run the
  whole DiT independently and combine once per step, avoiding per-layer
  communication entirely.
- **Single-branch image models on 2 GPUs**: Ulysses and TP trade places by
  model. Communication-heavy DiTs measured faster with Ulysses; smaller DiTs
  with TP. Measure both; do not copy a winner across models.
- **Long video / packed sequences**: Ulysses up to the head-divisibility limit,
  then ring for the remaining factor — sequence length scales past the head
  count where Ulysses alone cannot.
- **When Ulysses's head divisibility blocks the degree you need** (`H/T` not
  divisible by the target `U`): the row-splitting slot sidesteps it — ring
  today, or `--kv-gather-degree` (it splits rows, so it adds no head
  constraint either).
- **TP beyond 2 ranks** rarely improves image-DiT latency: the per-block
  all-reduce grows with rank count faster than the GEMM savings.

## Data parallelism

`--dp-size N` runs N full engine replicas on `num_gpus / N` GPUs each, every
replica with its own ingress. Generation requests round-robin across replicas,
realtime sessions stick to the replica holding their state, and control
operations (weights, LoRA, memory occupation, shutdown) apply to every replica;
replicas exchange nothing on the request path. Monolithic serving only, and the
ingress currently binds on the local host — one replica per node needs
per-replica host addressing first.
