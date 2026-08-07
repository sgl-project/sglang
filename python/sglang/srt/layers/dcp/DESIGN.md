# Decoupling Decode Context Parallelism from Attention Tensor Parallelism

**Status:** Phase 1 landed. Phases 2 and 3 proposed, not started.
**Area:** `python/sglang/srt/layers/dcp/`, MLA + GQA/MHA attention backends, `distributed/parallel_state.py`
**Related:** [#29736](https://github.com/sgl-project/sglang/issues/29736) (DCP roadmap), [#19436](https://github.com/sgl-project/sglang/issues/19436) (Helix: A2A + decoupled attention/FFN), [Helix paper (arXiv 2507.07120)](https://arxiv.org/abs/2507.07120)

---

## 1. Background: how DCP works today

Decode Context Parallelism shards the KV cache along the sequence dimension so
that a decode step does not have to replicate KV across every TP rank. Token at
global position `p` lives on the DCP rank `p % dcp_size`; each rank therefore
sees only a slice of the sequence and computes a *partial* attention result,
which the ranks merge by log-sum-exp.

Attention weights, however, are sharded on a different axis entirely:

```python
# models/deepseek_v2.py
attn_tp_size = get_parallel().attn_tp_size   # tp_size // attn_dp_size // attn_cp_size
self.num_local_heads = num_heads // attn_tp_size
```

There is no `dcp_size` term in that expression. DCP instead *borrows the head
dimension back at run time*. Every layer, every step:

1. **All-gather Q** across the DCP group, widening `num_local_heads` to
   `num_local_heads * dcp_size`.
2. Run the kernel on the widened head set against this rank's KV slice, yielding
   a partial output plus an LSE.
3. **Merge** across the DCP group, which reduces over the sequence split and
   scatters over heads, returning `num_local_heads` again — the layout `o_proj`
   expects.

Two merge patterns exist, selected by `--dcp-comm-backend`:

| Backend | Pattern | Ops/layer |
|---|---|---|
| `ag_rs` | all-gather LSE, correct, reduce-scatter output | 3 |
| `a2a` | one fused all-to-all of (output, LSE), local Triton combine | 2 |
| `fi_a2a` | same as `a2a` with the exchange on FlashInfer MNNVL | 2 |

## 2. The coupling

The *effective* attention parallelism under DCP is already the Helix layout:
heads are sharded `tp_size / dcp_size` ways and KV is sharded `dcp_size` ways.
Both merge patterns already deposit heads in exactly the flat-TP order `o_proj`
wants. Nothing about the math needs to change.

What is coupled is **where that layout lives**. It exists only as a per-layer
collective on activations, never as a property of the weights. Consequences:

- An extra NCCL op per layer per step for the Q all-gather.
- `--dcp-replicate-q-proj` exists as a workaround: at load time it all-gathers
  `q_b_proj` and `w_kc` into *duplicate* full-head buffers (`q_b_proj_qrep_weight`,
  `w_kc_qrep`), bf16-and-unquantized only, keeping both copies resident.
- The widened head count `num_local_heads * attn_dcp_size` was re-derived
  independently at five call sites, the comm-backend dispatch at three, and the
  "which head chunk do I keep" rule was implicit in three separate reduction
  primitives.
- For GQA, `attn_tp_size` still exceeds `num_kv_heads` on wide TP, so KV heads
  stay replicated. DCP shards tokens but does not address head replication.

## 3. The invariant that bounds the design space

Heads **must** be replicated across the DCP dimension. If rank A and rank B hold
different sequence slices, they must evaluate the *same* heads or their partials
have nothing to merge. So for the attention block:

```
attn_head_shard_degree x dcp_size == attn_tp_size
```

This is forced by correctness, not convention. There is only one valid layout;
the open questions are how it is spelled, where it is materialized, and how far
it reaches.

A second, sharper identity governs alignment with `o_proj`. The Q all-gather
concatenates partners' head shards in DCP-rank order, and both reductions are
structural — reduce-scatter and all-to-all each hand rank *r* chunk *r*. For the
chunk a rank keeps to be the head shard its `o_proj` weights were loaded for:

```
dcp_rank == attn_tp_rank % dcp_size
```

which holds exactly while DCP groups are contiguous, lowest-order slices of the
attention TP group.

---

## 4. Phase 1 (landed): a first-class attention comm group

`DcpAttnComm` in `attn_comm.py` becomes the single owner of the three facts
listed above. It is a read-through accessor — topology is resolved through
`get_parallel()` on each access rather than captured at construction — so
`ParallelContext.override()` keeps working in tests and the object can exist
before the DCP process group does.

### 4.1 Surface

| Member | Purpose |
|---|---|
| `enabled` / `size` / `rank` / `group` | Topology; `size` and `rank` degrade to 1/0 instead of raising when DCP is off |
| `comm_backend` | `ag_rs` \| `a2a` \| `fi_a2a` |
| `num_kernel_heads(n)` | Widen an `o_proj`-layout head count to what the kernel evaluates |
| `head_shard_index` | Which chunk of the widened head set this rank keeps |
| `local_head_offset(n)` / `narrow_local_heads(t, n)` | Select this rank's shard without communicating |
| `check_layout()` | Enforce the two identities in §3 |
| `gather_q_mla(...)` / `gather_q_heads(...)` | Widen queries, including the symmetric-memory staging |
| `combine_mla(...)` / `combine_mha(...)` | Merge partials, dispatching on `comm_backend` |
| `combine_mha_with_lse(...)` | Merge and return the merged LSE; `ag_rs`-only by construction |
| `init_workspace()` | Allocate the `fi_a2a` MNNVL workspace pre-capture |

Two normalizations are worth calling out. `combine_mla` transposes the `ag_rs`
result internally, so every backend returns the same layout rather than leaving
a stray `.transpose(0, 1)` at the call site. And `combine_mha_with_lse` is a
separate method rather than a `return_lse=True` flag, because only `ag_rs` can
surface the merged LSE — putting that constraint in the signature makes it
visible instead of a silent fallback.

### 4.2 Call sites migrated

| File | Was | Now |
|---|---|---|
| `models/deepseek_v2.py` | `num_local_heads * attn_dcp_size` for `attn_mqa_for_dcp_decode` | `num_kernel_heads(...)`, plus `check_layout()` |
| `.../attention_forward_methods/forward_mla.py` | inline 3-way dispatch, two head-count sites, local `_is_mla_dcp_lse_base_on_e` | `combine_mla(...)`, `num_kernel_heads(...)`, `gather_q_mla(...)` |
| `attention/triton_backend.py` | hardcoded `cp_lse_ag_out_rs_mha`, manual symmetric-memory gather | `combine_mha(...)`, `combine_mha_with_lse(...)`, `gather_q_heads(...)` |
| `attention/flashinfer_mla_backend.py` | `// attn_tp_size * attn_dcp_size` | `num_kernel_heads(...)` |
| `attention/tokenspeed_mla_backend.py` | `num_heads *= attn_dcp_size` | `num_kernel_heads(...)` |
| `model_executor/runner/base_runner.py` | `init_fi_a2a_workspace(dcp_group)` behind a server-args guard | `init_workspace()` |

Reads of `attn_dcp_size`/`attn_dcp_rank` that concern **KV ownership** rather
than head arithmetic (memory pools, `layout.py`, PD disaggregation) were left
alone. They are a separate axis, already centralized in `layers/dcp/layout.py`.

### 4.3 Rank layout, made explicit

Two changes turn the layout from an assumption into a decision:

- `build_dcp_group_ranks()` in `distributed/parallel_state.py` is now the single
  place the DCP group layout is chosen, with the identity it guarantees stated
  in its docstring.
- `ModelRunner.dcp_rank` was `ps.tp_rank % self.dcp_size`, an independent
  re-derivation of the same assumption that would silently diverge from the
  collectives if the layout changed. It is now a property reading the process
  group. KV ownership therefore follows the group's actual layout.

`head_shard_index` is deliberately a named concept rather than a synonym for
`rank`. It is the hook a future non-contiguous layout has to remap, and
`check_layout()` is what will demand that remapping.

### 4.4 Newly rejected configurations

`attn_tp_size % dcp_size != 0` now fails at `initialize_model_parallel()` with an
actionable message, and `check_layout()` re-checks it at backend and model
construction. This was previously unvalidated anywhere — see §5.

---

## 5. Defects surfaced

**No guard between DCP and DP-attention / prefill-CP.** `attn_tp_size` has no
`dcp_size` term and nothing checked that `dcp_size` divides it. The only
structural check was `tp_size % dcp_size == 0`. So
`--tp 8 --enable-dp-attention --dp-size 4 --dcp-size 4` yields `attn_tp_size=2`,
making the widened head count *twice the model's actual head count* and breaking
the `o_proj` alignment identity. Fixed in Phase 1.

**`--dcp-comm-backend a2a` was silently ignored for GQA/MHA.** `server_args`
accepted the flag but `triton_backend.py` called `cp_lse_ag_out_rs_mha`
unconditionally. Now dispatched, covered by a new equivalence test.

**The GQA/triton DCP path is incorrect on CUDA.** With Qwen3-0.6B at
`--tp 2 --dcp-size 2 --attention-backend triton`, output is garbage while
`--dcp-size 1` is coherent. This is **pre-existing** — clean `main` produces
byte-identical garbage — and matches the roadmap's note that the triton CUDA
path is untested (the landed work was AMD/HIP). Not addressed here; needs its
own investigation. It is the reason the end-to-end check for this refactor
proves *equivalence to baseline* rather than correctness.

---

## 6. Phase 2 (proposed): make the attention-TP layout static

### Motivation

Today the decoupled layout is reconstructed per layer per step by the Q
all-gather. Materializing it in the weights instead removes that collective
entirely, taking `ag_rs` from 3 NCCL ops/layer to 2 and `a2a` from 2 to 1.

### Mechanism

Shard `q_proj` / `q_b_proj` and the absorbed `w_kc` over
`attn_tp_size // dcp_size` ranks and replicate them across the DCP dimension,
rather than sharding over `attn_tp_size` and gathering at run time. The output
side needs no change: the merge already re-scatters heads into the flat-TP
layout that `o_proj` and the FFN consume. This is precisely what
`--dcp-replicate-q-proj` approximates, promoted from a load-time patch to a
weight-loading path.

### The HBM cost

This is the load-bearing tradeoff and the reason replication is opt-in today.
For DeepSeek-V3 (`num_heads=128`, `q_lora_rank=1536`, `qk_head_dim=192`,
61 layers) at TP8/DCP8, `q_b_proj` is `1536 x 128 x 192 ≈ 37.7M` params, so
72 MiB/layer in bf16. Sharded 8 ways that is 9 MiB/rank; fully replicated it is
72 MiB/rank. Across 61 layers the delta is roughly 3.8 GiB, plus about 0.8 GiB
for `w_kc` — call it **~4.6 GiB per rank** in the worst case. At TP16/DCP4 the
replication factor is 4 rather than 8 and the cost is proportionally smaller.

Note that a static layout is strictly *cheaper* than today's
`--dcp-replicate-q-proj`, which keeps the gathered buffers **in addition to**
the original sharded weights.

### Keeping prefill cheap

Prefill does not widen heads — it gathers KV instead — so a rank holding the
wide layout would do `dcp_size x` redundant Q-projection work on prefill. The
fix is to hold the wide layout and `narrow()` it to the local TP head shard for
prefill forwards. `layers/cp/cp_decode_attn_tp.py` already implements exactly
this machinery in the opposite direction (`CpDecodeAttnTpContext` slices
replicated weights down to a per-rank shard for decode, patching
`tp_q_head_num` and `input_size_per_partition` and restoring on exit) and is the
obvious thing to generalize.

### Suggested staging

1. Weight-loading path for the wide layout, behind the existing
   `--dcp-replicate-q-proj` flag, replacing the duplicate-buffer implementation.
2. Prefill narrowing, so the flag stops costing prefill throughput.
3. Quantized support — the current workaround is bf16/fp16-only because it
   all-gathers raw `.weight`; a real loading path can handle packed weights and
   scales.
4. Only then consider changing the default.

## 7. Phase 3 (proposed): `--attn-tp-size` and GQA

Issue [#19436](https://github.com/sgl-project/sglang/issues/19436) proposes
`--attention-tensor-parallel-size` as the user-facing knob, with the KV-parallel
degree derived as `tp_size / attn_tp_size`. Since `attn_tp x dcp == tp` is
forced (§3), this is the same layout as Phase 2 under a different
parameterization, and the two should not be built twice.

The one place it is *not* merely cosmetic is GQA. Choosing
`attn_tp_size <= num_kv_heads` is what eliminates KV-head replication when
`tp_size > num_kv_heads` — the headline benefit in the Helix RFC, and something
DCP alone does not deliver. That argues for the knob, but only after the GQA
DCP path is correct on CUDA (§5).

**Naming hazard:** `attn_tp_size` is already taken in SGLang and means
`tp_size // attn_dp_size // attn_cp_size`. It is also the comm group used for
the DP-attention token gather and the `o_proj` all-reduce. Redefining it to
absorb `dcp_size` would touch every model's head computation and every
DP-attention collective. A new name for the head-shard degree is strongly
preferred.

---

## 8. Testing

| Test | Covers |
|---|---|
| `test/registered/dcp/test_dcp_attn_comm_unit.py` | Head arithmetic, head-shard mapping, `check_layout()` accept/reject cases, group layout, LSE base selection (CPU) |
| `test/registered/dcp/test_dcp_mha_combine_backends.py` | `ag_rs` and `a2a` MHA merges agree at 2/4/8 GPUs |
| `test/registered/kernels/test_dcp_lse_combine.py` | Pre-existing; LSE combine kernels and CUDA-graph buffers |
| `test/registered/dcp/test_dcp_layout_unit.py` | Pre-existing; KV ownership and pool index math |

Phase 1 was additionally validated by running Qwen3-0.6B at `--tp 2 --dcp-size 2`
on this branch and on stashed `main`, confirming byte-identical output.

For Phases 2 and 3, the equivalence bar should be the same: a decode result
identical to the AllGather-Q path before switching any default, and the
DSv3.1 TP8/DCP8 GSM8K gate green.

## 9. Open questions

1. **Is the HBM cost acceptable at TP8/DCP8**, or should the wide layout be
   restricted to configurations where the replication factor is small?
2. **Should the merged-LSE constraint on `combine_mha_with_lse` be lifted?**
   `dcp_lse_combine_triton` can return the combined LSE; `dcp_a2a_lse_reduce`
   simply does not plumb it out. Doing so would let the MHA chunked-prefix path
   honour `--dcp-comm-backend`.
3. **What is the right name for the head-shard degree**, given `attn_tp_size` is
   taken (§7)?
4. **Does any real deployment want a non-contiguous DCP layout?** The hook
   (`head_shard_index`) exists, but nothing needs it yet.

## 10. References

- `attn_comm.py` — the abstraction
- `comm.py` — the merge and gather primitives it dispatches to
- `layout.py` — KV ownership index math (`pos % dcp_size == dcp_rank`)
- `planner.py` — extend and decode metadata construction
- `distributed/parallel_state.py::build_dcp_group_ranks` — the rank layout
- `layers/cp/cp_decode_attn_tp.py` — the weight-narrowing precedent for Phase 2
