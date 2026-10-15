# SPDX-License-Identifier: Apache-2.0
# Ported from vLLM PR #59705 / ROCm/ATOM#2419 (MIT); rewritten for SGLang's modules.
"""Views of one sparse MoE layer, checked against what the mono kernels read.

The kernels read tensors through raw pointers: a differently-laid-out weight
would not fault, it would produce plausible garbage. So every assumption is
asserted here once and a model that does not match is refused
(``MonoRefused``) rather than misread.

Two kinds of tensor are private copies rather than views of the original path:

- The qkv / index projection and o_proj. The public Quark MiniMax-M3 export
  leaves them in bf16, while K1 / K4's GEMVs are
  per-token-activation x per-channel-weight FP8 on aiter's (16, 16) preshuffled
  layout. A separate copy is quantized once per channel (amax / 448), keeping
  the original module-path weights unchanged.
- The router gate, kept as bf16 (K4's GEMV is bf16; SGLang holds it in fp32).

The MXFP4 expert tensors, the norms, the rotary table and the routing bias are
read in place.
"""

from __future__ import annotations

import msgspec
import torch

from sglang.kernels.ops.mono.gfx950.minimax_m3.config import (
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    N_ROUTED,
    O_K,
    ROTARY_DIM,
    TOP_K,
    TOPK_BLOCKS,
    IndexHeads,
)
from sglang.srt.layers.mono.guards import MonoRefused

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoRefused(what)


def _attr(obj, name: str, what: str):
    """``obj.name``, or a refusal naming the type that lacks it."""
    got = getattr(obj, name, None)
    _need(got is not None, f"{what}: {type(obj).__name__} has no {name}")
    return got


def linear_to_ptpc_fp8(
    weight: torch.Tensor,
    rows: int,
    cols: int,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a public M3 bf16 attention weight for the mono FP8 GEMVs."""
    from aiter.ops.shuffle import shuffle_weight

    _need(
        tuple(weight.shape) == (rows, cols),
        f"{name}: weight {weight.dtype} {tuple(weight.shape)}, want ({rows}, {cols})",
    )
    _need(
        weight.dtype == torch.bfloat16,
        f"{name}: expected bf16 M3 attention weight, got {weight.dtype}",
    )
    deq = weight.data.to(torch.float32)
    amax = deq.abs().amax(dim=1).clamp_(min=torch.finfo(torch.float32).tiny)
    scale = amax / FP8_MAX
    q = (deq / scale[:, None]).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    shuffled = shuffle_weight(q.contiguous(), layout=(16, 16))
    return shuffled.contiguous(), scale.contiguous()


def _bf16_vec(t: torch.Tensor, n: int, name: str) -> torch.Tensor:
    _need(
        t.dtype == torch.bfloat16 and t.numel() == n and t.is_contiguous(),
        f"{name}: {t.dtype} {tuple(t.shape)}, want contiguous bf16 [{n}]",
    )
    return t.data


def _gemma_norm_weight(module, n: int, name: str) -> torch.Tensor:
    """The standard RMSNorm weight of a GemmaRMSNorm: the kernels apply ``w + 1``."""
    _need(
        type(module).__name__ == "GemmaRMSNorm",
        f"{name}: {type(module).__name__}, kernels are built for GemmaRMSNorm",
    )
    return _bf16_vec(module.weight, n, name)


def cos_sin_bf16(rotary_emb, layer_id: int, tables: dict) -> torch.Tensor:
    """The [max_pos, rotary_dim] cos | sin table as bf16, one copy per source table
    in ``tables`` (SGLang hands every layer the same cached rotary module)."""
    t = rotary_emb.cos_sin_cache
    _need(
        t.dim() == 2 and t.shape[1] == ROTARY_DIM,
        f"layer {layer_id}: cos/sin cache {t.dtype} {tuple(t.shape)}",
    )
    got = tables.get(t.data_ptr())
    if got is None:
        got = tables[t.data_ptr()] = t.data.to(torch.bfloat16).contiguous()
    return got


def _ones(device) -> torch.Tensor:
    return torch.ones(1, dtype=torch.float32, device=device)


def kv_scale_tensor(value, device, name: str) -> torch.Tensor:
    """A cache scale as the one-fp32 device tensor the kernels dereference."""
    if value is None:
        return _ones(device)
    if isinstance(value, torch.Tensor):
        _need(value.numel() == 1, f"{name}: {tuple(value.shape)}, want one scale")
        return value.data.to(device=device, dtype=torch.float32).reshape(1)
    return torch.tensor([float(value)], dtype=torch.float32, device=device)


class SparseMoeLayer(msgspec.Struct, frozen=True):
    """Everything K1 / K4 read for one sparse-attention MoE layer."""

    layer_id: int
    attn: object  # MiniMaxM3Attention
    g_in: torch.Tensor
    w_qkv: torch.Tensor
    s_qkv: torch.Tensor
    g_q: torch.Tensor
    g_k: torch.Tensor
    g_iq: torch.Tensor
    g_ik: torch.Tensor
    cos_sin: torch.Tensor
    w_o: torch.Tensor
    s_o: torch.Tensor
    g_post: torch.Tensor
    gate: torch.Tensor
    bias: torch.Tensor
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    @staticmethod
    def from_layer(
        layer, layer_id: int, heads: IndexHeads, cos_sin_tables: dict
    ) -> SparseMoeLayer:
        _need(heads.count == 1, "indexer context parallelism is not ported")
        attn = layer.self_attn
        _need(
            layer.is_layer_sparse and attn.is_sparse_attention_layer,
            f"layer {layer_id}: not a sparse-attention MoE layer",
        )
        _need(
            attn.disable_index_value,
            f"layer {layer_id}: the indexer carries values; kernels serve K-only indexers",
        )
        _need(
            (attn.num_heads, attn.num_kv_heads, attn.num_idx_heads, attn.head_dim,
             attn.idx_head_dim)
            == (LOCAL_Q_HEADS, LOCAL_KV_HEADS, heads.count, HEAD_DIM, HEAD_DIM),
            f"layer {layer_id}: heads {attn.num_heads}/{attn.num_kv_heads}/"
            f"{attn.num_idx_heads}, dims {attn.head_dim}/{attn.idx_head_dim}",
        )  # fmt: skip
        _need(
            attn.rotary_dim == ROTARY_DIM,
            f"layer {layer_id}: rotary_dim {attn.rotary_dim}",
        )
        _need(
            attn.use_gemma_norm and attn.qk_norm_type == "per_head",
            f"layer {layer_id}: kernels are built for per-head gemma qk norms",
        )

        # qkv | index_q | index_k: the fused projection SGLang builds after
        # loading, else the two bf16 projections concatenated in that order.
        fused = attn._fused_qkv_index
        if fused is not None:
            weight = fused.weight.data
        else:
            qp, ip = attn.qkv_proj, attn.index_qkv_proj
            _need(
                qp.weight.dtype == ip.weight.dtype == torch.bfloat16,
                f"layer {layer_id}: qkv ({qp.weight.dtype}) and index "
                f"({ip.weight.dtype}) projections are quantized differently",
            )
            weight = torch.cat([qp.weight.data, ip.weight.data], dim=0)
        w_qkv, s_qkv = linear_to_ptpc_fp8(
            weight, heads.rows, HIDDEN, f"layer {layer_id} qkv"
        )
        w_o, s_o = linear_to_ptpc_fp8(
            attn.o_proj.weight.data, HIDDEN, O_K, f"layer {layer_id} o_proj"
        )

        moe = layer.mlp
        experts = moe.experts
        _need(
            experts.moe_ep_rank == 0
            and experts.num_local_experts == N_ROUTED + 1
            and experts.num_fused_shared_experts == 1,
            f"layer {layer_id}: experts {experts.num_local_experts} local / "
            f"{experts.num_fused_shared_experts} fused shared; kernels need "
            f"{N_ROUTED} routed + the fused shared expert with native MXFP4 "
            "shared-expert fusion and no expert parallelism",
        )
        # FusedMoE counts the fused shared expert in its top_k
        routed_top_k = int(experts.top_k) - int(experts.num_fused_shared_experts)
        _need(
            routed_top_k == TOP_K,
            f"layer {layer_id}: {routed_top_k} routed experts a token, built for {TOP_K}",
        )
        _need(
            experts.intermediate_size_per_partition == INTER,
            f"layer {layer_id}: expert intermediate {experts.intermediate_size_per_partition}"
            f" a rank, built for {INTER}",
        )
        E = N_ROUTED + 1
        w13 = experts.w13_weight
        w2 = experts.w2_weight
        _need(
            getattr(w13, "is_shuffled", False) and getattr(w2, "is_shuffled", False),
            f"layer {layer_id}: expert weights are not AITER-shuffled MXFP4",
        )
        _need(
            w13.dtype == torch.uint8
            and w2.dtype == torch.uint8
            and w13.numel() == E * 2 * INTER * HIDDEN // 2
            and w2.numel() == E * HIDDEN * INTER // 2,
            f"layer {layer_id}: expert weight {w13.dtype} {tuple(w13.shape)} / "
            f"{w2.dtype} {tuple(w2.shape)}",
        )
        s13 = _attr(experts, "w13_weight_scale", f"layer {layer_id} experts")
        s2 = _attr(experts, "w2_weight_scale", f"layer {layer_id} experts")
        _need(
            s13.dtype == torch.uint8
            and s2.dtype == torch.uint8
            and s13.numel() == E * 2 * INTER * HIDDEN // 32
            and s2.numel() == E * HIDDEN * INTER // 32,
            f"layer {layer_id}: expert scales {s13.dtype} {s13.numel()} / "
            f"{s2.dtype} {s2.numel()}",
        )

        gate = moe.gate.weight
        _need(
            tuple(gate.shape) == (N_ROUTED, HIDDEN),
            f"layer {layer_id}: gate {tuple(gate.shape)}",
        )
        bias = moe.e_score_correction_bias
        _need(
            bias is not None and bias.numel() == N_ROUTED,
            f"layer {layer_id}: no routing bias correction",
        )
        radix = attn.attn
        dev = w13.device

        return SparseMoeLayer(
            layer_id=layer_id,
            attn=attn,
            g_in=_gemma_norm_weight(layer.input_layernorm, HIDDEN, "input_layernorm"),
            w_qkv=w_qkv,
            s_qkv=s_qkv,
            g_q=_gemma_norm_weight(attn.q_norm, HEAD_DIM, "q_norm"),
            g_k=_gemma_norm_weight(attn.k_norm, HEAD_DIM, "k_norm"),
            g_iq=_gemma_norm_weight(attn.index_q_norm, HEAD_DIM, "index_q_norm"),
            g_ik=_gemma_norm_weight(attn.index_k_norm, HEAD_DIM, "index_k_norm"),
            cos_sin=cos_sin_bf16(attn.rotary_emb, layer_id, cos_sin_tables),
            w_o=w_o,
            s_o=s_o,
            g_post=_gemma_norm_weight(
                layer.post_attention_layernorm, HIDDEN, "post_attention_layernorm"
            ),
            gate=gate.data.to(torch.bfloat16).contiguous(),
            bias=bias.data.to(torch.float32).contiguous(),
            w13=w13.data,
            s13=s13.data,
            w2=w2.data,
            s2=s2.data,
            k_scale=kv_scale_tensor(radix.k_scale, dev, "k_scale"),
            v_scale=kv_scale_tensor(radix.v_scale, dev, "v_scale"),
        )


def sparse_selection(config) -> tuple[int, int, int]:
    """(top-k, init, local) block counts of the model's sparse attention."""
    cfg = getattr(config, "sparse_attention_config", None) or {}
    got = (
        cfg.get("sparse_topk_blocks"),
        cfg.get("sparse_init_block"),
        cfg.get("sparse_local_block"),
    )
    _need(None not in got, f"sparse_attention_config lacks the selection: {cfg}")
    _need(
        tuple(int(x) for x in got) == (TOPK_BLOCKS, 0, 1),
        f"sparse selection {got}, kernels are built for {(TOPK_BLOCKS, 0, 1)}",
    )
    _need(
        int(cfg.get("sparse_block_size", 0)) == 128
        and str(cfg.get("sparse_score_type", "max")) == "max",
        f"sparse block {cfg.get('sparse_block_size')} / score {cfg.get('sparse_score_type')}",
    )
    return tuple(int(x) for x in got)
