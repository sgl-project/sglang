"""Kimi-K3.1 gfx950 DSA decode / verify prologue fusion (after vLLM #67847), behind
SGLANG_ENABLE_ROCM_K3_DSA_FUSED_PROLOGUE (default off).

Parts (SGLANG_ROCM_K3_DSA_FUSED_PROLOGUE_PARTS, default all):

- ``gemm``: one BF16 GEMM over ``[fused_qkv_a | wk | weights_proj | g_proj]`` (N = 3808
  at TP8) in place of three (qkv_a, the merged [wk | weights_proj] and the output gate).
  The four module weights become row views of the merged buffer, so the stock path reads
  the same values. The index key / head gate and the output gate are handed to the
  indexer and the gated o_proj of the same layer forward.
- ``index``: the index key LayerNorm, the index-K store (FP8 or MXFP4 pages) and the
  query quant + head-gate weights in one launch (``index_qk_prep_fp8/fp4``).
- ``kv``: the fp8_ds_mla KV store (tile quant + DCP-owner scatter) in one launch.

Every part declines (stock path) outside the shapes it mirrors; the decision depends
only on module setup and the forward mode, which all ranks share.
"""

from __future__ import annotations

import functools
import logging
from types import SimpleNamespace
from typing import Optional, Tuple

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported, print_info_once

logger = logging.getLogger(__name__)

_is_gfx95 = is_gfx95_supported()
PARTS = ("gemm", "index", "kv")


@functools.lru_cache(maxsize=1)
def enabled_parts() -> frozenset:
    """The prologue parts this process runs (empty when the switch is off or not gfx950)."""
    if not (_is_gfx95 and envs.SGLANG_ENABLE_ROCM_K3_DSA_FUSED_PROLOGUE.get()):
        return frozenset()
    raw = envs.SGLANG_ROCM_K3_DSA_FUSED_PROLOGUE_PARTS.get() or ""
    parts = frozenset(p.strip() for p in raw.split(",") if p.strip())
    unknown = parts - set(PARTS)
    if unknown:
        raise ValueError(
            f"SGLANG_ROCM_K3_DSA_FUSED_PROLOGUE_PARTS: unknown {sorted(unknown)}; "
            f"choose from {PARTS}"
        )
    return parts


def part_enabled(part: str) -> bool:
    return part in enabled_parts()


# ---- gemm: the merged projection front


def _bf16_weight(module) -> Optional[torch.Tensor]:
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if module is None or not isinstance(module.quant_method, UnquantizedLinearMethod):
        return None
    weight = module.weight
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        return None
    return weight


class DsaFront(msgspec.Struct, frozen=True):
    """The merged prologue GEMM of one DSA layer and its output split."""

    weight: torch.Tensor
    # (qkv_a, index key, index head gate, output gate) widths
    split: Tuple[int, ...]


def merge_front_weights(attn) -> Optional[DsaFront]:
    """Post-load, before graph capture: ``attn``'s merged front, or None when the layer
    cannot take it. Re-points the four module weights to row views of the buffer."""
    from sglang.srt.models.utils import bind_or_assign

    if not part_enabled("gemm"):
        return None
    indexer = attn.indexer
    prologue = None if indexer is None else indexer.rocm_nope_prologue
    if (
        prologue is None
        or not attn.use_output_gate
        or "_kimi_split_gguf_kv_b" in vars(attn)
    ):
        return None
    mods = [
        attn.fused_qkv_a_proj_with_mqa,
        indexer.wk,
        indexer.weights_proj,
        attn.g_proj,
    ]
    weights = [_bf16_weight(m) for m in mods]
    if any(w is None for w in weights) or len({w.shape[1] for w in weights}) != 1:
        return None
    sizes = tuple(w.shape[0] for w in weights)
    if sizes[1] != indexer.head_dim or sizes[2] != indexer.n_heads:
        return None
    previous = attn.dsa_front.weight if attn.dsa_front is not None else None
    merged = bind_or_assign(
        previous, torch.cat([w.data for w in weights], dim=0).contiguous()
    )
    off = 0
    for mod, n in zip(mods, sizes):
        mod.weight.data = merged[off : off + n]
        off += n
    # the stock [wk | weights_proj] GEMM keeps its own (now aliased) rows
    if prologue.k_gate_weight is not None:
        prologue.k_gate_weight = merged[sizes[0] : sizes[0] + sizes[1] + sizes[2]]
    print_info_once(
        f"DSA fused prologue: merged [qkv_a | wk | weights_proj | g_proj] GEMM, "
        f"N={merged.shape[0]} K={merged.shape[1]}"
    )
    return DsaFront(merged, sizes)


def front_applies(attn, hidden_states, forward_batch) -> bool:
    from sglang.srt.layers.cp.utils import is_cp_active

    mode = forward_batch.forward_mode
    return (
        attn.dsa_front is not None
        and isinstance(hidden_states, torch.Tensor)
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.ndim == 2
        and (mode.is_decode_or_idle() or mode.is_target_verify())
        and "_mono_pre" not in vars(attn)
        and not is_cp_active(forward_batch)
    )


def run_front(attn, hidden_states: torch.Tensor) -> torch.Tensor:
    """The merged GEMM: returns the qkv_a latent and hands the index key / gate to the
    indexer prologue and the output gate to the gated o_proj (consumed this forward)."""
    front = attn.dsa_front
    # qkv_a's own unquantized route (AITER tgemm / hipBLASLt), as the stock GEMMs take
    out = attn.fused_qkv_a_proj_with_mqa.quant_method.apply(
        SimpleNamespace(weight=front.weight), hidden_states
    )
    qkv, key, head_gate, out_gate = out.split(front.split, dim=-1)
    attn.indexer.rocm_nope_prologue.front_key_gate = (key, head_gate)
    attn.front_output_gate = out_gate
    return qkv


def clear_front(attn) -> None:
    """Drop what run_front handed over and nobody consumed (a skipped indexer)."""
    attn.front_output_gate = None
    if attn.indexer is not None and attn.indexer.rocm_nope_prologue is not None:
        attn.indexer.rocm_nope_prologue.front_key_gate = None


# ---- index: LayerNorm + index-K store + query quant + head gate


def _key_norm_params(indexer):
    from sglang.srt.layers.layernorm import LayerNorm

    kn = indexer.k_norm
    if not (
        isinstance(kn, LayerNorm)
        and kn.elementwise_affine
        and kn.use_bias
        and kn.weight.dtype == torch.float32
        and kn.bias.dtype == torch.float32
    ):
        return None
    return kn.weight, kn.bias, kn.variance_epsilon


def index_qk_prep(indexer, forward_batch, layer_id: int, query, gate, key):
    """(query, weights) as RocmNoPEPrologue.forward returns them, with the key stored;
    None declines (nothing was written)."""
    from sglang.kernels.ops.attention.dsa import dsa_prologue_hip as kernels
    from sglang.srt.layers.attention.dsa import dsa_indexer as stock
    from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    norm = _key_norm_params(indexer)
    if (
        norm is None
        or not stock._use_aiter
        or indexer.head_dim != 128
        or key.dtype != torch.bfloat16
        or key.ndim != 2
        or key.stride(-1) != 1
        or query.dtype != torch.bfloat16
        or gate.dtype != torch.bfloat16
        or gate.stride(-1) != 1
    ):
        return None
    pool = get_token_to_kv_pool()
    if isinstance(pool, LayerSplitDSATokenToKVPool):
        return None
    fp4 = indexer.fp4_index_k
    if not fp4 and (indexer.block_size != 128 or pool.index_page_size % 16):
        return None
    # the stock store's pool hooks, in its order
    if hasattr(pool, "invalidate_index_buffer_for_layer"):
        pool.invalidate_index_buffer_for_layer(layer_id)
    owned = not hasattr(pool, "_is_layer_owned") or pool._is_layer_owned(layer_id)
    dcp_size, dcp_rank = stock._index_k_dcp()
    loc = forward_batch.out_cache_loc.contiguous()
    q3 = query.view(-1, indexer.n_heads, indexer.head_dim)
    weight, bias, eps = norm
    common = dict(
        query=q3,
        gate=gate,
        softmax_scale=indexer.softmax_scale,
        key=key,
        norm_weight=weight,
        norm_bias=bias,
        eps=eps,
        loc=loc,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
    )
    if fp4:
        payload, scale = (
            pool.get_index_k_fp4_buffers(layer_id) if owned else (None, None)
        )
        return kernels.index_qk_prep_fp4(payload=payload, scale=scale, **common)
    cache = pool.get_index_k_with_scale_buffer(layer_id=layer_id) if owned else None
    return kernels.index_qk_prep_fp8(
        scale_fmt=indexer.scale_fmt,
        index_k_cache=cache,
        page_size=pool.index_page_size,
        preshuffle=stock._use_aiter_preshuffle,
        **common,
    )


# ---- kv: fp8_ds_mla quant + scatter


def fp8_ds_mla_store(pool, dst_buffer, loc, cache_k_nope, cache_k_rope) -> bool:
    """The pool's dsa_kv_cache_store_fp8 write in one launch; False declines."""
    from sglang.kernels.ops.attention.dsa import dsa_prologue_hip as kernels
    from sglang.kernels.ops.kvcache.paged_rows import paged_layout
    from sglang.srt.runtime_context import get_parallel

    if not part_enabled("kv"):
        return False
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = 0 if cache_k_rope is None else cache_k_rope.shape[-1]
    if (
        cache_k_nope.dtype != torch.bfloat16
        or cache_k_nope.stride(-1) != 1
        or nope_dim % 128
        or rope_dim not in (0, 64)
        or (
            rope_dim
            and (cache_k_rope.dtype != torch.bfloat16 or cache_k_rope.stride(-1) != 1)
        )
        or (cache_k_nope.ndim == 3 and cache_k_nope.shape[1] != 1)
        or paged_layout(dst_buffer) is not None
        or dst_buffer.stride(-1) != 1
    ):
        return False
    if pool.write_loc_is_dcp_resolved or pool.dcp_replicated:
        dcp_size, dcp_rank = 1, 0
    else:
        parallel = get_parallel()
        dcp_size, dcp_rank = parallel.attn_dcp_size, parallel.attn_dcp_rank
    kernels.fp8_ds_mla_store(
        dst_buffer,
        loc.contiguous(),
        cache_k_nope,
        cache_k_rope,
        scale_ue8m0=pool.kv_scale_ue8m0,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
    )
    return True
