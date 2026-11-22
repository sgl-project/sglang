"""Hopper-only Muse Spark paths: the gate shared by every one of them, the fused attention
(attn_srv verify, pfa extend / mixed, the SP o_proj push) and TP row-split pooler projections."""

import functools
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.cp.utils import is_cp_active
from sglang.srt.layers.muse_spark.fused import attn_srv, pfa
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    get_breakable_capture_recorder,
)
from sglang.srt.model_executor.runner_utils.prefill_graph import (
    is_in_full_prefill_graph,
)
from sglang.srt.runtime_context import get_forward, get_memory

_HOPPER = (9, 0)
# pooler / unpooler gemm_ag GPUs; must match gemm_ag_supported
SLICED_PROJECTION_CAPS = ((10, 3), _HOPPER)
# prefill sizes split pooler / unpooler rows over TP (GEMM cost > all-gather)
PREFILL_SLICE_MIN_TOKENS = 1024
# pfa packs each key read; below this kv-weighted mean extend (measured break-even) FA3 wins
PREFILL_MIN_MEAN_EXTEND = 32


def _on_hopper(device=None) -> bool:
    return (
        torch.version.hip is None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(device) == _HOPPER
    )


@functools.cache
def hopper_kernels_enabled(device=None) -> bool:
    """Gate of every Muse Spark Hopper path; other GPUs keep the stock code."""
    return envs.SGLANG_MUSE_SPARK_HOPPER_KERNELS.get() and _on_hopper(device)


def _row_split_group(x: torch.Tensor):
    if (
        x.dim() != 2
        or x.shape[0] < PREFILL_SLICE_MIN_TOKENS
        or not x.is_cuda
        or not hopper_kernels_enabled(x.device)
    ):
        return None
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    return None if get_forward().sp_active or group.world_size <= 1 else group


def _gather_rows(group, local: torch.Tensor, num_tokens: int) -> torch.Tensor:
    out = local.new_empty((local.shape[0] * group.world_size, local.shape[1]))
    group.all_gather_into_tensor(out, local)
    return out[:num_tokens]


def row_split(group, x: torch.Tensor, fn) -> torch.Tensor:
    """``fn`` over this rank's block of ``x``'s rows, all-gathered back into ``x``'s rows."""
    t = x.shape[0]
    rows = -(-t // group.world_size)
    local = fn(x[min(group.rank_in_group * rows, t) :][:rows])
    if local.shape[0] < rows:
        local = torch.cat(
            [local, local.new_zeros((rows - local.shape[0], local.shape[1]))]
        )
    return _gather_rows(group, local, t)


def sliced_prefill_projection(
    x: torch.Tensor, weight: torch.Tensor
) -> Optional[torch.Tensor]:
    group = _row_split_group(x)
    if group is None:
        return None
    from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

    return row_split(
        group, x.contiguous(), lambda rows: bf16_gemm_dispatch(rows, weight, None)
    )


def reduce_scatter_projection(
    partial: torch.Tensor, weight: torch.Tensor
) -> Optional[torch.Tensor]:
    """(sum of ``partial`` over TP ranks) @ weight.T."""
    group = _row_split_group(partial)
    if group is None:
        return None
    from sglang.srt.layers.layernorm_sp import sp_reduce_scatter
    from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

    return _gather_rows(
        group,
        bf16_gemm_dispatch(sp_reduce_scatter(partial), weight, None),
        partial.shape[0],
    )


def prepare_attention(module) -> None:
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    window, rope = module.attn.sliding_window_size, module.rotary_emb
    sliding = window is not None and window > -1
    theta = (
        float(rope.base)
        if sliding
        and type(rope) is RotaryEmbedding
        and not rope.is_neox_style
        and rope.rotary_dim == module.head_dim
        else None
    )
    # SGLang window excludes the query (FA3 window_size_left); the kernel's includes it
    covered = theta is not None and window + 1 == 2048 if sliding else rope is None
    module._fused_attn = (
        covered
        and module.use_qk_norm
        and module.use_o_gate
        and module.use_o_norm
        and hopper_kernels_enabled()
        and (module.num_heads, module.num_kv_heads, module.head_dim) == (16, 2, 64)
        and torch.get_default_dtype() == torch.bfloat16
        and module.norm_eps == 1e-5
        and not get_memory().enable_unified_memory
    )
    module._fused_sliding = sliding
    module._fused_cfg = {
        "window": window + 1 if sliding else 0,
        "rope_theta": theta if sliding else 0.0,
        "qk_scale_factor": module.scale_query_by,
        "sm_scale": module.head_dim**-0.5,
    }


def _fa3_ok(module, backend) -> bool:
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

    return (
        module._fused_attn
        and isinstance(backend, FlashAttentionBackend)
        and not backend.use_mla
        and backend.fa_impl_ver == 3
        and backend.topk <= 1
        and backend.page_size == 1
        and backend.kv_cache_dtype == torch.float8_e4m3fn
        and module.attn.k_scale is None
        and module.attn.v_scale is None
    )


def verify_max_batch(module, backend) -> int:
    return attn_srv.MAX_BATCH if _fa3_ok(module, backend) else 0


def run_attention(
    module,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    forward_batch,
    push=None,
    last_only=False,
) -> Optional[torch.Tensor]:
    if is_cp_active(forward_batch):
        return None
    if not forward_batch.forward_mode.is_target_verify():
        return run_prefill_attention(
            module, qkv, positions, forward_batch, push, last_only
        )
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

    spec = forward_batch.spec_info
    if not isinstance(spec, DFlashVerifyInput) or spec.ragged_verify_layout is not None:
        return None
    return _run_verify_attention(module, qkv, forward_batch, (None, None))


def _fa3_metadata(module):
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    md = backend.forward_metadata if _fa3_ok(module, backend) else None
    return (
        backend,
        None
        if md is None or md.page_table is None or md.cache_seqlens_int32 is None
        else md,
    )


def _paged_kv(module, backend, md, forward_batch):
    page_table, loc = md.page_table, forward_batch.out_cache_loc
    if module._fused_sliding and backend.use_sliding_window_kv_pool:
        page_table, loc = (
            md.swa_page_table,
            md.swa_out_cache_loc
            if md.swa_out_cache_loc is not None
            else backend.token_to_kv_pool.translate_loc_from_full_to_swa(loc),
        )
    return (
        None
        if page_table is None
        else (page_table, loc, *backend.get_paged_mha_kv_cache(module.attn))
    )


def _run_verify_attention(module, qkv, forward_batch, tree) -> Optional[torch.Tensor]:
    backend, md = _fa3_metadata(module)
    if md is None:
        return None
    batch = md.cache_seqlens_int32.shape[0]
    tokens = qkv.shape[0]
    if tokens != 8 * batch or not 0 < batch <= attn_srv.MAX_BATCH:
        return None
    kv = _paged_kv(module, backend, md, forward_batch)
    if kv is None:
        return None
    page_table, cache_loc, k_cache, v_cache = kv
    out = attn_srv.kernel_fn(
        qkv.view(batch, 8, -1),
        md.cache_seqlens_int32,
        cache_loc[:tokens],
        k_cache,
        v_cache,
        page_table,
        module._fused_cfg,
        tree,
    )
    return out.view(tokens, -1)


def _prefill_args(module, forward_batch):
    backend, md = _fa3_metadata(module)
    if md is None:
        return None
    ext, seq = forward_batch.extend_seq_lens_cpu, forward_batch.seq_lens_cpu
    if (
        ext is None
        or seq is None
        or md.cu_seqlens_q is None
        or md.cu_seqlens_q.shape[0] < md.cache_seqlens_int32.shape[0] + 1
    ):
        return None
    ext, seq = (v.tolist() if hasattr(v, "tolist") else list(v) for v in (ext, seq))
    kv = _paged_kv(module, backend, md, forward_batch)
    return (
        None if kv is None else (*kv, md.cache_seqlens_int32, md.cu_seqlens_q, ext, seq)
    )


def sp_o_proj_push_plan(module, num_tokens: int) -> Optional[tuple]:
    o = module.o_proj
    b = getattr(o, "_sp_a2a_push", None)
    if (
        b is None
        or not module._fused_attn
        or getattr(o, "_sp_full_weight", None) is None
        or num_tokens < o._sp_full_weight_min_tokens
    ):
        return None
    return (
        b.o_proj_push_plan(num_tokens, module.q_size)
        if get_forward().sp_active
        else None
    )


def _run_pfa(module, qkv, positions, out, a, push=None, last_only=False) -> None:
    page_table, cache_loc, k_cache, v_cache, cache_seqlens, cu_seqlens_q, ext, seq = a
    n, window = sum(ext), module._fused_cfg["window"]
    if n > 0:
        pfa.forward(
            qkv[:n],
            cache_loc[:n],
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            sum(min(k, e + window + 128) if window else k for e, k in zip(ext, seq)),
            module.head_dim**-0.5,
            module.scale_query_by,
            module.norm_eps,
            out=out if last_only else out[:n],
            window=window,
            cos_sin=module.rotary_emb.cos_sin_cache if window else None,
            positions=positions[:n] if window else None,
            push=push,
            last_only=last_only,
        )
    elif push:
        module.o_proj._sp_a2a_push.push_rows(push, push[4], push[5])
    if not (push or last_only) and n < out.shape[0]:
        out[n:].zero_()


def _run_mixed_verify(
    module, qkv, positions, out, a, forward_batch, push=None, last_only=False
) -> bool:
    page_table, cache_loc, k_cache, v_cache, cache_seqlens, cu_seqlens_q, ext, seq = a
    tails = forward_batch.mixed_verify_rows or 0
    rows = len(ext) - tails
    if rows < 0 or tails > attn_srv.MAX_BATCH or any(e != 8 for e in ext[rows:]):
        return False
    n, t = sum(ext[:rows]), 8 * tails
    o = rows if last_only else n
    if n:
        _run_pfa(
            module,
            qkv[:n],
            positions[:n],
            out[:o],
            (
                page_table[:rows],
                cache_loc[:n],
                k_cache,
                v_cache,
                cache_seqlens[:rows],
                cu_seqlens_q[: rows + 1],
                ext[:rows],
                seq[:rows],
            ),
            push,
            last_only,
        )
    elif push:
        module.o_proj._sp_a2a_push.push_rows(push, t, push[5] - t)
    attn_srv.kernel_fn(
        qkv[n : n + t].view(tails, 8, -1),
        cache_seqlens[rows:],
        cache_loc[n : n + t],
        k_cache,
        v_cache,
        page_table[rows:],
        module._fused_cfg,
        (None, None),
        out[o : o + t].view(tails, 8, -1),
        push and push[:4] + (n,),
    )
    if not push and o + t < out.shape[0]:
        out[o + t :].zero_()
    return True


@eager_on_graph
def _eager_prefill_attention(module, qkv, positions, out, push=None) -> None:
    from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch

    forward_batch = get_forward_batch()
    a = _prefill_args(module, forward_batch)
    # no fallback: the captured graph reads out; FA3 extend metadata always has these
    if a is None:
        raise RuntimeError(
            "fused prefill attention: batch not covered inside a prefill graph"
        )
    if forward_batch.mixed_verify_rows and _run_mixed_verify(
        module, qkv, positions, out, a, forward_batch, push
    ):
        return
    _run_pfa(module, qkv, positions, out, a, push)


def _capturing() -> bool:
    return (
        get_breakable_capture_recorder() is not None
        or torch.cuda.is_current_stream_capturing()
    )


def run_prefill_attention(
    module,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    forward_batch,
    push=None,
    last_only=False,
) -> Optional[torch.Tensor]:
    """With ``push`` (sp_o_proj_push_plan) a non-None result means the rows went to their owners."""
    from sglang.srt.model_executor.forward_mode import ForwardMode

    # a full prefill graph would freeze the host lengths pfa sizes its launch from
    if (
        not module._fused_attn
        or forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.MIXED)
        or is_in_full_prefill_graph()
    ):
        return None
    if qkv.dim() != 2 or not qkv.is_contiguous():
        return None
    a = _prefill_args(module, forward_batch)
    if a is None:
        return None
    tails, (ext, seq) = forward_batch.mixed_verify_rows or 0, a[6:]
    out = qkv.new_empty(
        (len(ext) + 7 * tails if last_only else qkv.shape[0], module.q_size)
    )
    if _capturing():
        _eager_prefill_attention(module, qkv, positions, out, push)
        return out
    if tails and _run_mixed_verify(
        module, qkv, positions, out, a, forward_batch, push, last_only
    ):
        return out
    window, n = module._fused_cfg["window"], sum(ext)
    kv = [min(k, window + e) if window else k for e, k in zip(ext, seq)]
    if (
        not 0 < n <= qkv.shape[0]
        or sum(e * k for e, k in zip(ext, kv)) < PREFILL_MIN_MEAN_EXTEND * sum(kv)
        or last_only
        and tails
    ):
        return None
    _run_pfa(module, qkv, positions, out, a, push, last_only)
    return out
