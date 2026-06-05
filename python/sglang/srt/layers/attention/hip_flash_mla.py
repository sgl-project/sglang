from typing import Any, Optional

import torch

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import is_hip

FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    if is_hip():
        import os

        backend = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "tilelang")
    else:
        import sgl_kernel.flash_mla as flash_mla

    if backend == "comparison":
        pack_ref, pack_fast_via_tester = flash_mla_with_kvcache_entrypoint(
            backend="torch", **kwargs
        )
        pack_fast_via_api = flash_mla_with_kvcache_entrypoint(
            backend="kernel", **kwargs
        )
        _assert_close(pack_ref=pack_fast_via_tester, pack_fast=pack_fast_via_api)
        _assert_close(pack_ref=pack_ref, pack_fast=pack_fast_via_tester)
        _assert_close(pack_ref=pack_ref, pack_fast=pack_fast_via_api)
        return pack_ref

    if backend == "torch":
        return flash_mla_with_kvcache_torch(**kwargs)

    if backend == "tilelang":
        from sglang.srt.layers.attention.dsa.tilelang_kernel import (
            dpsk_v4_fp8_attention_fwd,
        )

        return dpsk_v4_fp8_attention_fwd(**kwargs)

    if backend == "triton":
        from sglang.srt.layers.attention.nsa.triton_decode import (
            triton_fp8_attention_fwd,
        )

        return triton_fp8_attention_fwd(**kwargs)

    if backend == "kernel":
        return flash_mla.flash_mla_with_kvcache(**kwargs)

    raise NotImplementedError(f"unknown backend: {backend!r}")


def flash_mla_with_kvcache_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: Any,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
):

    from sglang.srt.layers.attention.flashmla_torch_fallback import (
        ExtraDecodeParams,
        FP8KVCacheLayout,
        KVScope,
        SparseDecodeInputs,
        SparseDecodeParams,
        dequantize_k_cache,
        ref_sparse_attn_decode,
    )

    assert block_table is None
    assert cache_seqlens is None
    assert is_fp8_kvcache

    b, s_q, h_q, d_qk = q.shape
    d_v = head_dim_v

    fp8_layout = FP8KVCacheLayout.MODEL1_FP8Sparse

    p = SparseDecodeParams(
        s_q=s_q,
        h_q=h_q,
        h_kv=1,
        d_qk=d_qk,
        d_v=d_v,
        decode=ExtraDecodeParams(
            b=b,
        ),
    )

    blocked_k_quantized = k_cache
    blocked_k = dequantize_k_cache(blocked_k_quantized.view(FP8_DTYPE), fp8_layout)
    kv_scope = KVScope(
        blocked_k=blocked_k,
        indices_in_kvcache=indices,
        topk_length=topk_length,
    )

    extra_kv_scope = None
    if extra_k_cache is not None:
        extra_blocked_k_quantized = extra_k_cache
        extra_blocked_k = dequantize_k_cache(
            extra_blocked_k_quantized.view(FP8_DTYPE), fp8_layout
        )
        extra_kv_scope = KVScope(
            blocked_k=extra_blocked_k,
            indices_in_kvcache=extra_indices_in_kvcache,
            topk_length=extra_topk_length,
        )

    t = SparseDecodeInputs(
        q=q,
        attn_sink=attn_sink,
        sm_scale=softmax_scale,
        kv_scope=kv_scope,
        extra_kv_scope=extra_kv_scope,
    )
    # print(f"hi {p=} {t=}")
    # print(
    #     f"hi info "
    #     f"{get_tensor_info(t.kv_scope.blocked_k)=} "
    #     f"{get_tensor_info(t.kv_scope.blocked_k_quantized)=} "
    #     f"{get_tensor_info(t.extra_kv_scope.blocked_k) if t.extra_kv_scope is not None else None=} "
    #     f"{get_tensor_info(t.extra_kv_scope.blocked_k_quantized) if t.extra_kv_scope is not None else None=} "
    # )

    pack_ref = ref_sparse_attn_decode(p, t)

    # tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
    # pack_fast_via_tester = flashmla_lib.run_flash_mla_decode(
    #     p, t, tile_scheduler_metadata, num_splits=None
    # )

    # return pack_ref, pack_fast_via_tester
    return pack_ref


def _assert_close(pack_ref, pack_fast):
    out_ref, lse_ref = pack_ref
    out_fast, lse_fast = pack_fast
    torch.testing.assert_close(out_fast, out_ref, atol=1e-2, rtol=10.0)
    torch.testing.assert_close(lse_fast, lse_ref, atol=1e-6, rtol=8.01 / 65536)
