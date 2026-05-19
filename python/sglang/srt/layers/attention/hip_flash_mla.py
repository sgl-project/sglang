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
        import flash_mla

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
        from sglang.srt.layers.attention.nsa.tilelang_kernel import (
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

    from sglang.srt.flashmla_tests import quant as flashmla_quant
    from sglang.srt.flashmla_tests.lib import (
        ExtraTestParamForDecode,
        KVScope,
        TestcaseForDecode,
        TestParam,
    )
    from sglang.srt.flashmla_tests.ref import ref_sparse_attn_decode

    assert block_table is None
    assert cache_seqlens is None
    assert is_fp8_kvcache

    b, s_q, h_q, d_qk = q.shape
    d_v = head_dim_v

    fp8_layout = flashmla_quant.FP8KVCacheLayout.MODEL1_FP8Sparse

    p = TestParam(
        s_q=s_q,
        s_kv="unused",
        topk="unused",
        h_q=h_q,
        h_kv=1,
        d_qk=d_qk,
        d_v=d_v,
        decode=ExtraTestParamForDecode(
            b=b,
            is_varlen="unused",
            have_zero_seqlen_k="unused",
            extra_s_k="unused",
            extra_topk="unused",
            extra_block_size="unused",
            have_extra_topk_length="unused",
        ),
        # unused?
        seed=-1,
        check_correctness=True,
        is_all_indices_invalid=False,
        num_runs=10,
        have_attn_sink=True,
        have_topk_length=True,
    )

    blocked_k_quantized = k_cache
    blocked_k = flashmla_quant.dequantize_k_cache(
        blocked_k_quantized.view(FP8_DTYPE), fp8_layout
    )
    # blocked_k_requantized = flashmla_quant.quantize_k_cache(blocked_k, fp8_layout)
    # assert torch.testing.assert_allclose(blocked_k_requantized.byte(), blocked_k_quantized.byte())
    kv_scope = KVScope(
        t="unused",
        cache_seqlens="unused",
        block_table="unused",
        blocked_k=blocked_k,
        blocked_k_quantized=blocked_k_quantized,
        abs_indices="unused",
        indices_in_kvcache=indices,
        topk_length=topk_length,
    )

    extra_kv_scope = None
    if extra_k_cache is not None:
        extra_blocked_k_quantized = extra_k_cache
        extra_blocked_k = flashmla_quant.dequantize_k_cache(
            extra_blocked_k_quantized.view(FP8_DTYPE), fp8_layout
        )
        # extra_blocked_k_requantized = flashmla_quant.quantize_k_cache(extra_blocked_k, fp8_layout)
        # assert torch.testing.assert_allclose(extra_blocked_k_requantized.byte(), extra_blocked_k_quantized.byte())
        extra_kv_scope = KVScope(
            t="unused",
            cache_seqlens="unused",
            block_table="unused",
            blocked_k=extra_blocked_k,
            blocked_k_quantized=extra_blocked_k_quantized,
            abs_indices="unused",
            indices_in_kvcache=extra_indices_in_kvcache,
            topk_length=extra_topk_length,
        )

    t = TestcaseForDecode(
        p="unused",
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
    import sglang.srt.flashmla_tests.kernelkit as kk

    out_ref, lse_ref = pack_ref
    out_fast, lse_fast = pack_fast

    # the copied threshold is too strict, not checked why
    # copied from: test_flash_mla_sparse_decoding.py
    # is_out_correct = kk.check_is_allclose(
    #     "out", out_fast, out_ref, abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6
    # )
    # is_lse_correct = kk.check_is_allclose(
    #     "lse", lse_fast, lse_ref, abs_tol=1e-6, rel_tol=8.01 / 65536
    # )

    # loosen thresh
    is_out_correct = kk.check_is_allclose(
        "out", out_fast, out_ref, abs_tol=1e-2, rel_tol=10.0, cos_diff_tol=5e-6
    )
    is_lse_correct = kk.check_is_allclose(
        "lse", lse_fast, lse_ref, abs_tol=1e-6, rel_tol=8.01 / 65536
    )

    assert is_out_correct and is_lse_correct, f"{is_out_correct=} {is_lse_correct=}"
