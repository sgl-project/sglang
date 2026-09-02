from contextlib import nullcontext

import pytest
import torch

import sglang.kernels.ops.layernorm.mhc as mhc
from sglang.kernels.ops.layernorm.mhc import mhc_fused_post_pre, mhc_post, mhc_pre
from sglang.srt.model_executor.runner_utils.capture_owner import (
    collect_full_cuda_graph_owners,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")


@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("num_tokens", [0, 1, 8, 17, 32, 64])
@pytest.mark.parametrize("use_norm", [False, True])
def test_mhc_fused_post_pre_matches_unfused(
    monkeypatch, hidden_size, num_tokens, use_norm
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TileLang mHC kernels")

    # This SM120 regression exercises the same TileLang prenorm fallback used
    # by the GLM serving profile. Server-argument normalization disables the
    # optional DeepGEMM path before model loading; the standalone kernel test
    # must establish that profile explicitly.
    monkeypatch.setenv("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "0")
    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_round_robin_split", lambda: False)
    # This is a single-process kernel unit test with no TP group initialized.
    # mhc_pre / mhc_fused_post_pre allocate the MoE input in the symmetric-memory
    # pool via use_symmetric_memory(get_tp_group(), ...); bypass that path so the
    # kernel runs with a plain torch.empty allocation. Mirrors the workaround in
    # test_mxfp4_sm90_cutlass.py for the same TP-group-not-initialized case.
    monkeypatch.setattr(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(mhc, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(mhc, "get_tp_group", lambda: None)
    torch.manual_seed(0)
    device = torch.device("cuda")
    hc_mult = 4
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    residual = (
        torch.randn(
            num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    post_prev = torch.rand(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_prev = (
        torch.rand(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
        * 0.25
    )
    fn = (
        torch.randn(hc_mult3, hc_hidden_size, device=device, dtype=torch.float32) * 0.01
    )
    hc_scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
    hc_base = torch.zeros(hc_mult3, device=device, dtype=torch.float32)
    norm_weight = (
        torch.ones(hidden_size, device=device, dtype=torch.bfloat16)
        if use_norm
        else None
    )
    norm_eps = 1e-6 if use_norm else None

    rms_eps = 1e-6
    hc_eps = 1e-6
    sinkhorn_repeat = 2

    residual_ref = post_ref = comb_ref = layer_ref = None
    if num_tokens > 0:
        residual_ref = mhc_post(x, residual, post_prev, comb_prev)
        post_ref, comb_ref, layer_ref = mhc_pre(
            residual_ref,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            hc_eps,
            2.0,
            sinkhorn_repeat,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
    with collect_full_cuda_graph_owners() as capture_owners:
        residual_out, post_out, comb_out, layer_out = mhc_fused_post_pre(
            x,
            residual,
            post_prev,
            comb_prev,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            hc_eps,
            2.0,
            sinkhorn_repeat,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )

    torch.cuda.synchronize()
    if num_tokens == 0:
        assert capture_owners == []
        assert residual_out.shape == residual.shape
        assert post_out.shape == (0, hc_mult, 1)
        assert comb_out.shape == (0, hc_mult, hc_mult)
        assert layer_out.shape == (0, hidden_size)
        assert residual_out.dtype == torch.bfloat16
        assert post_out.dtype == torch.float32
        assert comb_out.dtype == torch.float32
        assert layer_out.dtype == torch.bfloat16
        return

    assert len(capture_owners) == 2
    gemm_out_mul_owner, gemm_out_sqrsum_owner = capture_owners
    assert gemm_out_mul_owner.dtype == torch.float32
    assert gemm_out_sqrsum_owner.dtype == torch.float32
    if num_tokens <= 32:
        # Small-batch path: the split-K pair is the kernel operand set.
        assert gemm_out_mul_owner.ndim == 3
        assert gemm_out_mul_owner.shape[1:] == (num_tokens, hc_mult3)
        assert gemm_out_sqrsum_owner.ndim == 2
        assert gemm_out_sqrsum_owner.shape == gemm_out_mul_owner.shape[:2]
    else:
        # Large-batch TileLang fallback (DeepGEMM prenorm disabled): the
        # reallocated 2-D/1-D pair is the real kernel operand set and must
        # be the retained owners, not the discarded split-K allocations.
        assert gemm_out_mul_owner.ndim == 2
        assert gemm_out_mul_owner.shape == (num_tokens, hc_mult3)
        assert gemm_out_sqrsum_owner.ndim == 1
        assert gemm_out_sqrsum_owner.shape == (num_tokens,)

    assert residual_ref is not None
    assert post_ref is not None
    assert comb_ref is not None
    assert layer_ref is not None
    assert residual_out.shape == residual_ref.shape
    assert post_out.shape == post_ref.shape
    assert comb_out.shape == comb_ref.shape
    assert layer_out.shape == layer_ref.shape

    torch.testing.assert_close(residual_out, residual_ref, atol=0, rtol=0)
    torch.testing.assert_close(post_out, post_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(comb_out, comb_ref, atol=1e-3, rtol=1e-3)
    layer_atol = 2e-2 if use_norm else 2e-3
    layer_rtol = 2e-2 if use_norm else 2e-3
    torch.testing.assert_close(layer_out, layer_ref, atol=layer_atol, rtol=layer_rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_norm_weight_conversion_copy_is_retained(monkeypatch):
    monkeypatch.setenv("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "0")
    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_round_robin_split", lambda: False)
    monkeypatch.setattr(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(mhc, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(mhc, "get_tp_group", lambda: None)
    torch.manual_seed(20260830)
    device = torch.device("cuda")
    hc_mult, hidden_size, num_tokens = 4, 4096, 8
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    residual = (
        torch.randn(
            num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    post_prev = torch.rand(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_prev = (
        torch.rand(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
        * 0.25
    )
    fn = (
        torch.randn(hc_mult3, hc_hidden_size, device=device, dtype=torch.float32) * 0.01
    )
    hc_scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
    hc_base = torch.zeros(hc_mult3, device=device, dtype=torch.float32)
    # A float32 norm weight forces the bf16 conversion copy inside the kernel.
    norm_weight = torch.ones(hidden_size, device=device, dtype=torch.float32)

    with collect_full_cuda_graph_owners() as capture_owners:
        mhc_fused_post_pre(
            x,
            residual,
            post_prev,
            comb_prev,
            fn,
            hc_scale,
            hc_base,
            1e-6,
            1e-4,
            1e-4,
            2.0,
            2,
            norm_weight=norm_weight,
            norm_eps=1e-6,
        )
    torch.cuda.synchronize()

    assert len(capture_owners) == 3
    converted = capture_owners[-1]
    assert converted.dtype == torch.bfloat16
    assert converted.shape == (hidden_size,)
    assert converted is not norm_weight


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
