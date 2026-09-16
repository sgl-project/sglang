from contextlib import nullcontext

import pytest
import torch

import sglang.kernels.ops.layernorm.mhc as mhc
from sglang.kernels.ops.layernorm.mhc import mhc_fused_post_pre, mhc_post, mhc_pre
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-large")
register_cuda_ci(est_time=45, stage="nightly", runner_config="4-gpu-gb300")


@pytest.mark.parametrize(
    "sm,hc_mult,hidden_size,num_tokens,expected",
    [
        (103, 4, 4096, 1, True),
        (103, 4, 4096, 512, True),
        (103, 4, 4096, 0, False),
        (103, 4, 4096, 513, False),
        (103, 4, 7168, 6, False),
        (103, 2, 4096, 6, False),
        (103, 8, 4096, 6, False),
        (100, 4, 4096, 6, False),
        (90, 4, 4096, 6, False),
    ],
)
def test_mhc_post_split_guard(
    monkeypatch, sm, hc_mult, hidden_size, num_tokens, expected
):
    monkeypatch.setattr(mhc, "get_device_sm", lambda: sm)
    assert mhc._use_split_mhc_post(num_tokens, hc_mult, hidden_size) == expected


@pytest.mark.parametrize(
    "num_tokens,hidden_size,split_hidden",
    [
        (6, 4096, True),
        (96, 4096, True),
        (480, 4096, True),
        (512, 4096, True),
        (513, 4096, False),
        (8000, 4096, False),
        (6, 7168, False),
    ],
)
def test_mhc_post_matches_default_kernel(
    monkeypatch, num_tokens, hidden_size, split_hidden
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TileLang mHC kernels")
    if split_hidden and mhc.get_device_sm() != 103:
        pytest.skip("The split mHC post path is selected only on SM103")

    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_interleave", lambda: False)
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(
        num_tokens, 4, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    post = torch.rand(num_tokens, 4, 1, device="cuda", dtype=torch.float32)
    comb = torch.rand(num_tokens, 4, 4, device="cuda", dtype=torch.float32)
    expected = torch.empty_like(residual)
    mhc.mhc_post_tilelang(
        comb,
        residual,
        post.squeeze(-1),
        x,
        expected,
        4,
        hidden_size,
        split_hidden=False,
    )
    if split_hidden:
        split_output = torch.empty_like(residual)
        mhc.mhc_post_tilelang(
            comb,
            residual,
            post.squeeze(-1),
            x,
            split_output,
            4,
            hidden_size,
            split_hidden=True,
        )
        torch.testing.assert_close(split_output, expected, atol=0, rtol=0)
    actual = mhc_post(x, residual, post, comb)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "hidden_size,num_tokens",
    [(h, n) for h in (4096, 7168) for n in (0, 1, 8, 17, 32, 64)]
    + [(4096, 512), (4096, 513)],
)
@pytest.mark.parametrize("use_norm", [False, True])
def test_mhc_fused_post_pre_matches_unfused(
    monkeypatch, hidden_size, num_tokens, use_norm
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TileLang mHC kernels")

    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_interleave", lambda: False)
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
        assert residual_out.shape == residual.shape
        assert post_out.shape == (0, hc_mult, 1)
        assert comb_out.shape == (0, hc_mult, hc_mult)
        assert layer_out.shape == (0, hidden_size)
        assert residual_out.dtype == torch.bfloat16
        assert post_out.dtype == torch.float32
        assert comb_out.dtype == torch.float32
        assert layer_out.dtype == torch.bfloat16
        return

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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
