import pytest
import torch

from sglang.srt.layers.telechat4_mhc_torch import (
    telechat4_mhc_post_torch,
    telechat4_mhc_pre_torch,
)
from sglang.srt.layers.telechat4_mhc_triton import (
    MHC_FLAT_SIZE,
    MHC_HIDDEN_SIZE,
    MHC_OUTPUT_SIZE,
    MHC_STREAMS,
    telechat4_mhc_post,
    telechat4_mhc_pre,
)
from sglang.srt.runtime_context import get_forward


def _torch_pre(residual, fn, hc_scale, hc_base, sinkhorn_repeat=20):
    residual_fp32 = residual.float()
    residual_flat = residual_fp32.reshape(-1, MHC_FLAT_SIZE)
    inv_rms = torch.rsqrt(residual_flat.square().mean(dim=-1, keepdim=True) + 1e-6)
    mixes = torch.nn.functional.linear(residual_flat, fn.float()) * inv_rms
    pre_logits, post_logits, comb_logits = torch.split(mixes, [4, 4, 16], dim=-1)
    pre_mix = torch.sigmoid(pre_logits * hc_scale[0] + hc_base[:4]) + 1e-6
    post_mix = torch.sigmoid(post_logits * hc_scale[1] + hc_base[4:8]) * 2.0
    comb_mix = (comb_logits * hc_scale[2] + hc_base[8:]).reshape(-1, 4, 4)
    comb_mix = torch.exp(comb_mix - comb_mix.amax(dim=-1, keepdim=True))
    comb_mix = comb_mix / comb_mix.sum(dim=-1, keepdim=True) + 1e-6
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + 1e-6)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + 1e-6)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + 1e-6)
    layer_input = (pre_mix.unsqueeze(-1) * residual_fp32).sum(dim=1)
    return post_mix.unsqueeze(-1), comb_mix, layer_input.to(torch.bfloat16)


def _torch_post(x, residual, post_mix, comb_mix):
    output = post_mix.squeeze(-1).unsqueeze(-1) * x.float().unsqueeze(1)
    output += (comb_mix.unsqueeze(-1) * residual.float().unsqueeze(2)).sum(dim=1)
    return output.to(torch.bfloat16)


@pytest.mark.parametrize("implementation", ["split", "split_direct"])
@pytest.mark.parametrize("num_tokens", [1, 16, 64, 128, 256, 512, 1024])
def test_telechat4_mhc_triton_matches_torch(num_tokens, implementation):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")

    torch.manual_seed(0)
    residual = (
        torch.randn(
            num_tokens,
            MHC_STREAMS,
            MHC_HIDDEN_SIZE,
            device="npu",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    fn = (
        torch.randn(
            MHC_OUTPUT_SIZE,
            MHC_FLAT_SIZE,
            device="npu",
            dtype=torch.bfloat16,
        )
        * 0.01
    ).contiguous()
    hc_scale = torch.tensor([0.5, 0.25, 0.25], device="npu", dtype=torch.float32)
    hc_base = torch.randn(MHC_OUTPUT_SIZE, device="npu", dtype=torch.float32) * 0.01

    expected_pre = _torch_pre(residual, fn, hc_scale, hc_base)
    actual_pre = telechat4_mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        implementation=implementation,
    )
    fallback_pre = telechat4_mhc_pre_torch(
        residual, fn.float(), hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20
    )
    torch.testing.assert_close(actual_pre[0], expected_pre[0], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual_pre[1], expected_pre[1], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual_pre[2], expected_pre[2], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(fallback_pre[0], expected_pre[0], atol=0, rtol=0)
    torch.testing.assert_close(fallback_pre[1], expected_pre[1], atol=0, rtol=0)
    torch.testing.assert_close(fallback_pre[2], expected_pre[2], atol=0, rtol=0)

    x = (
        torch.randn(
            num_tokens,
            MHC_HIDDEN_SIZE,
            device="npu",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    post_mix, comb_mix, _ = actual_pre
    comb_mix_for_post = comb_mix.transpose(-1, -2).contiguous()
    if implementation == "split":
        assert comb_mix.is_contiguous()
        assert comb_mix_for_post.data_ptr() != comb_mix.data_ptr()
    else:
        assert not comb_mix.is_contiguous()
        assert comb_mix_for_post.data_ptr() == comb_mix.data_ptr()
    expected_post = _torch_post(x, residual, post_mix, comb_mix_for_post)
    actual_post = telechat4_mhc_post(x, residual, post_mix, comb_mix_for_post)
    fallback_comb_mix = fallback_pre[1].transpose(-1, -2).contiguous()
    fallback_expected_post = _torch_post(
        x, residual, fallback_pre[0], fallback_comb_mix
    )
    fallback_post = telechat4_mhc_post_torch(
        x, residual, fallback_pre[0], fallback_comb_mix
    )
    torch.testing.assert_close(actual_post, expected_post, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(fallback_post, fallback_expected_post, atol=0, rtol=0)


def test_telechat4_mhc_triton_rejects_other_hidden_sizes():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")

    residual = torch.empty(1, 4, 4096, device="npu", dtype=torch.bfloat16)
    fn = torch.empty(24, 16384, device="npu", dtype=torch.bfloat16)
    hc_scale = torch.empty(3, device="npu", dtype=torch.float32)
    hc_base = torch.empty(24, device="npu", dtype=torch.float32)
    with pytest.raises(ValueError, match="3584"):
        telechat4_mhc_pre(residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20)


def test_telechat4_mhc_auto_layout_only_for_eager_extend():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")

    residual = torch.zeros(1, 4, 3584, device="npu", dtype=torch.bfloat16)
    fn = torch.zeros(24, 14336, device="npu", dtype=torch.bfloat16)
    hc_scale = torch.ones(3, device="npu", dtype=torch.float32)
    hc_base = torch.zeros(24, device="npu", dtype=torch.float32)

    with get_forward().scoped(is_extend_in_batch=False):
        graph_comb = telechat4_mhc_pre(
            residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20
        )[1]
    with get_forward().scoped(is_extend_in_batch=True):
        eager_extend_comb = telechat4_mhc_pre(
            residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20
        )[1]

    assert graph_comb.is_contiguous()
    assert not eager_extend_comb.is_contiguous()
    assert (
        eager_extend_comb.transpose(-1, -2).contiguous().data_ptr()
        == eager_extend_comb.data_ptr()
    )


def test_telechat4_mhc_ascendc_dispatch_matches_torch(monkeypatch, request):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")

    try:
        import sgl_kernel_npu  # noqa: F401
    except ImportError:
        pytest.skip("sgl-kernel-npu is required")
    if not hasattr(torch.ops.npu, "hc_pre") or not hasattr(torch.ops.npu, "hc_post"):
        pytest.skip("sgl-kernel-npu was built without mHC operators")

    from sglang.srt.models import telechat4 as telechat4_model

    monkeypatch.setattr(telechat4_model, "_NPU_MHC_BACKEND", "ascendc")
    telechat4_model._get_npu_mhc_backend.cache_clear()
    request.addfinalizer(telechat4_model._get_npu_mhc_backend.cache_clear)
    assert telechat4_model._get_npu_mhc_backend() == "ascendc"

    torch.manual_seed(20260807)
    num_tokens = 8
    residual = torch.randn(
        num_tokens,
        MHC_STREAMS,
        MHC_HIDDEN_SIZE,
        device="npu",
        dtype=torch.bfloat16,
    )
    fn = (
        torch.randn(
            MHC_OUTPUT_SIZE,
            MHC_FLAT_SIZE,
            device="npu",
            dtype=torch.float32,
        )
        / MHC_FLAT_SIZE**0.5
    ).contiguous()
    hc_scale = torch.tensor([0.8, 1.1, 0.7], device="npu")
    hc_base = torch.randn(MHC_OUTPUT_SIZE, device="npu") * 0.1

    actual_pre = telechat4_model.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        1,
        8,
    )
    expected_pre = telechat4_mhc_pre_torch(
        residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20
    )
    for actual, expected in zip(actual_pre, expected_pre):
        torch.testing.assert_close(actual, expected, atol=0.03125, rtol=5e-3)

    x = torch.randn(
        num_tokens,
        MHC_HIDDEN_SIZE,
        device="npu",
        dtype=torch.bfloat16,
    )
    comb_for_post = actual_pre[1].transpose(-1, -2).contiguous()
    actual_post = telechat4_model.mhc_post(x, residual, actual_pre[0], comb_for_post)
    expected_post = telechat4_mhc_post_torch(x, residual, actual_pre[0], comb_for_post)
    torch.testing.assert_close(actual_post, expected_post, atol=0.03125, rtol=5e-3)
