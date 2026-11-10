import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="TileLang mixed-dtype Gemma RMSNorm requires an NVIDIA V100",
)


def test_mixed_dtype_gemma_rmsnorm_matches_fp32_residual_reference(monkeypatch):
    pytest.importorskip("tilelang")
    from sglang.srt.layers.tilelang_gemma_rmsnorm_sm70 import (
        can_use_gemma_fused_add_rmsnorm_sm70,
        gemma_fused_add_rmsnorm_sm70,
    )

    monkeypatch.setenv("SGLANG_V100_GEMMA_RMSNORM", "1")
    torch.manual_seed(23)
    rows, hidden_size = 256, 5120
    x_initial = torch.randn(rows, hidden_size, device="cuda", dtype=torch.float16)
    residual_initial = torch.randn(
        rows, hidden_size, device="cuda", dtype=torch.float32
    )
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32) * 0.1
    x = x_initial.clone()
    residual = residual_initial.clone()

    assert can_use_gemma_fused_add_rmsnorm_sm70(x, residual, weight, None)
    actual_x, actual_residual = gemma_fused_add_rmsnorm_sm70(x, residual, weight, 1e-6)
    expected_residual = residual_initial + x_initial.float()
    expected_x = (
        expected_residual
        * torch.rsqrt(expected_residual.square().mean(dim=-1, keepdim=True) + 1e-6)
        * (weight + 1)
    ).half()

    assert actual_x.data_ptr() == x.data_ptr()
    assert actual_residual.data_ptr() == residual.data_ptr()
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual_x, expected_x, rtol=1e-3, atol=1e-3)


def test_mixed_dtype_gemma_rmsnorm_gate_is_strict(monkeypatch):
    pytest.importorskip("tilelang")
    from sglang.srt.layers.tilelang_gemma_rmsnorm_sm70 import (
        can_use_gemma_fused_add_rmsnorm_sm70,
    )

    x = torch.empty(256, 5120, device="cuda", dtype=torch.float16)
    residual = torch.empty_like(x, dtype=torch.float32)
    weight = torch.empty(5120, device="cuda", dtype=torch.float32)

    monkeypatch.setenv("SGLANG_V100_GEMMA_RMSNORM", "0")
    assert not can_use_gemma_fused_add_rmsnorm_sm70(x, residual, weight, None)
    monkeypatch.setenv("SGLANG_V100_GEMMA_RMSNORM", "invalid")
    with pytest.raises(ValueError, match="must be a boolean value"):
        can_use_gemma_fused_add_rmsnorm_sm70(x, residual, weight, None)
