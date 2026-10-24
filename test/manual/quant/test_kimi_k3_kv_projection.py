"""Run on SM103 to validate the Kimi K3 KV projection runner."""

from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.quantization import fp8_utils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="Requires SM103",
)


@pytest.mark.parametrize("m", [1, 1536, 6655, 6656, 9729, 16384])
def test_kv_projection_dispatch_and_accuracy(m, monkeypatch):
    torch.manual_seed(42)
    x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(3072, 512, device="cuda").to(torch.float8_e4m3fn)
    # Nonuniform scales exercise the transpose required by CUTLASS.
    scales = torch.rand(24, 4, device="cuda") * 0.1 + 0.01
    run = fp8_utils.flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
    with patch.object(
        fp8_utils, "_get_flashinfer_groupwise_backend", return_value="trtllm"
    ):
        monkeypatch.setenv("SGLANG_K3_KV_B_PROJ_CUTLASS", "0")
        baseline = run(x, w, [128, 128], scales)
        monkeypatch.setenv("SGLANG_K3_KV_B_PROJ_CUTLASS", "1")
        with patch.object(
            fp8_utils, "gemm_fp8_nt_groupwise", wraps=fp8_utils.gemm_fp8_nt_groupwise
        ) as gemm:
            actual = run(x, w, [128, 128], scales)
        expected = "cutlass" if m in (1536, 6656, 16384) else "trtllm"
        assert gemm.call_args.kwargs["backend"] == expected
    assert torch.isfinite(actual).all()
    relative_error = (
        actual.float() - baseline.float()
    ).norm() / baseline.float().norm()
    assert relative_error < 0.01
    if expected == "trtllm":
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
