"""Muse Spark RMSNorm + per-token NVFP4 quant: JIT kernel against the torch reference."""

import pytest
import torch

from sglang.kernels.ops.layernorm.muse_spark_rmsnorm_nvfp4_quant import (
    _FUSED_NORM_FP4_QUANT,
)
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci

# The SM100 pool has no single-GPU runner_config.
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="cvt.e2m1x2 needs SM100+",
)


def _run(backend, x, weight, quantize, want_normed):
    return _FUSED_NORM_FP4_QUANT(
        x,
        weight,
        1e-6,
        1.0 / 2688.0,
        quantize=quantize,
        want_normed=want_normed,
        backend=backend,
    )


@pytest.mark.parametrize("num_tokens", [1, 33, 512])
@pytest.mark.parametrize("hidden_size", [4096, 8192])
@pytest.mark.parametrize("has_weight", [True, False])
@pytest.mark.parametrize(
    "quantize, want_normed", [(True, False), (True, True), (False, True)]
)
def test_rmsnorm_nvfp4_quant_matches_native(
    num_tokens, hidden_size, has_weight, quantize, want_normed
):
    torch.manual_seed(num_tokens + hidden_size)
    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    x[0, :16] = 0  # an all-zero scale block
    weight = (
        1.0 + 0.1 * torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
        if has_weight
        else None
    )
    out = _run(KernelBackend.JIT, x, weight, quantize, want_normed)
    ref = _run(KernelBackend.TORCH, x, weight, quantize, want_normed)

    fp4, sf, scale, normed = out
    ref_fp4, ref_sf, ref_scale, ref_normed = ref
    if want_normed:
        torch.testing.assert_close(normed, ref_normed, rtol=1e-2, atol=1e-2)
    else:
        assert normed is None and ref_normed is None
    if not quantize:
        assert fp4 is None and ref_fp4 is None
        return
    torch.testing.assert_close(scale, ref_scale, rtol=1e-2, atol=0)
    # IEEE vs rcp.approx reciprocals: only values on a rounding boundary differ.
    assert (sf != ref_sf).float().mean().item() < 1e-3
    assert (fp4 != ref_fp4).float().mean().item() < 1e-3


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
