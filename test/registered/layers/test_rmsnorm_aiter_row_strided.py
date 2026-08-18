"""Guard for the aiter RMSNorm row-strided fast path.

RMSNorm.forward_aiter (ROCm/aiter) skips the `.contiguous()` clone for a 2D view
whose last dim is unit-stride but whose row pitch is strided -- e.g. a q_lora / kv_a
slice split from a packed QKV-a projection on the DeepSeek MLA prefill path. This
test asserts the no-clone path is bit-identical to feeding a contiguous copy, i.e.
aiter's row-wise rmsnorm2d_fwd consumes the row-strided view correctly.
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_amd_ci

# ROCm/aiter-only guard; register for the AMD 1-GPU small suite.
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")

import pytest
import torch

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.utils import is_hip

pytestmark = pytest.mark.skipif(
    not is_hip(), reason="aiter RMSNorm path is ROCm/HIP-only"
)


def _rmsnorm(hidden: int, dtype: torch.dtype) -> RMSNorm:
    norm = RMSNorm(hidden, eps=1e-6).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.02)  # non-trivial weights
    return norm


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden", [512, 1536])  # kv_a (512), q_lora (1536)
def test_forward_aiter_row_strided_matches_contiguous(dtype, hidden):
    torch.manual_seed(0)
    M = 128
    norm = _rmsnorm(hidden, dtype)
    fm = getattr(norm, "_forward_method", None)
    if fm is None or getattr(fm, "__func__", None) is not RMSNorm.forward_aiter:
        pytest.skip("aiter RMSNorm path not selected (SGLANG_USE_AITER off)")

    # Row-strided view with UNIT last-dim stride: take the first `hidden` columns
    # of a wider [M, 2*hidden] buffer -> stride (2*hidden, 1): contiguous rows,
    # strided row pitch. Mirrors a slice split from a packed QKV-a projection.
    wide = torch.randn(M, 2 * hidden, dtype=dtype, device="cuda")
    strided = wide[:, :hidden]
    assert not strided.is_contiguous()
    assert strided.stride(-1) == 1

    contig = strided.contiguous()
    assert torch.equal(strided, contig)  # identical values, different layout

    out_strided = norm.forward_aiter(strided)  # P1: fed to kernel without clone
    out_contig = norm.forward_aiter(contig)

    # Bit-identical: the row-strided fast path must match the contiguous path.
    assert torch.equal(out_strided, out_contig), (
        f"row-strided aiter RMSNorm output differs from contiguous "
        f"(dtype={dtype}, hidden={hidden}, "
        f"max_abs_diff={(out_strided.float() - out_contig.float()).abs().max().item()})"
    )

    # Sanity: it is actually an RMSNorm (matches the reference within dtype tol).
    ref = norm.forward_native(contig)
    torch.testing.assert_close(out_contig.float(), ref.float(), rtol=2e-2, atol=2e-2)
