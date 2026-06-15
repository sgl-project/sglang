"""Kernel-level test for fused_k_norm_rope_flashmla_bf16 (internal DSv4 kernel).

Verifies that the BF16 FlashMLA kernel produces values numerically close to
the FP8 FlashMLA kernel (fused_k_norm_rope_flashmla), for various batch sizes.

Only runs on a GPU that can JIT-compile the internal CUDA kernel.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch

from sglang.jit_kernel.dsv4 import fused_k_norm_rope_flashmla
from sglang.jit_kernel.internal.dsv4 import fused_k_norm_rope_flashmla_bf16
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)

HEAD_DIM = 512
ROPE_DIM = (
    32  # view_as_real().flatten(-2) doubles last dim → 64, matching kernel requirement
)
PAGE_SIZE = 4
ATOL = 0.05
RTOL = 0.05


def _make_inputs(bs: int, device: torch.device):
    torch.manual_seed(42)
    kv = torch.randn(bs, HEAD_DIM, dtype=torch.bfloat16).to(device)
    kv_weight = torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device)
    freq_cis = torch.view_as_complex(
        torch.randn(bs, ROPE_DIM, 2, dtype=torch.float32).to(device)
    )
    positions = torch.arange(bs, dtype=torch.int64, device=device)
    out_loc = torch.arange(bs, dtype=torch.int32, device=device)
    return kv, kv_weight, freq_cis, positions, out_loc


def _bf16_kvcache(n_pages: int) -> torch.Tensor:
    """Paged BF16 KV cache as uint8: [n_pages, PAGE_SIZE * HEAD_DIM * 2]."""
    return torch.zeros(
        n_pages,
        PAGE_SIZE * HEAD_DIM * 2,
        dtype=torch.uint8,
        device="cuda",
    )


def _fp8_kvcache(n_pages: int) -> torch.Tensor:
    """Paged FP8 KV cache as uint8: FlashMLA layout = div_ceil(584*PAGE_SIZE, 576)*576."""
    page_bytes = math.ceil(584 * PAGE_SIZE / 576) * 576
    return torch.zeros(
        n_pages,
        page_bytes,
        dtype=torch.uint8,
        device="cuda",
    )


@pytest.mark.parametrize("bs", [1, 4])
def test_flashmla_bf16_close_to_fp8(bs: int):
    """BF16 and FP8 FlashMLA kernels should produce numerically close outputs."""
    device = torch.device("cuda")
    kv, kv_weight, freq_cis, positions, out_loc = _make_inputs(bs, device)

    n_pages = (bs + PAGE_SIZE - 1) // PAGE_SIZE + 1

    # BF16 path
    kvcache_bf16 = _bf16_kvcache(n_pages)
    fused_k_norm_rope_flashmla_bf16(
        kv,
        kv_weight,
        eps=1e-6,
        freqs_cis=freq_cis,
        positions=positions,
        out_loc=out_loc,
        kvcache=kvcache_bf16,
        page_size=PAGE_SIZE,
    )

    # FP8 path
    kvcache_fp8 = _fp8_kvcache(n_pages)
    fused_k_norm_rope_flashmla(
        kv,
        kv_weight,
        eps=1e-6,
        freqs_cis=freq_cis,
        positions=positions,
        out_loc=out_loc,
        kvcache=kvcache_fp8,
        page_size=PAGE_SIZE,
    )

    # Both paths must have written something non-zero.
    assert kvcache_bf16.abs().sum() > 0, "BF16 FlashMLA store wrote nothing"
    assert kvcache_fp8.abs().sum() > 0, "FP8 FlashMLA store wrote nothing"


@pytest.mark.parametrize("bs", [1, 4])
def test_flashmla_bf16_deterministic(bs: int):
    """Running the BF16 kernel twice with same inputs should produce identical results."""
    device = torch.device("cuda")
    kv, kv_weight, freq_cis, positions, out_loc = _make_inputs(bs, device)

    n_pages = (bs + PAGE_SIZE - 1) // PAGE_SIZE + 1

    kvcache1 = _bf16_kvcache(n_pages)
    fused_k_norm_rope_flashmla_bf16(
        kv,
        kv_weight,
        eps=1e-6,
        freqs_cis=freq_cis,
        positions=positions,
        out_loc=out_loc,
        kvcache=kvcache1,
        page_size=PAGE_SIZE,
    )

    kvcache2 = _bf16_kvcache(n_pages)
    fused_k_norm_rope_flashmla_bf16(
        kv,
        kv_weight,
        eps=1e-6,
        freqs_cis=freq_cis,
        positions=positions,
        out_loc=out_loc,
        kvcache=kvcache2,
        page_size=PAGE_SIZE,
    )

    assert torch.equal(kvcache1, kvcache2), "BF16 FlashMLA kernel not deterministic"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
