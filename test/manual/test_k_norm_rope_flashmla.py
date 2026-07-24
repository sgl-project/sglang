"""Tests for DeepSeek-V4 fused norm + RoPE + flashmla cache layout kernels."""

import pytest
import torch

from sglang.jit_kernel.dsv4.elementwise import fused_k_norm_rope_flashmla
from sglang.jit_kernel.dsv4.fused_k_norm_rope_flashmla_torch import (
    fused_k_norm_rope_flashmla_torch,
)
from sglang.srt.utils import get_device


@pytest.mark.parametrize("max_pos", [16, 128])
def test_fused_k_norm_rope_flashmla_correctness(max_pos):
    """Test Q norm + rope against reference."""
    torch.manual_seed(42)
    rope_dim = 64
    page_size = 256
    head_dim = 512
    eps = 1e-6

    kv = torch.randn(max_pos, head_dim, dtype=torch.bfloat16, device=get_device())
    kv_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=get_device())
    freqs_cis = torch.randn(
        max_pos, rope_dim // 2, dtype=torch.complex64, device=get_device()
    )
    positions = torch.randint(
        0, max_pos, (max_pos,), dtype=torch.int64, device=get_device()
    )
    """
    out_loc = torch.randint(
        0, max_pos, (max_pos,), dtype=torch.int32, device=get_device()
    )
    """
    out_loc = torch.randperm(max_pos, device=get_device(), dtype=torch.int32)
    kvcache = torch.zeros((128, 149760), dtype=torch.uint8, device=get_device())

    fused_k_norm_rope_flashmla(
        kv, kv_weight, eps, freqs_cis, positions, out_loc, kvcache, page_size
    )

    # Reference
    ref_kvcache = torch.zeros_like(kvcache)
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    fused_k_norm_rope_flashmla_torch(
        kv, kv_weight, freqs_real, positions, out_loc, ref_kvcache, eps, page_size
    )

    torch.testing.assert_close(
        kvcache.float(), ref_kvcache.float(), rtol=1e-2, atol=1e-2
    )
