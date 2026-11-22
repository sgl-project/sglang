"""Bit-exactness of ``_math_sdpa`` against the aten math SDPA backend.

``_math_sdpa`` re-implements the math backend only to drop the
``_safe_softmax`` all-``-inf``-row guard that aten pays on every unmasked
attention call (it is never needed here: ViT sequences are never padded,
so no query row has an all-``-inf`` row). The guarded and unguarded forms
share one expression, so every input must map to an identical result.
"""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    attention as h3_attention,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, suite="nightly-kernel-test-1-gpu-amd", nightly=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def aten_math_reference(query, key, value):
    with sdpa_kernel(SDPBackend.MATH):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_math_sdpa_matches_aten_math_backend(dtype):
    # ViT decoder shape at 2048^2: single frame, 512 tokens, 16 heads x 64.
    shape = (1, 16, 512, 64)
    torch.manual_seed(20261122)
    query = torch.randn(shape, device="cuda", dtype=dtype)
    key = torch.randn(shape, device="cuda", dtype=dtype)
    value = torch.randn(shape, device="cuda", dtype=dtype)
    out = h3_attention._math_sdpa(query, key, value)
    ref = aten_math_reference(query, key, value)
    assert torch.equal(out, ref)


def test_rocm_math_gate_dispatch_hits_math_fast_path(monkeypatch):
    # Force the fast path on whatever device runs this (CI is gfx942, where
    # the gate is off): both symbols are read at call time, and _math_sdpa
    # uses portable torch ops only. Nulling the flash symbol simulates
    # gfx1151, where aiter's CK mha_fwd rejects the architecture.
    monkeypatch.setattr(h3_attention, "_FORCE_ROCM_MATH_SDPA", True)
    monkeypatch.setattr(h3_attention, "_rocm_flash_attn", None)
    # _sdpa_attention takes [B, S, H, D].
    shape = (1, 512, 16, 64)
    torch.manual_seed(20261122)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    out = h3_attention._sdpa_attention(query, key, value)
    ref = h3_attention._math_sdpa(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    ).transpose(1, 2)
    assert torch.equal(out, ref)
    # And the replica is bit-for-bit the aten math backend.
    ref_aten = aten_math_reference(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    ).transpose(1, 2)
    assert torch.equal(out, ref_aten)
