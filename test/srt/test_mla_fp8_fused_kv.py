# -*- coding: utf-8 -*-
"""
PyTest: correctness of fused KV write vs baseline (non-fused).
Tests the mla_rope_quantize_fp8_fused kernel from sgl_kernel extension.
"""
import torch
import pytest

# Try to import fusion kernel (standalone or from sgl_kernel)
_has_sgl_kernel = False
mla_rope_quantize_fp8_fused = None
try:
    from mla_fusion_kernel import mla_rope_quantize_fp8_fused
    _has_sgl_kernel = True
except ImportError:
    try:
        from sgl_kernel import mla_rope_quantize_fp8_fused
        _has_sgl_kernel = True
    except ImportError:
        pass

requires_ext = pytest.mark.skipif(not _has_sgl_kernel, reason="sgl_kernel extension not available")

@requires_ext
@pytest.mark.parametrize("nnz", [256, 1024])
@pytest.mark.parametrize("Dn,Dr", [(512, 64)])
def test_fused_matches_baseline(nnz, Dn, Dr):
    device = "cuda"
    torch.manual_seed(0)

    # Inputs (half); in a real path you may use bfloat16. We pick half for demo.
    q_nope = torch.randn(nnz, Dn, device=device, dtype=torch.float16)
    q_rope = torch.randn(nnz, Dr, device=device, dtype=torch.float16)
    k_nope = torch.randn(nnz, Dn, device=device, dtype=torch.float16)
    k_rope = torch.randn(nnz, Dr, device=device, dtype=torch.float16)

    max_seq = max(2048, nnz)
    # cos/sin cache: [max_seq, 2*Dr]
    # Simple deterministic cache for test
    t = torch.linspace(0, 1, steps=max_seq, device=device, dtype=torch.float32)[:, None]
    idx = torch.arange(Dr, device=device, dtype=torch.float32)[None, :]
    # frequencies: small values to avoid overflow
    freqs = 0.1 * (idx + 1.0)
    cos = torch.cos(t * freqs)
    sin = torch.sin(t * freqs)
    cos_sin = torch.cat([cos, sin], dim=1)  # [max_seq, 2*Dr]

    pos_ids = torch.randint(low=0, high=max_seq, size=(nnz,), device=device, dtype=torch.long)

    # Baseline: produce k_nope_out/k_rope_out and emulate set_mla_kv_buffer (bytes concat)
    q_out_base = torch.empty(nnz, Dn + Dr, device=device, dtype=torch.uint8)
    k_nope_out = torch.empty(nnz, Dn, device=device, dtype=torch.uint8)
    k_rope_out = torch.empty(nnz, Dr, device=device, dtype=torch.uint8)

    mla_rope_quantize_fp8_fused(
        q_nope, q_rope, k_nope, k_rope, cos_sin, pos_ids, False,
        q_out_base, k_nope_out, k_rope_out, None, None
    )

    # emulate set_mla_kv_buffer_triton: concat bytes into KV buffer
    slots = nnz + 8
    kv_base = torch.zeros(slots, 1, Dn + Dr, device=device, dtype=torch.uint8)
    loc = torch.arange(nnz, device=device, dtype=torch.long)
    kv_base[loc, 0, :Dn] = k_nope_out
    kv_base[loc, 0, Dn:] = k_rope_out

    # Fused: direct KV write, skip K outputs
    q_out_fused = torch.empty_like(q_out_base)
    kv_fused = torch.zeros_like(kv_base)

    mla_rope_quantize_fp8_fused(
        q_nope, q_rope, k_nope, k_rope, cos_sin, pos_ids, False,
        q_out_fused, None, None, kv_fused, loc
    )

    # Assertions
    assert torch.equal(q_out_base, q_out_fused), "q_out must match exactly (bytewise)"
    assert torch.equal(kv_base, kv_fused), "KV fused write must match baseline concat"

