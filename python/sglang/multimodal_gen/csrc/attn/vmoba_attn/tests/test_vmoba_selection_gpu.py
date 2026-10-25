# SPDX-License-Identifier: Apache-2.0
"""CUDA parity checks for VMoBA chunk selection."""

import importlib.util
import pathlib

import pytest
import torch

_VMOBA_PATH = pathlib.Path(__file__).resolve().parents[1] / "vmoba" / "vmoba.py"
_spec = importlib.util.spec_from_file_location("vmoba_under_test", _VMOBA_PATH)
vmoba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vmoba)


def _layout(seqlens, chunk_size):
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), dtype=torch.int32)
    cu_chunk, _, n, chunk_to_batch = vmoba.calc_chunks(cu, chunk_size)
    return cu, cu_chunk, int(n), chunk_to_batch


def _reference_topk_attention(q, k, v, seqlens, chunk_size, topk):
    """Plain torch reference for selected-chunk attention (no MoBA kernels)."""
    scale = q.shape[-1] ** -0.5
    out = torch.empty_like(q, dtype=torch.float32)
    batch_start = 0
    for seqlen in seqlens:
        batch_end = batch_start + seqlen
        chunk_bounds = [
            (start, min(start + chunk_size, batch_end))
            for start in range(batch_start, batch_end, chunk_size)
        ]
        chunk_means = torch.stack(
            [k[start:end].float().mean(dim=0) for start, end in chunk_bounds]
        )
        # Match moba_attn_varlen's gate and self-chunk amplification.
        gate = torch.einsum(
            "chd,lhd->chl", chunk_means, q[batch_start:batch_end].float()
        )
        for chunk_idx, (start, end) in enumerate(chunk_bounds):
            local_start, local_end = start - batch_start, end - batch_start
            gate[chunk_idx, :, local_start:local_end] += 1e9
        selected = torch.zeros_like(gate, dtype=torch.bool)
        selected.scatter_(
            0,
            torch.topk(gate, k=min(topk, len(chunk_bounds)), dim=0).indices,
            True,
        )

        for local_q in range(seqlen):
            for head in range(q.shape[1]):
                selected_chunks = torch.where(selected[:, head, local_q])[0].tolist()
                key_indices = torch.cat(
                    [
                        torch.arange(*chunk_bounds[idx], device=q.device)
                        for idx in selected_chunks
                    ]
                )
                scores = (
                    k[key_indices, head].float()
                    @ q[batch_start + local_q, head].float()
                ) * scale
                weights = scores.softmax(dim=0)
                out[batch_start + local_q, head] = (
                    weights @ v[key_indices, head].float()
                )
        batch_start = batch_end
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_gpu_mixed_sparse_selection_matches_torch_reference():
    pytest.importorskip("flash_attn")
    torch.manual_seed(0)
    seqlens, chunk_size, heads, head_dim = [128, 128], 32, 2, 32
    q, k, v = (
        torch.randn(256, heads, head_dim, device="cuda", dtype=torch.float16)
        for _ in range(3)
    )
    cu_seqlens = torch.tensor([0, 128, 256], dtype=torch.int32, device="cuda")

    actual = vmoba.moba_attn_varlen(
        q,
        k,
        v,
        cu_seqlens,
        max(seqlens),
        chunk_size,
        2,
        select_mode="topk",
        threshold_type="query_head",
    )
    expected = _reference_topk_attention(q, k, v, seqlens, chunk_size, topk=2)

    assert torch.allclose(actual.float(), expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_gpu_self_chunk_only_fallback_matches_flash_attention():
    pytest.importorskip("flash_attn")
    torch.manual_seed(1)
    seqlens, chunk_size, heads, head_dim = [128, 128], 32, 2, 32
    q, k, v = (
        torch.randn(256, heads, head_dim, device="cuda", dtype=torch.float16)
        for _ in range(3)
    )
    cu_seqlens = torch.tensor([0, 128, 256], dtype=torch.int32, device="cuda")
    _, cu_chunks, _, _ = _layout(seqlens, chunk_size)
    cu_chunks = cu_chunks.to(device="cuda")

    actual = vmoba.moba_attn_varlen(
        q,
        k,
        v,
        cu_seqlens,
        max(seqlens),
        chunk_size,
        1,
        select_mode="topk",
        threshold_type="query_head",
    )
    expected = vmoba.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_chunks,
        cu_chunks,
        chunk_size,
        chunk_size,
        causal=False,
    )

    assert torch.allclose(actual, expected, atol=5e-3, rtol=5e-3)
