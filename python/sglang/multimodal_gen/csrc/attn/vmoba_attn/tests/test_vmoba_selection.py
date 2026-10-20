# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for VMoBA chunk layout and chunk selection.

These tests check which (chunk, head, query) entries get selected, so they do
not need flash-attn or a GPU. Only the flash-attn kernels are replaced.
"""

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


def _gate_inputs(seqlens, chunk_size, heads=4, seed=0):
    """Build (gate, valid, self_mask) the same way moba_attn_varlen does."""
    torch.manual_seed(seed)
    cu, cu_chunk, n, chunk_to_batch = _layout(seqlens, chunk_size)
    seqlen = int(cu[-1])
    gate = torch.randn(n, heads, seqlen)
    pos = torch.arange(seqlen).unsqueeze(0)
    start, end = cu_chunk[:n].unsqueeze(1), cu_chunk[1 : n + 1].unsqueeze(1)
    self_mask = ((pos >= start) & (pos < end)).unsqueeze(1).expand(-1, heads, -1)
    b_start = cu[chunk_to_batch.long()].unsqueeze(1)
    b_end = cu[chunk_to_batch.long() + 1].unsqueeze(1)
    outside = ((pos < b_start) | (pos >= b_end)).unsqueeze(1)
    gate = gate.masked_fill(outside, -float("inf"))
    return gate, gate != -float("inf"), self_mask.contiguous()


THRESHOLD_FNS = [
    "_select_threshold_query_head",
    "_select_threshold_block",
    "_select_threshold_overall",
    "_select_threshold_head_global",
]


@pytest.mark.parametrize("fn_name", THRESHOLD_FNS)
def test_threshold_zero_selects_only_self_chunk(fn_name):
    gate, valid, self_mask = _gate_inputs([96, 96], 32)
    out = getattr(vmoba, fn_name)(gate, valid, self_mask, 0.0)
    assert torch.equal(out, valid & self_mask)


@pytest.mark.parametrize("fn_name", THRESHOLD_FNS)
def test_selection_grows_with_threshold(fn_name):
    gate, valid, self_mask = _gate_inputs([16 * 8], 8)
    fn = getattr(vmoba, fn_name)
    counts = [int(fn(gate, valid, self_mask, t).sum()) for t in (0.0, 0.5, 0.99, 1.0)]
    # a threshold that cannot be reached must select at least as much as 0.99
    # (it used to fall back to a cutoff of 1 and select far less)
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


@pytest.mark.parametrize("seqlens", [[64], [40, 64], [64, 64]])
def test_threshold_overall_when_every_sequence_fits_one_chunk(seqlens):
    gate, valid, self_mask = _gate_inputs(seqlens, 64)
    out = vmoba._select_threshold_overall(gate, valid, self_mask, 0.25)
    assert torch.equal(out, valid & self_mask)


def test_calc_chunks_rejects_empty_sequence():
    cu = torch.tensor([0, 100, 100, 228], dtype=torch.int32)
    with pytest.raises(ValueError):
        vmoba.calc_chunks(cu, 64)


def test_calc_chunks_layout():
    _, cu_chunk, n, chunk_to_batch = _layout([100, 128], 64)
    assert n == 4
    assert cu_chunk.tolist() == [0, 64, 100, 164, 228]
    assert chunk_to_batch.tolist() == [0, 0, 1, 1]


def _fake_varlen_attn(q, k, v, cu_q, cu_k, max_q, max_k, causal=False):
    """Plain torch reference for flash_attn_varlen_func (external kernel)."""
    out = torch.zeros_like(q)
    scale = q.shape[-1] ** -0.5
    for i in range(len(cu_q) - 1):
        a, b = int(cu_q[i]), int(cu_q[i + 1])
        s = torch.einsum("qhd,khd->hqk", q[a:b], k[a:b]) * scale
        out[a:b] = torch.einsum("hqk,khd->qhd", s.softmax(-1), v[a:b])
    return out


def _self_chunk_reference(q, k, v, seqlens, chunk_size):
    _, cu_chunk, _, _ = _layout(seqlens, chunk_size)
    return _fake_varlen_attn(q, k, v, cu_chunk, cu_chunk, 0, 0)


@pytest.fixture
def fake_flash(monkeypatch):
    monkeypatch.setattr(vmoba, "flash_attn_varlen_func", _fake_varlen_attn)


def test_topk_one_means_self_chunk_only(fake_flash):
    torch.manual_seed(0)
    seqlens, chunk = [128, 128], 32
    q, k, v = (torch.randn(256, 2, 16) for _ in range(3))
    cu = torch.tensor([0, 128, 256], dtype=torch.int32)
    out = vmoba.moba_attn_varlen(
        q, k, v, cu, 128, chunk, 1, select_mode="topk", threshold_type="query_head"
    )
    ref = _self_chunk_reference(q, k, v, seqlens, chunk)
    assert torch.allclose(out, ref, atol=1e-5)


def test_topk_head_global_stays_inside_each_sequence(monkeypatch):
    torch.manual_seed(0)
    chunk, heads = 64, 2
    q, k, v = (torch.randn(256, heads, 16) for _ in range(3))
    cu = torch.tensor([0, 128, 256], dtype=torch.int32)
    captured = {}

    def capture(*args):
        captured["moba_q_sh_indices"] = args[-1]
        captured["cu_q"] = args[6]
        return torch.zeros_like(q)

    monkeypatch.setattr(vmoba.MixedAttention, "apply", staticmethod(capture))
    vmoba.moba_attn_varlen(
        q, k, v, cu, 128, chunk, 3, select_mode="topk", threshold_type="head_global"
    )
    # selected (query, head) pairs: each query has exactly one non-self chunk
    # available in its own sequence, so at most 1 per (query, head)
    assert captured["moba_q_sh_indices"].numel() <= 256 * heads


def test_empty_sequence_is_dropped(fake_flash):
    torch.manual_seed(0)
    q, k, v = (torch.randn(228, 2, 16) for _ in range(3))
    cu = torch.tensor([0, 100, 100, 228], dtype=torch.int32)
    out = vmoba.moba_attn_varlen(
        q, k, v, cu, 128, 64, 1, select_mode="topk", threshold_type="query_head"
    )
    ref = _self_chunk_reference(q, k, v, [100, 128], 64)
    assert torch.allclose(out, ref, atol=1e-5)
