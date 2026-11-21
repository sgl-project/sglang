# SPDX-License-Identifier: Apache-2.0
"""Regression tests for VMoBA chunk layout and chunk selection.

The CPU tests check which (chunk, head, query) entries get selected with the
flash-attn kernels replaced; the CUDA tests run flash-attn 2 itself.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.csrc.attn.vmoba_attn.vmoba import vmoba


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


def _fake_varlen_attn(
    q, k, v, cu_q, cu_k, max_q, max_k, causal=False, softmax_scale=None
):
    """Plain torch reference for flash_attn_varlen_func (external kernel)."""
    out = torch.zeros_like(q)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    for i in range(len(cu_q) - 1):
        a, b = int(cu_q[i]), int(cu_q[i + 1])
        s = torch.einsum("qhd,khd->hqk", q[a:b], k[a:b]) * scale
        out[a:b] = torch.einsum("hqk,khd->qhd", s.softmax(-1), v[a:b])
    return out


def _self_chunk_reference(q, k, v, seqlens, chunk_size, softmax_scale=None):
    _, cu_chunk, _, _ = _layout(seqlens, chunk_size)
    return _fake_varlen_attn(
        q, k, v, cu_chunk, cu_chunk, 0, 0, softmax_scale=softmax_scale
    )


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


def test_softmax_scale_is_respected(fake_flash):
    torch.manual_seed(0)
    seqlens, chunk, scale = [128, 128], 32, 0.2
    q, k, v = (torch.randn(256, 2, 16) for _ in range(3))
    cu = torch.tensor([0, 128, 256], dtype=torch.int32)
    out = vmoba.moba_attn_varlen(
        q,
        k,
        v,
        cu,
        128,
        chunk,
        1,
        select_mode="topk",
        threshold_type="query_head",
        softmax_scale=scale,
    )
    ref = _self_chunk_reference(q, k, v, seqlens, chunk, softmax_scale=scale)
    assert torch.allclose(out, ref, atol=1e-5)
    default_ref = _self_chunk_reference(q, k, v, seqlens, chunk)
    assert not torch.allclose(out, default_ref, atol=1e-5)


def test_topk_head_global_stays_inside_each_sequence(monkeypatch):
    torch.manual_seed(0)
    chunk, heads = 64, 2
    q, k, v = (torch.randn(256, heads, 16) for _ in range(3))
    cu = torch.tensor([0, 128, 256], dtype=torch.int32)
    captured = {}

    def capture(*args):
        captured["moba_q_sh_indices"] = args[10]
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
    pytest.importorskip("flash_attn.flash_attn_interface")
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
    pytest.importorskip("flash_attn.flash_attn_interface")
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


def test_flash_attn_grid_uses_actual_segment_lengths(monkeypatch):
    torch.manual_seed(0)
    chunk, heads = 32, 2
    q, k, v = (torch.randn(256, heads, 16) for _ in range(3))
    cu = torch.tensor([0, 128, 256], dtype=torch.int32)
    calls = []

    def fake_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, **kwargs):
        calls.append((cu_seqlens_q, cu_seqlens_k, kwargs))
        lse = torch.zeros(q.shape[1], q.shape[0])
        return torch.zeros_like(q), lse, None, None

    monkeypatch.setattr(vmoba, "_flash_attn_varlen_forward", fake_forward)
    vmoba.moba_attn_varlen(
        q, k, v, cu, 128, chunk, 2, select_mode="topk", threshold_type="query_head"
    )
    (self_cu, _, self_kw), (moba_cu_q, moba_cu_k, moba_kw) = calls
    assert self_kw["max_seqlen_q"] == self_kw["max_seqlen_k"] == chunk
    assert int(self_cu.diff().max()) <= chunk
    # the MoBA branch grid follows the busiest chunk, not the full sequence
    assert moba_kw["max_seqlen_q"] == int(moba_cu_q.diff().max())
    assert moba_kw["max_seqlen_q"] < 128
    assert moba_kw["max_seqlen_k"] == chunk
    assert int(moba_cu_k.diff().max()) <= chunk


MOBA_CONFIG = {
    "temporal_chunk_size": 2,
    "temporal_topk": 3,
    "spatial_chunk_size": [3, 4],
    "spatial_topk": 20,
    "st_chunk_size": [4, 6, 4],
    "st_topk": 15,
    "moba_select_mode": "threshold",
    "moba_threshold": 0.25,
    "moba_threshold_type": "query_head",
    "first_full_layer": 2,
    "first_full_step": 12,
    "temporal_layer": 1,
    "spatial_layer": 1,
    "st_layer": 1,
}


@pytest.fixture
def vmoba_backend(monkeypatch):
    from sglang.multimodal_gen.runtime.layers.attention.backends import vmoba as backend

    calls = []

    def fake_moba(q, k, v, **kwargs):
        calls.append(("moba", kwargs))
        return torch.zeros_like(q)

    def fake_full(*args, **kwargs):
        calls.append(("full", kwargs))
        return _fake_varlen_attn(*args, **kwargs)

    monkeypatch.setattr(backend, "moba_attn_varlen", fake_moba)
    monkeypatch.setattr(backend, "flash_attn_varlen_func", fake_full)
    return backend, calls


@pytest.mark.parametrize(
    "layer, step, expected",
    [
        (0, 20, "full"),
        (1, 20, "full"),
        (3, 0, "full"),
        (3, 11, "full"),
        (3, 12, "moba"),
    ],
)
def test_backend_full_attention_for_leading_layers_and_steps(
    vmoba_backend, layer, step, expected
):
    backend, calls = vmoba_backend
    impl = backend.VMOBAAttentionImpl(
        num_heads=2, head_size=16, softmax_scale=0.2, prefix=f"blocks.{layer}.attn1"
    )
    metadata = backend.VideoMobaAttentionMetadataBuilder().build(
        current_timestep=step,
        raw_latent_shape=(4, 12, 16),
        patch_size=(1, 2, 2),
        **MOBA_CONFIG,
    )
    seq = 4 * 6 * 8
    q, k, v = (torch.randn(1, seq, 2, 16) for _ in range(3))
    out = impl.forward(q, k, v, metadata)

    assert [c[0] for c in calls] == [expected]
    assert calls[0][1]["softmax_scale"] == 0.2
    assert out.shape == q.shape
    if expected == "full":
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * 0.2
        ref = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v)
        assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.parametrize("nested", [False, True])
def test_denoising_stage_builds_vmoba_metadata(nested):
    import addict

    from sglang.multimodal_gen.runtime.layers.attention.backends.vmoba import (
        VideoMobaAttentionMetadata,
        VMOBAAttentionBackend,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingStage,
    )

    config = {"moba_config": MOBA_CONFIG} if nested else MOBA_CONFIG
    stage = SimpleNamespace(attn_backend=VMOBAAttentionBackend)
    server_args = SimpleNamespace(
        attention_backend_config=addict.Dict(config),
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(patch_size=(1, 2, 2))
        ),
    )
    batch = SimpleNamespace(raw_latent_shape=torch.Size([1, 16, 21, 60, 104]))

    metadata = DenoisingStage._build_attn_metadata(stage, 5, batch, server_args)

    assert isinstance(metadata, VideoMobaAttentionMetadata)
    assert metadata.current_timestep == 5
    assert metadata.patch_resolution == [21, 30, 52]
    assert metadata.first_full_layer == 2
    assert metadata.spatial_chunk_size == [3, 4]


def test_denoising_stage_requires_vmoba_config():
    import addict

    from sglang.multimodal_gen.runtime.layers.attention.backends.vmoba import (
        VMOBAAttentionBackend,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingStage,
    )

    stage = SimpleNamespace(attn_backend=VMOBAAttentionBackend)
    server_args = SimpleNamespace(
        attention_backend_config=addict.Dict(),
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(patch_size=(1, 2, 2))
        ),
    )
    batch = SimpleNamespace(raw_latent_shape=torch.Size([1, 16, 21, 60, 104]))
    with pytest.raises(ValueError, match="--attention-backend-config"):
        DenoisingStage._build_attn_metadata(stage, 0, batch, server_args)
