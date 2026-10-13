# SPDX-License-Identifier: Apache-2.0
"""VSA-H3 backend contracts.

The load-bearing check is sparsity -> 0: every tile is inside the budget, so
the block-sparse kernel must reproduce dense attention over the packed rows to
bf16 rounding. That single assertion pins the tile routing indices, ragged
tile masking, the untile permutation, and the softmax scale.
"""

from __future__ import annotations

import math

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn_h3 import (
    VSA_H3_TILE_ELEMS,
    VideoSparseAttentionH3Impl,
    VideoSparseAttentionH3MetadataBuilder,
    _topk_tile_lists,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="VSA-H3 kernels need CUDA"
)

# Ragged on purpose: text 70 and audio 100 are not tile multiples, and the
# video canvas (5, 6, 10) is ragged in every tile dimension.
PREFIX_SEGMENTS = (70, 0, 100)
VIDEO_SHAPE = (5, 6, 10)
HEADS = 4
HEAD_DIM = 128


def _build_metadata(sparsity: float, device, **kwargs):
    return VideoSparseAttentionH3MetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=VIDEO_SHAPE,
        patch_size=(1, 1, 1),
        VSA_sparsity=sparsity,
        prefix_segments=PREFIX_SEGMENTS,
        device=device,
        **kwargs,
    )


def _packed_qkv(device, seed: int = 7):
    used = sum(PREFIX_SEGMENTS) + math.prod(VIDEO_SHAPE)
    total = (used + 63) // 64 * 64
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = [
        torch.randn(
            (total, HEADS, HEAD_DIM), generator=generator, dtype=torch.float32
        ).to(device=device, dtype=torch.bfloat16)
        for _ in range(3)
    ]
    return used, total, tensors


def _dense_reference(q, k, v, used: int) -> torch.Tensor:
    qf = q[:used].float().permute(1, 0, 2)
    kf = k[:used].float().permute(1, 0, 2)
    vf = v[:used].float().permute(1, 0, 2)
    scores = qf @ kf.transpose(-2, -1) / math.sqrt(HEAD_DIM)
    return (torch.softmax(scores, dim=-1) @ vf).permute(1, 0, 2)


def _impl():
    impl = VideoSparseAttentionH3Impl(
        num_heads=HEADS,
        head_size=HEAD_DIM,
        causal=False,
        softmax_scale=HEAD_DIM**-0.5,
        num_kv_heads=HEADS,
        prefix="blocks.3.attn",
    )
    assert impl.layer_idx == 3
    return impl


def _run(impl, meta, used, total, q, k, v, gate=None):
    cu = torch.tensor([0, used, total], dtype=torch.int32, device=q.device)
    return impl.forward_varlen(
        q,
        k,
        v,
        cu_seqlens=cu,
        max_seqlen=used,
        cu_seqlens_host=(0, used, total),
        attn_metadata=meta,
        gate_compress=gate,
    )


@requires_cuda
def test_zero_sparsity_matches_dense() -> None:
    device = torch.device("cuda")
    meta = _build_metadata(0.0, device)
    used, total, (q, k, v) = _packed_qkv(device)
    assert meta.total_seq_length == used

    out = _run(_impl(), meta, used, total, q, k, v)
    reference = _dense_reference(q, k, v, used)
    diff = (out[:used].float() - reference).abs().max().item()
    assert diff < 2e-2, f"sparse(0) vs dense max diff {diff}"
    assert torch.all(out[used:] == 0)


@requires_cuda
def test_zero_gate_is_noop_and_trained_gate_activates() -> None:
    device = torch.device("cuda")
    meta = _build_metadata(0.5, device)
    used, total, (q, k, v) = _packed_qkv(device)
    impl = _impl()

    base = _run(impl, meta, used, total, q, k, v).clone()
    zero_gate = torch.zeros_like(q)
    gated_zero = _run(impl, meta, used, total, q, k, v, gate=zero_gate)
    assert torch.equal(base, gated_zero)

    gate = torch.randn_like(q) * 0.1
    gated = _run(impl, meta, used, total, q, k, v, gate=gate)
    assert not torch.equal(base, gated)


@requires_cuda
def test_dense_overrides() -> None:
    device = torch.device("cuda")
    meta = _build_metadata(0.9, device, dense_layers=(3,))
    used, total, (q, k, v) = _packed_qkv(device)
    out = _run(_impl(), meta, used, total, q, k, v)
    reference = _dense_reference(q, k, v, used)
    diff = (out[:used].float() - reference).abs().max().item()
    assert diff < 2e-2, f"dense-layer opt-out vs dense max diff {diff}"

    assert _build_metadata(0.9, device, dense_first_n_steps=0).VSA_sparsity == 0.9
    assert _build_metadata(0.9, device, dense_first_n_steps=1).VSA_sparsity == 0.0


def _lists_to_mask(lists: torch.Tensor, n_tiles: int) -> torch.Tensor:
    mask = torch.zeros(*lists.shape[:-1], n_tiles, dtype=torch.bool)
    mask.scatter_(-1, lists.long(), True)
    return mask


def test_topk_tile_list_semantics() -> None:
    num_prefix, num_video = 3, 10
    n_tiles = num_prefix + num_video
    scores = torch.randn(2, n_tiles, n_tiles)
    keep = math.ceil(0.5 * num_video)

    exempt = _topk_tile_lists(scores, num_prefix, num_video, 0.5, True)
    assert exempt.shape == (2, num_video, num_prefix + keep)
    assert exempt.dtype == torch.int32
    assert torch.equal(exempt, exempt.sort(dim=-1).values)
    mask = _lists_to_mask(exempt, n_tiles)
    assert mask[..., :num_prefix].all()
    assert (mask[..., num_prefix:].sum(dim=-1) == keep).all()

    compete = _topk_tile_lists(scores, num_prefix, num_video, 0.5, False)
    assert compete.shape == (2, num_video, min(keep + num_prefix, n_tiles))
    assert (_lists_to_mask(compete, n_tiles).sum(dim=-1) == keep + num_prefix).all()


def _masked_dense_reference(meta, used, q, k, v, gate, sparsity):
    """fp32 reference over the padded tile layout with the top-k tile mask."""
    n_tiles = meta.num_tiles
    seq_pad = n_tiles * VSA_H3_TILE_ELEMS
    valid = meta.pack_index >= 0
    src = meta.pack_index.clamp(min=0).long()

    def tile(x):
        t = x[src].float().permute(1, 0, 2)  # [H, S_pad, D]
        return t * valid.to(t.dtype)[None, :, None]

    qt, kt, vt, gt = tile(q), tile(k), tile(v), tile(gate)
    sizes = meta.variable_block_sizes.float()
    pooled = [
        t.view(HEADS, n_tiles, VSA_H3_TILE_ELEMS, HEAD_DIM).sum(2)
        / sizes[None, :, None]
        for t in (qt, kt, vt)
    ]
    scores = pooled[0] @ pooled[1].transpose(-1, -2) / math.sqrt(HEAD_DIM)
    lists = _topk_tile_lists(
        scores, meta.num_prefix_tiles, meta.num_video_tiles, sparsity, meta.exempt
    )
    tile_mask = torch.ones(HEADS, n_tiles, n_tiles, dtype=torch.bool, device=q.device)
    tile_mask[:, meta.num_prefix_tiles :] = _lists_to_mask(lists.cpu(), n_tiles).to(
        q.device
    )
    row_mask = tile_mask.repeat_interleave(VSA_H3_TILE_ELEMS, 1).repeat_interleave(
        VSA_H3_TILE_ELEMS, 2
    )
    row_mask &= valid[None, None, :]
    logits = qt @ kt.transpose(-1, -2) / math.sqrt(HEAD_DIM)
    logits = logits.masked_fill(~row_mask, float("-inf"))
    out = torch.softmax(logits, dim=-1) @ vt
    compress = torch.softmax(scores, dim=-1) @ pooled[2]  # [H, n_tiles, D]
    out = out + compress.repeat_interleave(VSA_H3_TILE_ELEMS, 1) * gt
    return out[:, meta.unpack_index.long()].permute(1, 0, 2)  # [used, H, D]


@requires_cuda
def test_sparse_gated_matches_masked_dense_reference() -> None:
    device = torch.device("cuda")
    for exempt in (True, False):
        meta = _build_metadata(0.5, device, exempt=exempt)
        used, total, (q, k, v) = _packed_qkv(device)
        gate = (torch.randn_like(q) * 0.1).to(torch.bfloat16)
        out = _run(_impl(), meta, used, total, q, k, v, gate=gate)
        reference = _masked_dense_reference(meta, used, q, k, v, gate, 0.5)
        diff = (out[:used].float() - reference).abs().max().item()
        assert diff < 2e-2, f"exempt={exempt}: sparse+gate vs reference {diff}"
        assert torch.all(out[used:] == 0)


def test_metadata_tile_geometry_accounts_every_row() -> None:
    device = torch.device("cpu")
    meta = _build_metadata(0.9, device)
    used = sum(PREFIX_SEGMENTS) + math.prod(VIDEO_SHAPE)
    assert int(meta.variable_block_sizes.sum()) == used
    assert meta.unpack_index.numel() == used
    assert int((meta.pack_index >= 0).sum()) == used
    assert meta.pack_index.numel() == meta.num_tiles * VSA_H3_TILE_ELEMS
    # Prefix chunks never straddle segment boundaries: 70 -> 64+6, 100 -> 64+36.
    assert meta.num_prefix_tiles == 4
    assert meta.variable_block_sizes[: meta.num_prefix_tiles].tolist() == [
        64,
        6,
        64,
        36,
    ]
    video_tiles = (
        math.ceil(VIDEO_SHAPE[0] / 4)
        * math.ceil(VIDEO_SHAPE[1] / 4)
        * math.ceil(VIDEO_SHAPE[2] / 4)
    )
    assert meta.num_video_tiles == video_tiles
    assert meta.variable_block_sizes.numel() == meta.num_prefix_tiles + video_tiles
    assert VSA_H3_TILE_ELEMS == 64
