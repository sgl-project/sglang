# SPDX-License-Identifier: Apache-2.0
"""VSA-H3 Ref2VA regions and 128-token logical tiles.

The oracle below is written at logical-tile granularity from FastVideo's
"p2_multi_region" mask semantics, independent of the kernel-block split the
backend serves 128-token tiles with.
"""

from __future__ import annotations

import math

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn_h3 import (
    VideoSparseAttentionH3Impl,
    VideoSparseAttentionH3MetadataBuilder,
    _compute_topk,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    _ref2va_vsa_segments,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="VSA-H3 kernels need CUDA"
)

HEADS = 4
HEAD_DIM = 128
TARGET = (5, 6, 10)
# text, image ref, reference video (ragged in every tile dimension), its
# audio, a standalone audio ref, target audio, generated video
SEGMENTS = (70, 30, 24, (3, 5, 9), 40, 50, TARGET)


def _build(sparsity: float, device, *, tile_size=128, ref_keep_rate=0.25, **kw):
    return VideoSparseAttentionH3MetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=TARGET,
        patch_size=(1, 1, 1),
        VSA_sparsity=sparsity,
        prefix_segments=(),
        device=device,
        packed_segments=SEGMENTS,
        tile_size=tile_size,
        ref_keep_rate=ref_keep_rate,
        **kw,
    )


def _used() -> int:
    return sum(math.prod(s) if isinstance(s, tuple) else s for s in SEGMENTS)


def test_region_geometry_accounts_every_row() -> None:
    meta = _build(0.9, torch.device("cpu"))
    used = _used()
    assert meta.total_seq_length == used
    assert meta.blocks_per_tile == 2
    assert int(meta.variable_block_sizes.sum()) == used
    assert int((meta.pack_index >= 0).sum()) == used
    assert sorted(meta.unpack_index.tolist()) == sorted(
        (meta.pack_index >= 0).nonzero().flatten().tolist()
    )
    # Dense rows chunk per segment into 128-token tiles: 70, 30, 24, 40, 50.
    assert meta.num_prefix_tiles == 5
    logical = meta.variable_block_sizes.view(-1, 2).sum(-1)
    assert logical[:5].tolist() == [70, 30, 24, 40, 50]
    # Reference (3, 5, 9) and target (5, 6, 10) at (4, 4, 8) tiles.
    reference_tiles = 1 * 2 * 2
    target_tiles = 2 * 2 * 2
    assert meta.video_tile_spans == (
        (5, 5 + reference_tiles),
        (5 + reference_tiles, 5 + reference_tiles + target_tiles),
    )
    assert meta.num_video_tiles == reference_tiles + target_tiles
    assert meta.span_sparsities == (0.75, 0.9)
    # Each kernel block holds at most 64 rows, the lower half first.
    halves = meta.variable_block_sizes.view(-1, 2)
    assert torch.all(halves[:, 0] == logical.clamp(max=64))


def test_reference_regions_need_keep_rate_and_exempt() -> None:
    with pytest.raises(ValueError, match="vsa_ref_keep_rate"):
        _build(0.9, torch.device("cpu"), ref_keep_rate=None)
    with pytest.raises(ValueError, match="compete"):
        _build(0.9, torch.device("cpu"), exempt=False)


def test_ref2va_segments_follow_packed_layout() -> None:
    blocks = [
        {"kind": "image", "latent_h": 8, "latent_w": 12},
        {
            "kind": "video_audio",
            "ref_audio_t": 6,
            "latent_t": 3,
            "latent_h": 8,
            "latent_w": 12,
        },
        {"kind": "audio", "ref_audio_t": 5},
    ]
    latent = (4, 10, 16)
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=33,
        latent_t=latent[0],
        latent_h=latent[1],
        latent_w=latent[2],
        audio_t=20,
        ref_blocks=blocks,
        include_video_pos=True,
    )
    segments = _ref2va_vsa_segments(packed, latent, (1, 2, 2))
    assert segments == (33, 0, 24, 12, (3, 4, 6), 10, 40, (4, 5, 8))
    # Region rows are exactly the layout's sparse-eligible video rows.
    cursor, region_rows = 0, []
    for segment in segments:
        rows = math.prod(segment) if isinstance(segment, tuple) else segment
        if isinstance(segment, tuple):
            region_rows.append(torch.arange(cursor, cursor + rows))
        cursor += rows
    assert cursor == int(packed["cu_seqlens"][1])
    assert torch.equal(torch.cat(region_rows), packed["video_pos"])


def _oracle(meta, used, q, k, v, gate):
    """Masked dense attention over logical tiles, plus the gated compression."""
    split = meta.blocks_per_tile
    tile = meta.tile_elems
    logical = meta.variable_block_sizes.view(-1, split).sum(-1).float()
    n = logical.numel()
    valid = meta.pack_index >= 0
    src = meta.pack_index.clamp(min=0).long()

    def tiled(x):
        t = x[src].float().permute(1, 0, 2)
        return t * valid.to(t.dtype)[None, :, None]

    qt, kt, vt, gt = tiled(q), tiled(k), tiled(v), tiled(gate)
    pooled = [
        t.view(HEADS, n, tile, HEAD_DIM).sum(2) / logical[None, :, None]
        for t in (qt, kt, vt)
    ]
    scores = pooled[0] @ pooled[1].transpose(-1, -2) / math.sqrt(HEAD_DIM)
    prefix = meta.num_prefix_tiles
    mask = torch.zeros(HEADS, n, n, dtype=torch.bool, device=q.device)
    for (start, end), sparsity in zip(meta.video_tile_spans, meta.span_sparsities):
        keep = _compute_topk(sparsity, end - start)
        picked = scores[..., start:end].topk(keep, dim=-1).indices + start
        mask.scatter_(-1, picked, True)
    mask[..., :prefix] = True
    mask[:, :prefix, :] = True
    row_mask = mask.repeat_interleave(tile, 1).repeat_interleave(tile, 2)
    row_mask &= valid[None, None, :]
    logits = qt @ kt.transpose(-1, -2) / math.sqrt(HEAD_DIM)
    out = torch.softmax(logits.masked_fill(~row_mask, float("-inf")), -1) @ vt
    compress = torch.softmax(scores, dim=-1) @ pooled[2]
    out = out + compress.repeat_interleave(tile, 1) * gt
    return out[:, meta.unpack_index.long()].permute(1, 0, 2)


def _impl():
    return VideoSparseAttentionH3Impl(
        num_heads=HEADS,
        head_size=HEAD_DIM,
        causal=False,
        softmax_scale=HEAD_DIM**-0.5,
        num_kv_heads=HEADS,
        prefix="blocks.3.attn",
    )


def _qkvg(device, used, seed=5):
    total = (used + 63) // 64 * 64
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, k, v, gate = (
        torch.randn((total, HEADS, HEAD_DIM), generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        for _ in range(4)
    )
    return total, q, k, v, gate * 0.1


@requires_cuda
@pytest.mark.parametrize("tile_size", [64, 128])
def test_regions_match_masked_dense_oracle(tile_size: int) -> None:
    device = torch.device("cuda")
    meta = _build(0.5, device, tile_size=tile_size)
    used = _used()
    total, q, k, v, gate = _qkvg(device, used)
    cu = torch.tensor([0, used, total], dtype=torch.int32, device=device)
    out = _impl().forward_varlen(
        q,
        k,
        v,
        cu_seqlens=cu,
        max_seqlen=used,
        cu_seqlens_host=(0, used, total),
        attn_metadata=meta,
        gate_compress=gate,
    )
    reference = _oracle(meta, used, q, k, v, gate)
    diff = (out[:used].float() - reference).abs().max().item()
    assert diff < 2e-2, f"tile {tile_size}: regions vs oracle max diff {diff}"
    assert torch.all(out[used:] == 0)

    # The deferred (Ulysses) gate fold sees the same per-block compression.
    from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn_h3 import (
        vsa_h3_fold_gate,
    )

    unfolded, out_compress = _impl().forward_varlen(
        q,
        k,
        v,
        cu_seqlens=cu,
        max_seqlen=used,
        cu_seqlens_host=(0, used, total),
        attn_metadata=meta,
        return_compress=True,
    )
    vsa_h3_fold_gate(unfolded, gate, out_compress, meta, 0)
    assert torch.equal(unfolded, out)
