# SPDX-License-Identifier: Apache-2.0
"""hybrid_window_attn_h3 (VDN-H3 window softmax) backend contracts.

The load-bearing check: on a ragged packed layout the backend must reproduce
a masked dense softmax with exactly the VDN mask (chunk-aligned window,
anchor frames dense as rows and columns, text/audio dense both ways, padding
outside everything) to bf16 rounding. radius >= F must reproduce dense
attention.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3_vdn import (
    VDNHybridAttentionArchConfig,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.hybrid_window_attn_h3 import (
    HybridWindowAttentionH3Impl,
    HybridWindowAttentionH3MetadataBuilder,
    window_mask_frames,
    window_mask_reference,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn import VDNH3Layout

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="hybrid_window_attn_h3 kernels need CUDA"
)

# ragged on purpose: 70 and 100 are not tile multiples, 12 frames is not a chunk multiple
TEXT_LEN = 70
AUDIO_ROWS = 100
NUM_FRAMES = 12
FRAME_H, FRAME_W = 6, 8
TOKENS_PER_FRAME = FRAME_H * FRAME_W
HEADS = 4
HEAD_DIM = 128


def _layout() -> VDNH3Layout:
    video_start = TEXT_LEN + AUDIO_ROWS
    used = video_start + NUM_FRAMES * TOKENS_PER_FRAME
    seq_len = (used + 63) // 64 * 64
    return VDNH3Layout(
        seq_len=seq_len,
        used=used,
        text_len=TEXT_LEN,
        video_start=video_start,
        num_frames=NUM_FRAMES,
        tokens_per_frame=TOKENS_PER_FRAME,
        frame_height=FRAME_H,
        frame_width=FRAME_W,
    )


def _hybrid(**overrides) -> VDNHybridAttentionArchConfig:
    kwargs = dict(chunk=5, radius=1, anchor_frames="both")
    kwargs.update(overrides)
    return VDNHybridAttentionArchConfig(**kwargs)


def _qkv(device, seed: int = 7):
    layout = _layout()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = [
        torch.randn(
            (layout.seq_len, HEADS, HEAD_DIM), generator=generator, dtype=torch.float32
        ).to(device=device, dtype=torch.bfloat16)
        for _ in range(3)
    ]
    return layout, tensors


def _masked_reference(q, k, v, mask: torch.Tensor, used: int) -> torch.Tensor:
    qf = q[:used].float().permute(1, 0, 2)
    kf = k[:used].float().permute(1, 0, 2)
    vf = v[:used].float().permute(1, 0, 2)
    scores = qf @ kf.transpose(-2, -1) / math.sqrt(HEAD_DIM)
    scores = scores.masked_fill(~mask[None], float("-inf"))
    return (torch.softmax(scores, dim=-1) @ vf).permute(1, 0, 2)


def _prepare_flash_attention() -> None:
    # the platform resolver runs this before the first forward; a direct impl must too
    from sglang.multimodal_gen.runtime.platforms import current_platform

    current_platform._prepare_flash_attention_for_blackwell()


def _impl() -> HybridWindowAttentionH3Impl:
    _prepare_flash_attention()
    impl = HybridWindowAttentionH3Impl(
        num_heads=HEADS,
        head_size=HEAD_DIM,
        causal=False,
        softmax_scale=HEAD_DIM**-0.5,
        num_kv_heads=HEADS,
        prefix="blocks.3.attn",
    )
    assert impl.layer_idx == 3
    return impl


def _run(impl, meta, layout, q, k, v, gate=None):
    cu = torch.tensor(
        [0, layout.used, layout.seq_len], dtype=torch.int32, device=q.device
    )
    return impl.forward_varlen(
        q,
        k,
        v,
        cu_seqlens=cu,
        max_seqlen=layout.used,
        cu_seqlens_host=(0, layout.used, layout.seq_len),
        attn_metadata=meta,
        softmax_gate=gate,
    )


def test_window_bounds_and_anchor_frames() -> None:
    hybrid = _hybrid()
    bounds, dense_rows, dense_cols = window_mask_frames(hybrid, NUM_FRAMES)
    # chunk 5, radius 1: frame 7 (chunk 1) sees chunks 0..2 = frames 0..14 -> clamped 11
    assert bounds[7] == (0, 11)
    assert bounds[0] == (0, 9)
    assert bounds[11] == (5, 11)
    assert dense_rows == {0, NUM_FRAMES - 1} and dense_cols == {0, NUM_FRAMES - 1}
    assert not hybrid.full_cover(NUM_FRAMES)
    assert _hybrid(radius=NUM_FRAMES).full_cover(NUM_FRAMES)
    # 102 = 20 * 5 + 2: the last chunk is short but still a whole chunk
    raw = hybrid.window_bounds(102)
    assert raw[100] == raw[101] == (95, 109)
    assert raw[99] == (90, 104)


def test_mask_reference_partition_is_exact() -> None:
    """Every (video q, video k) pair is in the softmax window or in the
    linear branch's complement exactly once; anchors are absent from the
    branch."""
    layout = _layout()
    hybrid = _hybrid()
    mask = window_mask_reference(hybrid, layout, torch.device("cpu"))
    # globals dense both ways
    assert mask[: layout.video_start].all() and mask[:, : layout.video_start].all()
    assert mask[layout.video_end : layout.used].all()
    bounds = hybrid.window_bounds(NUM_FRAMES)
    vs, tpf = layout.video_start, TOKENS_PER_FRAME
    for qf in range(NUM_FRAMES):
        row = mask[vs + qf * tpf, vs : layout.video_end].view(NUM_FRAMES, tpf)
        frame_kept = row.all(dim=1)
        assert (frame_kept == row.any(dim=1)).all()  # whole frames
        if qf in (0, NUM_FRAMES - 1):
            assert frame_kept.all()
            continue
        lo, hi = max(bounds[qf][0], 0), min(bounds[qf][1], NUM_FRAMES - 1)
        expected_softmax = {f for f in range(lo, hi + 1)} | {0, NUM_FRAMES - 1}
        # the branch covers the inner frames 1..F-2 outside the window
        expected_linear = {f for f in range(1, NUM_FRAMES - 1) if f < lo or f > hi}
        kept = {f for f in range(NUM_FRAMES) if frame_kept[f]}
        assert kept == expected_softmax
        assert kept.isdisjoint(expected_linear)
        assert kept | expected_linear == set(range(NUM_FRAMES))


@requires_cuda
def test_window_matches_masked_dense() -> None:
    device = torch.device("cuda")
    layout, (q, k, v) = _qkv(device)
    hybrid = _hybrid()
    meta = HybridWindowAttentionH3MetadataBuilder().build(
        layout=layout, hybrid=hybrid, device=device
    )
    assert not meta.full_cover
    out = _run(_impl(), meta, layout, q, k, v)
    mask = window_mask_reference(hybrid, layout, device)
    reference = _masked_reference(q, k, v, mask, layout.used)
    diff = (out[: layout.used].float() - reference).abs().max().item()
    assert diff < 2e-2, f"window vs masked dense max diff {diff}"
    assert torch.all(out[layout.used :] == 0)


@requires_cuda
def test_full_cover_matches_dense() -> None:
    device = torch.device("cuda")
    layout, (q, k, v) = _qkv(device, seed=11)
    hybrid = _hybrid(radius=NUM_FRAMES)
    meta = HybridWindowAttentionH3MetadataBuilder().build(
        layout=layout, hybrid=hybrid, device=device
    )
    assert meta.full_cover
    out = _run(_impl(), meta, layout, q, k, v)
    full = torch.ones(layout.used, layout.used, dtype=torch.bool, device=device)
    reference = _masked_reference(q, k, v, full, layout.used)
    diff = (out[: layout.used].float() - reference).abs().max().item()
    assert diff < 2e-2, f"full cover vs dense max diff {diff}"


@requires_cuda
def test_decomposed_passes_are_arithmetic_neutral() -> None:
    """Bounding the gathered K/V rows per pass splits the window into several
    varlen calls without changing any query's kept set."""
    device = torch.device("cuda")
    layout, (q, k, v) = _qkv(device, seed=9)
    one = HybridWindowAttentionH3MetadataBuilder().build(
        layout=layout, hybrid=_hybrid(), device=device
    )
    many = HybridWindowAttentionH3MetadataBuilder().build(
        layout=layout, hybrid=_hybrid(), device=device, max_gather_rows=1
    )
    assert len(one.decomposed.passes) == 1 and len(many.decomposed.passes) > 1
    impl = _impl()
    a = _run(impl, one, layout, q, k, v).clone()
    b = _run(impl, many, layout, q, k, v)
    assert torch.equal(a, b)


@requires_cuda
def test_dense_fallback_off_the_dit_blocks() -> None:
    """The token refiner resolves the same backend but runs plain dense FA."""
    device = torch.device("cuda")
    layout, (q, k, v) = _qkv(device, seed=5)
    _prepare_flash_attention()
    impl = HybridWindowAttentionH3Impl(
        num_heads=HEADS,
        head_size=HEAD_DIM,
        causal=False,
        softmax_scale=HEAD_DIM**-0.5,
        num_kv_heads=HEADS,
        prefix="token_refiner.blocks.0.attn",
    )
    assert impl.layer_idx is None
    out = _run(impl, None, layout, q, k, v)
    full = torch.ones(layout.used, layout.used, dtype=torch.bool, device=device)
    reference = _masked_reference(q, k, v, full, layout.used)
    assert (out[: layout.used].float() - reference).abs().max().item() < 2e-2


@requires_cuda
def test_metadata_layout_mismatch_is_rejected() -> None:
    device = torch.device("cuda")
    layout, (q, k, v) = _qkv(device)
    meta = HybridWindowAttentionH3MetadataBuilder().build(
        layout=layout, hybrid=_hybrid(), device=device
    )
    cu = torch.tensor(
        [0, layout.used - 48, layout.seq_len], dtype=torch.int32, device=device
    )
    with pytest.raises(ValueError, match="diverged"):
        _impl().forward_varlen(
            q,
            k,
            v,
            cu_seqlens=cu,
            max_seqlen=layout.used - 48,
            cu_seqlens_host=(0, layout.used - 48, layout.seq_len),
            attn_metadata=meta,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
