"""Segmented recurrent-state pass for the Triton ``chunk_kda`` prefill kernel.

``chunk_kda(..., state_segments=S)`` splits every sequence into ``S``
chunk-aligned segments, runs the sequential state kernel on all segments in
parallel and folds the segment states (``h_in @ M_s + B_s``) before re-running
the segments from their true incoming states. The output and the written-back
final state must match the unsegmented kernel up to the extra fp32/bf16
rounding of the fold; sequences too short to split must take the unsegmented
path bit for bit.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("chunk_kda state segments need a CUDA device", allow_module_level=True)

from sglang.kernels.ops.attention.fla.index import prepare_chunk_offsets  # noqa: E402
from sglang.kernels.ops.attention.fla.kda import (  # noqa: E402
    chunk_kda,
    prepare_segment_cu_seqlens,
)

CHUNK = 64


def _inputs(seed, lens, num_heads=4, head_dim=128, with_state=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(lens)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, device="cuda", dtype=dtype)

    q = randn(1, total, num_heads, head_dim)
    k = randn(1, total, num_heads, head_dim)
    v = (0.1 * randn(1, total, num_heads, head_dim, dtype=torch.float32)).to(
        torch.bfloat16
    )
    gate = randn(1, total, num_heads, head_dim)
    beta = torch.sigmoid(randn(1, total, num_heads, dtype=torch.float32)).to(
        torch.bfloat16
    )
    a_log = randn(num_heads, dtype=torch.float32)
    dt_bias = randn(num_heads * head_dim, dtype=torch.float32)
    state = torch.zeros(
        len(lens), num_heads, head_dim, head_dim, device="cuda", dtype=torch.float32
    )
    if with_state:
        state = 0.05 * randn(*state.shape, dtype=torch.float32)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    return q, k, v, gate, beta, a_log, dt_bias, state, cu_seqlens


def _run(inputs, state_segments, indices=None):
    # chunk_kda writes its output into ``v`` in place and the final state into
    # ``state`` in place: hand every run its own copies.
    q, k, v, gate, beta, a_log, dt_bias, state, cu_seqlens = (t.clone() for t in inputs)
    if indices is None:
        indices = torch.arange(state.shape[0], device="cuda", dtype=torch.int32)
    o = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=state,
        initial_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        state_segments=state_segments,
    )
    return o.float(), state


def _rel_err(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


@pytest.mark.parametrize(
    "lens",
    [(3000,), (4096, 8192), (5000, 1000, 7000), (64 * 8 + 1, 64 * 9)],
)
@pytest.mark.parametrize("segments", [4, 8])
@pytest.mark.parametrize("with_state", [False, True])
def test_segmented_matches_unsegmented(lens, segments, with_state):
    inputs = _inputs(
        hash((lens, segments, with_state)) % 2**31, lens, with_state=with_state
    )
    o_ref, state_ref = _run(inputs, 1)
    o_seg, state_seg = _run(inputs, segments)
    assert torch.isfinite(o_seg).all() and torch.isfinite(state_seg).all()
    assert _rel_err(o_seg, o_ref) < 1e-2, _rel_err(o_seg, o_ref)
    assert ((o_seg - o_ref).abs().mean() / o_ref.abs().mean()).item() < 1e-3
    assert _rel_err(state_seg, state_ref) < 2e-2, _rel_err(state_seg, state_ref)


def test_short_sequence_falls_back_bit_exactly():
    # 100 tokens cannot be split into 2 chunk-aligned segments -> unsegmented path.
    inputs = _inputs(7, (100, 3000), with_state=True)
    o_ref, state_ref = _run(inputs, 1)
    o_seg, state_seg = _run(inputs, 8)
    assert torch.equal(o_seg, o_ref) and torch.equal(state_seg, state_ref)


def test_segment_boundaries_are_chunk_aligned_and_non_empty():
    cu = torch.tensor([0, 3000, 3000 + 4096, 3000 + 4096 + 777], device="cuda")
    for segments in (2, 4, 8):
        seg = prepare_segment_cu_seqlens(cu, segments, CHUNK)
        s = (seg.numel() - 1) // 3
        assert 1 < s <= segments
        seg2 = seg.view(-1)
        assert (seg2[1:] > seg2[:-1]).all(), "empty segment"
        starts = seg2[:-1].view(3, s)
        assert torch.equal(starts[:, 0], cu[:-1])
        assert (((starts - cu[:-1, None]) % CHUNK) == 0).all()
        # same global chunk count as the unsegmented enumeration
        assert (
            prepare_chunk_offsets(seg, CHUNK)[-1]
            == prepare_chunk_offsets(cu, CHUNK)[-1]
        )
    assert (
        prepare_segment_cu_seqlens(
            torch.tensor([0, 100, 4196], device="cuda"), 8, CHUNK
        )
        is None
    )


def test_padded_row_slot_minus_one():
    # A padded batch row carries slot -1: zero incoming state, no write-back.
    inputs = _inputs(11, (2048, 4096), with_state=True)
    indices = torch.tensor([1, -1], device="cuda", dtype=torch.int32)
    o_ref, state_ref = _run(inputs, 1, indices)
    o_seg, state_seg = _run(inputs, 8, indices)
    assert _rel_err(o_seg, o_ref) < 1e-2
    assert _rel_err(state_seg, state_ref) < 2e-2
    # slot 0 is untouched by both runs and must still equal the input state.
    assert torch.equal(state_seg[0], inputs[7][0])
    assert torch.equal(state_ref[0], inputs[7][0])
