"""Unit tests for per-position ``token_ids_logprob`` gathering.

Covers the additive per-position path in ``sglang.srt.layers.utils.logprob`` used by
OPD top-k teacher scoring: each input position may carry its own id-list
(``List[List[int]]``) so the response is a sparse ``[R, k]`` instead of the dense
``[R, |union|]`` you get from a flat global union.

Alignment note (the subtle part): a per-position list is indexed by *absolute input
position*, but pruned row ``r`` holds the logits that *generated* the token at
position ``start_len + r + 1``. The response assembler then prepends a ``None`` for
position 0 and pops the trailing sample row
(see ``_process_input_token_ids_logprobs``), so the gather shifts each lookup by
``start_len + 1`` to make the returned ``input_token_ids_logprobs`` line up 1:1 with
the caller's per-position list. ``start_len`` is the absolute position of the
request's first pruned row (``global_start_pos`` = prefix_len + extend_logprob_start_len)
so the alignment stays correct under chunked prefill, where each forward pass scores
only a slice of the full sequence. ``test_end_to_end_alignment_matches_flat`` and
``test_per_position_chunked_prefill_offset`` pin the single-pass and cross-pass contracts.

These are pure-tensor tests (CPU); no server/GPU required.
"""

import pytest
import torch

from sglang.srt.layers.utils.logprob import (
    LogprobStage,
    _is_per_position_token_ids,
    get_token_ids_logprobs_chunk,
    get_token_ids_logprobs_raw,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def test_detection():
    assert _is_per_position_token_ids([[1, 2], [3, 4]]) is True
    assert _is_per_position_token_ids([1, 2, 3]) is False
    assert _is_per_position_token_ids(None) is False
    assert _is_per_position_token_ids([]) is False


def test_prefill_flat_unchanged():
    # Regression: the flat path must behave exactly as before (same ids every position).
    torch.manual_seed(0)
    lp = torch.randn(3, 8)
    vals, idxs = get_token_ids_logprobs_raw(
        lp, [[2, 5]], stage=LogprobStage.PREFILL, extend_logprob_pruned_lens_cpu=[3]
    )
    assert idxs == [[[2, 5], [2, 5], [2, 5]]]
    expected = lp[:, [2, 5]].tolist()
    assert vals[0] == expected


def test_prefill_per_position_shift():
    # 5 absolute positions; the id-list for position p is gathered at row p-1 (the
    # row whose logits generated token p), and the trailing sample row is padded empty.
    torch.manual_seed(1)
    lp = torch.randn(5, 10)
    pos_ids = [[], [], [1, 3], [0, 9], [4, 7]]  # indexed by absolute position
    vals, idxs = get_token_ids_logprobs_raw(
        lp, [pos_ids], stage=LogprobStage.PREFILL, extend_logprob_pruned_lens_cpu=[5]
    )
    # position p's ids shift down to row p-1; row 4 (sample) becomes empty padding.
    assert idxs == [[[], [1, 3], [0, 9], [4, 7], []]]
    assert vals[0][0] == []
    assert vals[0][1] == [lp[1, 1].item(), lp[1, 3].item()]
    assert vals[0][2] == [lp[2, 0].item(), lp[2, 9].item()]
    assert vals[0][3] == [lp[3, 4].item(), lp[3, 7].item()]
    assert vals[0][4] == []


def test_prefill_per_position_multi_request_packed():
    # Two requests packed together: per-position (pruned_len 3) then flat (pruned_len 2).
    torch.manual_seed(2)
    lp = torch.randn(5, 10)
    vals, idxs = get_token_ids_logprobs_raw(
        lp,
        [[[1, 2], [3, 4], [5, 6]], [7, 8]],
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=[3, 2],
    )
    # request 0 (per-position): position p's ids gathered at row p-1, last row padded.
    assert idxs[0] == [[3, 4], [5, 6], []]
    assert vals[0][0] == [lp[0, 3].item(), lp[0, 4].item()]
    assert vals[0][1] == [lp[1, 5].item(), lp[1, 6].item()]
    assert vals[0][2] == []
    # request 1 (flat) starts at row 3, unchanged behavior
    assert vals[1] == lp[3:5, [7, 8]].tolist()
    assert idxs[1] == [[7, 8], [7, 8]]


def test_decode_rejects_per_position():
    lp = torch.randn(1, 10)
    with pytest.raises(AssertionError):
        get_token_ids_logprobs_raw(lp, [[[1, 2]]], stage=LogprobStage.DECODE)


def test_chunk_per_position_split_across_two_chunks():
    # A single 4-position sequence split across two 2-row chunks. The per-position
    # lookup applies the same start_len+1 shift, so position p's ids land at row p-1
    # across the chunk boundary; the trailing sample row is padded empty.
    torch.manual_seed(3)
    lp = torch.randn(4, 10)
    pos_ids = [[1, 2], [3, 4], [5, 6], [7, 8]]  # whole 4-position sequence
    val, idx = [], []

    remaining = get_token_ids_logprobs_chunk(
        lp[0:2], [pos_ids], [4], val, idx, split_pruned_len=0
    )
    assert remaining == 2  # ran out of rows after 2 positions
    assert idx[0] == [[3, 4], [5, 6]]

    remaining = get_token_ids_logprobs_chunk(
        lp[2:4], [pos_ids], [4], val, idx, split_pruned_len=2
    )
    assert remaining == 0
    # rows 0..3 score positions 1..4; position 4 is out of range -> empty padding.
    assert idx[0] == [[3, 4], [5, 6], [7, 8], []]
    assert val[0][0] == [lp[0, 3].item(), lp[0, 4].item()]
    assert val[0][1] == [lp[1, 5].item(), lp[1, 6].item()]
    assert val[0][2] == [lp[2, 7].item(), lp[2, 8].item()]
    assert val[0][3] == []


def _assemble(vals_for_req):
    # Replay scheduler_output_processor_mixin._process_input_token_ids_logprobs:
    # prepend None for position 0, then drop the trailing sample row.
    out = [None]
    out.extend(vals_for_req)
    out.pop()
    return out


def test_end_to_end_alignment_matches_flat():
    # The contract test: after the response assembly + miles' values[1:][-R:] trim,
    # per-position logprobs must equal the flat-union logprobs at every (resp_pos, id).
    n, r_len, k, prompt_len = 40, 16, 5, 24
    per_pos = [[] for _ in range(prompt_len)] + [
        list(range(5 + r * k, 5 + r * k + k)) for r in range(r_len)
    ]
    union = sorted({t for pos in per_pos for t in pos})
    # value encodes (row, id) so any row/id misalignment is detectable.
    rows = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    cols = torch.arange(100, dtype=torch.float64).unsqueeze(0) / 1000.0
    lp = rows + cols

    fv, _ = get_token_ids_logprobs_raw(lp, [union], LogprobStage.PREFILL, [n])
    pv, pi = get_token_ids_logprobs_raw(lp, [per_pos], LogprobStage.PREFILL, [n])

    vf = _assemble(fv[0])
    vp, vp_idx = _assemble(pv[0]), _assemble(pi[0])
    assert len(vf) == len(vp) == n

    flat_vals = vf[1:][-r_len:]
    pp_vals, pp_ids = vp[1:][-r_len:], vp_idx[1:][-r_len:]
    compared = 0
    for r in range(r_len):
        fmap = {tid: v for tid, v in zip(union, flat_vals[r])}
        pmap = {tid: v for tid, v in zip(pp_ids[r], pp_vals[r])}
        for tid in per_pos[prompt_len + r]:
            assert tid in pmap, f"id {tid} missing at resp pos {r}: ids={pp_ids[r]}"
            assert pmap[tid] == fmap[tid], (r, tid, pmap[tid], fmap[tid])
            compared += 1
    assert compared == r_len * k


def test_per_position_chunked_prefill_offset():
    # Cross-forward-pass chunked prefill: a scored sequence split across two passes.
    # Each pass calls the gather with the FULL per-position list but its OWN per-pass
    # pruned_len and global_start_pos (= prefix_len + extend_logprob_start_len). Without
    # the offset, a non-final chunk silently gathers the wrong absolute position; with
    # it, the assembled + trimmed result equals the single-pass flat-union baseline.
    n, r_len, k, prompt_len, split = 40, 16, 5, 24, 30
    per_pos = [[] for _ in range(prompt_len)] + [
        list(range(5 + r * k, 5 + r * k + k)) for r in range(r_len)
    ]
    union = sorted({t for pos in per_pos for t in pos})
    rows = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    cols = torch.arange(100, dtype=torch.float64).unsqueeze(0) / 1000.0
    lp = rows + cols  # lp[row, id] = row + id/1000 ; row is absolute

    # Two forward passes: rows [0, split) then [split, n).
    v1, i1 = get_token_ids_logprobs_raw(
        lp[0:split], [per_pos], LogprobStage.PREFILL, [split], global_start_pos_cpu=[0]
    )
    v2, _ = get_token_ids_logprobs_raw(
        lp[split:n],
        [per_pos],
        LogprobStage.PREFILL,
        [n - split],
        global_start_pos_cpu=[split],
    )
    vp = _assemble(v1[0] + v2[0])
    vp_idx = _assemble(
        i1[0]
        + get_token_ids_logprobs_raw(
            lp[split:n],
            [per_pos],
            LogprobStage.PREFILL,
            [n - split],
            global_start_pos_cpu=[split],
        )[1][0]
    )
    vf = _assemble(
        get_token_ids_logprobs_raw(lp, [union], LogprobStage.PREFILL, [n])[0][0]
    )
    assert len(vp) == len(vf) == n

    flat_vals = vf[1:][-r_len:]
    pp_vals, pp_ids = vp[1:][-r_len:], vp_idx[1:][-r_len:]
    for r in range(r_len):
        fmap = {tid: v for tid, v in zip(union, flat_vals[r])}
        pmap = {tid: v for tid, v in zip(pp_ids[r], pp_vals[r])}
        for tid in per_pos[prompt_len + r]:
            assert pmap.get(tid) == fmap[tid], (r, tid)

    # The offset is load-bearing. Row prompt_len-1 (in pass 1) generates the first
    # response token (abs position prompt_len): with the offset it gathers that
    # position's ids; with the single-pass formula (global_start_pos=None) it overshoots.
    _, i_bad = get_token_ids_logprobs_raw(
        lp[0:split], [per_pos], LogprobStage.PREFILL, [split], global_start_pos_cpu=None
    )
    assert i1[0][prompt_len - 1] == per_pos[prompt_len]
    assert i_bad[0][prompt_len - 1] != per_pos[prompt_len]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
