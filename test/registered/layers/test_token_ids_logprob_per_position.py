"""Unit tests for per-position ``token_ids_logprob`` gathering."""

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
    torch.manual_seed(0)
    lp = torch.randn(3, 8)
    vals, idxs = get_token_ids_logprobs_raw(
        lp, [[2, 5]], stage=LogprobStage.PREFILL, extend_logprob_pruned_lens_cpu=[3]
    )
    assert idxs == [[[2, 5], [2, 5], [2, 5]]]
    expected = lp[:, [2, 5]].tolist()
    assert vals[0] == expected


def test_prefill_per_position_shift():
    torch.manual_seed(1)
    lp = torch.randn(5, 10)
    pos_ids = [[], [], [1, 3], [0, 9], [4, 7]]
    vals, idxs = get_token_ids_logprobs_raw(
        lp, [pos_ids], stage=LogprobStage.PREFILL, extend_logprob_pruned_lens_cpu=[5]
    )
    assert idxs == [[[], [1, 3], [0, 9], [4, 7], []]]
    assert vals[0][0] == []
    assert vals[0][1] == [lp[1, 1].item(), lp[1, 3].item()]
    assert vals[0][2] == [lp[2, 0].item(), lp[2, 9].item()]
    assert vals[0][3] == [lp[3, 4].item(), lp[3, 7].item()]
    assert vals[0][4] == []


def test_prefill_per_position_multi_request_packed():
    torch.manual_seed(2)
    lp = torch.randn(5, 10)
    vals, idxs = get_token_ids_logprobs_raw(
        lp,
        [[[1, 2], [3, 4], [5, 6]], [7, 8]],
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=[3, 2],
    )
    assert idxs[0] == [[3, 4], [5, 6], []]
    assert vals[0][0] == [lp[0, 3].item(), lp[0, 4].item()]
    assert vals[0][1] == [lp[1, 5].item(), lp[1, 6].item()]
    assert vals[0][2] == []
    assert vals[1] == lp[3:5, [7, 8]].tolist()
    assert idxs[1] == [[7, 8], [7, 8]]


def test_decode_rejects_per_position():
    lp = torch.randn(1, 10)
    with pytest.raises(AssertionError):
        get_token_ids_logprobs_raw(lp, [[[1, 2]]], stage=LogprobStage.DECODE)


def test_chunk_per_position_split_across_two_chunks():
    torch.manual_seed(3)
    lp = torch.randn(4, 10)
    pos_ids = [[1, 2], [3, 4], [5, 6], [7, 8]]
    val, idx = [], []

    remaining = get_token_ids_logprobs_chunk(
        lp[0:2], [pos_ids], [4], val, idx, split_pruned_len=0
    )
    assert remaining == 2
    assert idx[0] == [[3, 4], [5, 6]]

    remaining = get_token_ids_logprobs_chunk(
        lp[2:4], [pos_ids], [4], val, idx, split_pruned_len=2
    )
    assert remaining == 0
    assert idx[0] == [[3, 4], [5, 6], [7, 8], []]
    assert val[0][0] == [lp[0, 3].item(), lp[0, 4].item()]
    assert val[0][1] == [lp[1, 5].item(), lp[1, 6].item()]
    assert val[0][2] == [lp[2, 7].item(), lp[2, 8].item()]
    assert val[0][3] == []


def _assemble(vals_for_req):
    out = [None]
    out.extend(vals_for_req)
    out.pop()
    return out


def test_end_to_end_alignment_matches_flat():
    n, r_len, k, prompt_len = 40, 16, 5, 24
    per_pos = [[] for _ in range(prompt_len)] + [
        list(range(5 + r * k, 5 + r * k + k)) for r in range(r_len)
    ]
    union = sorted({t for pos in per_pos for t in pos})
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
    n, r_len, k, prompt_len, split = 40, 16, 5, 24, 30
    per_pos = [[] for _ in range(prompt_len)] + [
        list(range(5 + r * k, 5 + r * k + k)) for r in range(r_len)
    ]
    union = sorted({t for pos in per_pos for t in pos})
    rows = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    cols = torch.arange(100, dtype=torch.float64).unsqueeze(0) / 1000.0
    lp = rows + cols

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

    # Offset is required once the scored sequence spans more than one forward pass.
    _, i_bad = get_token_ids_logprobs_raw(
        lp[0:split], [per_pos], LogprobStage.PREFILL, [split], global_start_pos_cpu=None
    )
    assert i1[0][prompt_len - 1] == per_pos[prompt_len]
    assert i_bad[0][prompt_len - 1] != per_pos[prompt_len]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
