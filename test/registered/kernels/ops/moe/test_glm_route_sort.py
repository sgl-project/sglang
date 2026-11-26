# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ROCm GLM-5.2 route+sort kernel.

The kernel stands in for three aiter launches on decode-sized batches: the
biased top-k, the shared-expert append and moe_sorting. Each case runs those
three as the reference and holds the kernel to them: the routing bit for bit,
the sort row for row on real tokens. CUDA-graph padding rows are compared as a
bag, since opus keeps one of a padding row's expert-0 routes and which one is
its own business.

Run:  SGLANG_USE_AITER=1 pytest test/registered/kernels/ops/moe/test_glm_route_sort.py -v
"""

import sys
from collections import Counter

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

if not is_hip():
    pytest.skip("The GLM route+sort kernel is the ROCm path.", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

from sglang.kernels.ops.moe import glm_route_sort

if not glm_route_sort.supported_hardware():
    pytest.skip("The kernel targets gfx942/gfx950.", allow_module_level=True)

# Built here rather than gated on available(), so a kernel that stopped
# compiling fails the run instead of skipping it.
glm_route_sort.build()

try:
    from aiter import biased_grouped_topk as aiter_biased_grouped_topk
    from aiter.fused_moe import moe_sorting as aiter_moe_sorting
except ImportError:
    pytest.skip("Requires aiter.", allow_module_level=True)

from sglang.kernels.ops.moe.fused_moe_triton_kernels import fused_append_shared_experts

register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd-mi35x")

E = glm_route_sort.NUM_EXPERTS
TOPK = glm_route_sort.TOPK
MODEL_DIM = 6144
SCALING = 2.5
SHARED_WEIGHT = 1.0


def _inputs(m, dtype, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.randn((m, E), device="cuda", generator=gen).to(dtype)
    bias = torch.randn(E, device="cuda", generator=gen) * 0.1
    return scores, bias


def _expert_mask(routed_begin, num_routed):
    if num_routed == E:
        return None
    mask = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
    mask[routed_begin : routed_begin + num_routed] = 1
    mask[E] = 1
    return mask


def _aiter_route_and_sort(scores, bias, n_real, block_m, routed_begin, num_routed):
    """What the aiter path runs for GLM-5.2: route, append, mask padding, sort."""
    m = scores.shape[0]
    weights = torch.empty((m, TOPK - 1), dtype=torch.float32, device="cuda")
    ids = torch.empty((m, TOPK - 1), dtype=torch.int32, device="cuda")
    aiter_biased_grouped_topk(scores.float(), bias, weights, ids, 1, 1, True, SCALING)
    ids, weights = fused_append_shared_experts(ids, weights, 1, SHARED_WEIGHT, E)
    ids, weights = ids.clone(), weights.clone()
    ids[n_real:, : TOPK - 1] = 0
    weights[n_real:] = 0.0
    sort = aiter_moe_sorting(
        ids,
        weights,
        E + 1,
        MODEL_DIM,
        torch.bfloat16,
        block_m,
        _expert_mask(routed_begin, num_routed),
    )
    return weights, ids, sort[:4]


def _route_sort(scores, bias, n_real, block_m, routed_begin, num_routed, moe_buf=None):
    # n_real=None passes no padding count, which a graph capture needs: building
    # the count tensor there would be a host-to-device copy.
    return glm_route_sort.glm_route_sort(
        scores,
        bias,
        routed_begin=routed_begin,
        num_routed=num_routed,
        block_m=block_m,
        routed_scaling_factor=SCALING,
        shared_weight=SHARED_WEIGHT,
        moe_buf=moe_buf,
        num_token_non_padded=(
            None
            if n_real is None
            else torch.tensor([n_real], dtype=torch.int32, device=scores.device)
        ),
    )


def _padding_bag(sorted_ids, sorted_weights, sorted_experts, n, block_m, n_real):
    bag = Counter()
    for row in range(n):
        packed = int(sorted_ids[row])
        token = packed & 0x00FFFFFF
        if token >= n_real:
            bag[
                (int(sorted_experts[row // block_m]), token, float(sorted_weights[row]))
            ] += 1
    return bag


def _assert_matches_aiter(m, n_real, dtype, block_m, routed_begin, num_routed, seed):
    scores, bias = _inputs(m, dtype, seed)
    moe_buf = torch.ones((m, MODEL_DIM), dtype=torch.bfloat16, device="cuda")
    w, ids, sort = _route_sort(
        scores, bias, n_real, block_m, routed_begin, num_routed, moe_buf=moe_buf
    )
    ref_w, ref_ids, ref = _aiter_route_and_sort(
        scores, bias, n_real, block_m, routed_begin, num_routed
    )
    assert torch.equal(ids, ref_ids)
    assert torch.equal(w, ref_w)
    assert torch.count_nonzero(moe_buf).item() == 0

    got = [
        sort[k].cpu()
        for k in ("sorted_token_ids", "sorted_weights", "sorted_expert_ids")
    ]
    want = [t.cpu() for t in ref[:3]]
    num_valid, ref_num_valid = sort["num_valid_ids"].cpu(), ref[3].cpu()
    assert int(num_valid[0]) == int(ref_num_valid[0])
    assert int(num_valid[1]) == m
    n = int(num_valid[0])
    assert torch.equal(got[2][: n // block_m], want[2][: n // block_m])
    real = (want[0][:n] & 0x00FFFFFF) < n_real
    assert torch.equal(got[0][:n][real], want[0][:n][real])
    assert torch.equal(got[1][:n][real], want[1][:n][real])
    assert _padding_bag(*got, n, block_m, n_real) == _padding_bag(
        *want, n, block_m, n_real
    )


# 1 and 8 bound the single-block launch, 9 is the first multi-block one and 64
# the cap; (64, 64) and (192, 64) are EP4 ranks, (0, 32) an EP8 one.
@pytest.mark.parametrize("m", [1, 2, 7, 8, 9, 33, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("block_m", [16, 32, 64])
@pytest.mark.parametrize("ep", [(0, 256), (64, 64), (192, 64), (0, 32)])
def test_matches_aiter_route_and_sort(m, dtype, block_m, ep):
    _assert_matches_aiter(m, m, dtype, block_m, *ep, seed=m * 131 + block_m)


@pytest.mark.parametrize("m, n_real", [(4, 1), (12, 7), (64, 33), (9, 0), (1, 0)])
def test_padded_rows_route_like_the_hip_mask(m, n_real):
    """select_experts masks CUDA-graph padding rows to expert 0 and weight 0 on
    HIP, and AITER matches the sort by tensor address, so the sort has to
    describe the masked rows."""
    _assert_matches_aiter(m, n_real, torch.bfloat16, 32, 64, 64, seed=m + n_real)


def test_multi_block_graph_replay():
    """The sorting block resets the arrival counter; without it every replay
    after the first would elect no sorting block."""
    m = 33
    scores, bias = _inputs(m, torch.float32, seed=5)
    moe_buf = torch.ones((m, MODEL_DIM), dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Allocates this stream's arrival counter outside the capture.
        _route_sort(scores, bias, None, 32, 0, E)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        w, ids, sort = _route_sort(scores, bias, None, 32, 0, E, moe_buf=moe_buf)
    ref_w, ref_ids, ref = _aiter_route_and_sort(scores, bias, m, 32, 0, E)
    n = int(ref[3][0])
    for _ in range(3):
        sort["sorted_token_ids"].fill_(-1)
        moe_buf.fill_(1.0)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ids, ref_ids)
        assert torch.equal(w, ref_w)
        assert torch.count_nonzero(moe_buf).item() == 0
        assert torch.equal(sort["sorted_token_ids"][:n], ref[0][:n])


def test_multi_block_waits_for_an_eager_counter_under_capture():
    """A counter first allocated inside a capture would only be zeroed when that
    graph replays, so ready() turns a fresh stream's multi-block launch away."""
    m = 33
    scores, _ = _inputs(m, torch.float32, seed=6)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        assert not glm_route_sort.ready(scores.device, m)
        assert glm_route_sort.ready(scores.device, 8)


def test_rejects_a_moe_buf_it_cannot_zero():
    """The sort tells AITER the buffer was zeroed, so a buffer with the wrong
    row count has to fail rather than be skipped."""
    m = 4
    scores, bias = _inputs(m, torch.float32, seed=3)
    moe_buf = torch.ones((m + 1, MODEL_DIM), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(Exception):
        _route_sort(scores, bias, m, 32, 0, E, moe_buf=moe_buf)


@pytest.mark.parametrize("ep_size, ep_rank", [(1, 0), (4, 1)])
@pytest.mark.parametrize("m", [1, 12, 64])
def test_aiter_accepts_the_registered_sort(monkeypatch, ep_size, ep_rank, m):
    """AITER drops a registered sort whose aux lacks a key it matches on, logs
    it once and sorts again, so a missing key only shows up as a lost speedup.
    The aux also has to hold the tensors it is matched on by address."""
    try:
        from aiter.presorted_moe import match_plain_presorted_moe
    except ImportError:
        pytest.skip("AITER without the plain-layout presort (ROCm/aiter#8045)")
    from types import SimpleNamespace

    from sglang.srt.layers.moe import glm_aiter_presort

    registrations = []
    monkeypatch.setattr(glm_aiter_presort, "_registrar", lambda: registrations.append)
    monkeypatch.setattr(
        glm_aiter_presort,
        "get_parallel",
        lambda: SimpleNamespace(moe_ep_size=ep_size, moe_ep_rank=ep_rank),
    )
    scores, bias = _inputs(m, torch.float32, seed=m)
    hidden = torch.empty((m, MODEL_DIM), dtype=torch.bfloat16, device="cuda")
    w, ids = glm_aiter_presort.route_and_register(
        hidden, scores, bias, SCALING, SHARED_WEIGHT, True, None
    )
    (aux,) = registrations
    output = glm_aiter_presort.take_zeroed_output(
        hidden.shape, hidden.dtype, hidden.device
    )
    assert aux["topk_ids"] is ids and aux["topk_weights"] is w
    assert aux["moe_buf"] is output
    assert aux["moe_buf_ptr"] == output.data_ptr()

    num_routed = E // ep_size
    mask = _expert_mask(ep_rank * num_routed, num_routed)
    assert (
        match_plain_presorted_moe(
            aux,
            ids,
            w,
            block_m=aux["block_m"],
            num_experts=E + 1,
            topk=TOPK,
            expert_mask=mask,
            output=output,
        )
        is aux
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
