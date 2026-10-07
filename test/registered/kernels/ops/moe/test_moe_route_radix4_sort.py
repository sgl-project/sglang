# SPDX-License-Identifier: Apache-2.0
"""Fused radix4+sort keeps the radix-4 row and emits a valid BM=16 MoE layout."""

import sys
from collections import defaultdict

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

if not is_hip():
    pytest.skip("The radix-4 router is the ROCm path.", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

from sglang.kernels.ops.moe import moe_route_radix4

if not moe_route_radix4.supported_hardware():
    pytest.skip("The kernel targets gfx942/gfx950.", allow_module_level=True)

moe_route_radix4.build()
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

NUM_EXPERTS = 896
TOPK = 16
BM = 16
MODEL_DIM = 3584


def _payload_from_route(ids, weights):
    bag = defaultdict(list)
    m = ids.shape[0]
    for token in range(m):
        for slot in range(TOPK):
            bag[int(ids[token, slot])].append(
                (token, slot, float(weights[token, slot]))
            )
    for expert in bag:
        bag[expert].sort()
    return bag


def _payload_from_sorted(
    sorted_token_ids, sorted_weights, sorted_expert_ids, n_blocks, m
):
    bag = defaultdict(list)
    for block in range(n_blocks):
        expert = int(sorted_expert_ids[block])
        start = block * BM
        live = []
        for row in range(BM):
            packed = int(sorted_token_ids[start + row])
            token = packed & 0x00FFFFFF
            slot = (packed >> 24) & 0xFF
            weight = float(sorted_weights[start + row])
            if token == m:
                assert weight == 0.0
                continue
            live.append((token, slot, weight))
        if live:
            bag[expert].extend(live)
    for expert in bag:
        bag[expert].sort()
    return bag


# 1 is the grid that skips the arrival handshake, 1..64 is what decode fuses,
# and 128 is past the point where the leader can hold the pairs in registers and
# has to stream them instead.
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 64, 128])
def test_fused_sort_matches_route_and_layout(m):
    generator = torch.Generator(device="cuda").manual_seed(7 + m)
    scores = torch.randn(
        (m, NUM_EXPERTS), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    bias = torch.randn(
        NUM_EXPERTS, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    moe_buf = torch.ones((m, MODEL_DIM), dtype=torch.bfloat16, device="cuda")

    ref_w, ref_ids = moe_route_radix4.route_radix4(
        scores, bias, TOPK, True, 2.5, fuse_sort=False
    )
    w, ids, aux = moe_route_radix4.route_radix4_with_sort(
        scores, bias, TOPK, True, 2.5, moe_buf=moe_buf
    )

    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(w, ref_w, rtol=0, atol=0)
    assert torch.count_nonzero(moe_buf).item() == 0
    assert int(aux["num_valid_ids"][1]) == m

    expected = _payload_from_route(ref_ids.cpu(), ref_w.cpu())
    got = _payload_from_sorted(
        aux["sorted_token_ids"].cpu(),
        aux["sorted_weights"].cpu(),
        aux["sorted_expert_ids"].cpu(),
        int(aux["num_valid_ids"][0]) // BM,
        m,
    )
    assert got.keys() == expected.keys()
    for expert in expected:
        assert got[expert] == expected[expert]

    used = int(aux["num_valid_ids"][0])
    m_indices = aux["m_indices"].cpu()[:used]
    packed = aux["sorted_token_ids"].cpu()[:used]
    for i in range(used):
        token = int(packed[i]) & 0x00FFFFFF
        assert int(m_indices[i]) == token

    reverse = aux["reverse_sorted"].cpu()
    for token in range(m):
        for slot in range(TOPK):
            pos = int(reverse[token * TOPK + slot])
            packed_i = int(packed[pos])
            assert (packed_i & 0x00FFFFFF) == token
            assert ((packed_i >> 24) & 0xFF) == slot


def test_fused_sort_zeros_moe_buf():
    m = 2
    scores = torch.randn((m, NUM_EXPERTS), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    moe_buf = torch.ones((m, MODEL_DIM), dtype=torch.bfloat16, device="cuda")
    moe_route_radix4.route_radix4_with_sort(scores, bias, TOPK, True, 2.5, moe_buf)
    assert torch.count_nonzero(moe_buf).item() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
