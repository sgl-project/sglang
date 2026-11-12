# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_ACCEPT_RAND_HOIST: the accept's uniforms drawn on a side stream before the verify must equal, bit
for bit, the ones the served order draws after it. The reference is the served order itself: per round a captured draft
graph with an in-graph philox draw (the selector head's form), a captured verify graph with no RNG, then the accept's
two torch.rand calls and one later consumer (the forced-AL simulator's place), all on the main stream."""

import os
import random

import pytest
import torch

from sglang.srt.speculative.dflash_accept_rand import AcceptRandHoist
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

DEV = "cuda"
BLOCK = 8
MAX_BS = 8
ROUNDS = 1000
# Test-driver mutant: draw before the draft replay (the wrong hoist point); must fail the bitwise test.
MUTATION = os.environ.get("K31_MUTATION", "")


def _graphs():
    draft_uniforms = torch.empty((MAX_BS, BLOCK - 1), dtype=torch.float32, device=DEV)
    draft_out = torch.empty_like(draft_uniforms)
    verify_in = torch.randn((256, 256), device=DEV)
    verify_out = torch.empty_like(verify_in)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm up outside capture
        draft_out.copy_(draft_uniforms.uniform_() * 2.0)
        verify_out.copy_(verify_in @ verify_in)
    torch.cuda.current_stream().wait_stream(side)
    draft, verify = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(draft):
        draft_out.copy_(draft_uniforms.uniform_() * 2.0)
    with torch.cuda.graph(verify):
        verify_out.copy_(verify_in @ verify_in)
    # The graphs replay into these tensors: they must outlive every replay, or later allocations land in their memory.
    return draft, verify, draft_out, (draft_uniforms, verify_in, verify_out)


def _run(hoist: bool, seed: int, interloper: bool = False):
    draft, verify, draft_out, _keepalive = _graphs()
    torch.cuda.manual_seed(seed)
    sizes = random.Random(seed)
    hoister = AcceptRandHoist(DEV) if hoist else None
    out = []
    for _ in range(ROUNDS):
        bs = sizes.randint(1, MAX_BS)
        if hoister is not None and MUTATION == "hoist_early":
            hoister.draw(bs, BLOCK)
        draft.replay()
        out.append(draft_out[:bs].clone())
        if hoister is not None and MUTATION != "hoist_early":
            hoister.draw(bs, BLOCK)
        if interloper:
            torch.rand((1,), device=DEV)
        verify.replay()
        if hoister is None:
            uniforms = torch.rand((bs, BLOCK - 1), dtype=torch.float32, device=DEV)
            uniforms_final = torch.rand((bs,), dtype=torch.float32, device=DEV)
        else:
            uniforms, uniforms_final = hoister.take(bs, BLOCK)
        out += [uniforms.clone(), uniforms_final.clone()]
        out.append(torch.rand((bs,), dtype=torch.float32, device=DEV))
    torch.cuda.synchronize()
    return out, hoister


@pytest.mark.parametrize("seed", [0, 1234])
def test_hoisted_uniforms_are_bitwise_served(seed):
    served, _ = _run(hoist=False, seed=seed)
    hoisted, hoister = _run(hoist=True, seed=seed)
    assert len(served) == len(hoisted)
    bad = [i for i, (a, b) in enumerate(zip(served, hoisted)) if not torch.equal(a, b)]
    print(f"R1 seed {seed}: {len(served)} tensors, {len(bad)} differ, taken {hoister.taken}")
    assert not bad, f"first differing tensor {bad[0]} (round {bad[0] // 4}, slot {bad[0] % 4})"
    assert hoister.taken == ROUNDS


def test_consumer_between_draw_and_take_raises():
    with pytest.raises(RuntimeError, match="consumed between"):
        _run(hoist=True, seed=7, interloper=True)


def test_shape_mismatch_raises():
    hoister = AcceptRandHoist(DEV)
    hoister.draw(2, BLOCK)
    with pytest.raises(RuntimeError, match="drew"):
        hoister.take(3, BLOCK)


def test_untaken_draw_raises():
    hoister = AcceptRandHoist(DEV)
    hoister.draw(2, BLOCK)
    with pytest.raises(RuntimeError, match="never taken"):
        hoister.draw(2, BLOCK)
