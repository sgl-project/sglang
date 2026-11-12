# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_DRAFT_P1_HPACK: the head-packed partial pass of the DFLASH draft split-KV attention
(kernels/ops/attention/dflash_draft_splitkv.py, hpack=True) is bitwise equal to the per-head pass (hpack=False), which
test_dflash_draft_splitkv.py checks against FP64 references.

Every workspace the module allocates is NaN-filled first, so a partial pass that leaves any (sequence, head, tile) row
unwritten shows up as NaN or a mismatch instead of reading the previous call's values from a reused block. Cases: draft
batch 2 / 3 / 4 / 8 / 16, ragged rails and query lengths, the served and
a short sliding window, non-unit KV scales, 12 query heads (six head groups), an
overlong rail (NaN rows for that sequence only, as the per-head pass), CUDA-graph replay, and batch 1 (per-head pass).
"""

import itertools

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

D, DV, PAGE = 576, 512, 64
SM_SCALE = 1.0 / (192**0.5)
MAX_KV = 1024 + PAGE - 1
DEV = "cuda"


def _mod():
    import sglang.kernels.ops.attention.dflash_draft_splitkv as m

    return m


class _NaNEmpty:
    """Stands in for the module's `torch`: torch.empty returns NaN-filled floating tensors."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def empty(*args, **kwargs):
        t = torch.empty(*args, **kwargs)
        if t.is_floating_point():
            t.fill_(float("nan"))
        return t


@pytest.fixture(autouse=True)
def _poison_workspace(monkeypatch):
    monkeypatch.setattr(_mod(), "torch", _NaNEmpty())


def _case(seed, lens, qlens, heads=6, slots=1 << 15):
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = sum(qlens)
    kb = (torch.randn(slots, 1, D, device=DEV, generator=g) * 0.5).to(torch.float8_e4m3fn)
    k_ext = (torch.randn(n, 1, D, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    return dict(
        q=(torch.randn(n, heads, D, device=DEV, generator=g) * 0.5).to(torch.bfloat16),
        k_ext=k_ext,
        v_ext=k_ext[..., :DV].contiguous(),
        kb=kb,
        vb=kb[..., :DV],
        kv_indices=torch.randperm(slots, device=DEV, generator=g)[: sum(lens)].to(torch.int64).contiguous(),
        kv_indptr=torch.tensor([0, *itertools.accumulate(lens)], dtype=torch.int32, device=DEV),
        qo_indptr=torch.tensor([0, *itertools.accumulate(qlens)], dtype=torch.int32, device=DEV),
        maxq=max(qlens),
    )


def _run(c, *, hpack, win=4095, k_scale=1.0, v_scale=1.0, o=None):
    if o is None:
        o = torch.full((c["q"].shape[0], c["q"].shape[1], DV), 7.0, dtype=torch.bfloat16, device=DEV)
    _mod().dflash_draft_splitkv_attention_fwd(
        c["q"], c["k_ext"], c["v_ext"], o, c["kb"], c["vb"], c["qo_indptr"], c["kv_indptr"], c["kv_indices"],
        c["maxq"], k_scale, v_scale, SM_SCALE, max_kv_len=MAX_KV, sliding_window_size=win, pdl=True,
        pdl_wait_first=True, hpack=hpack,
    )
    return o


def _assert_bitwise(c, **kw):
    ref = _run(c, hpack=False, **kw).clone()
    got = _run(c, hpack=True, **kw).clone()
    assert not torch.isnan(ref.float()).any()
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (got.float() - ref.float()).abs().max().item()


@pytest.mark.parametrize("bs", [2, 3, 4, 8, 16])
@pytest.mark.parametrize("qscale", [1.0, 4.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_full_rails_bitwise(bs, qscale, seed):
    c = _case(100 + bs + 1000 * seed, [MAX_KV] * bs, [8] * bs)
    c["q"] = (c["q"].float() * qscale).to(torch.bfloat16)
    _assert_bitwise(c)


@pytest.mark.parametrize("win", [4095, 1024, 100])
def test_ragged_bitwise(win):
    _assert_bitwise(_case(7, [MAX_KV, 700, 64, 1, 1000, 333], [8, 3, 8, 1, 8, 5]), win=win)


def test_kv_scales_bitwise():
    _assert_bitwise(_case(11, [MAX_KV, 900, 513, 64], [8] * 4), k_scale=0.5, v_scale=2.0)


@pytest.mark.parametrize("bs", [4, 16])
def test_twelve_heads_bitwise(bs):
    _assert_bitwise(_case(13, [MAX_KV] * bs, [8] * bs, heads=12))


def test_overlong_rail_nan_rows_match():
    c = _case(17, [MAX_KV, MAX_KV + 64, 64, 500], [8] * 4)
    ref = _run(c, hpack=False).clone()
    got = _run(c, hpack=True).clone()
    nan_rows = lambda t: [bool(torch.isnan(t[8 * b : 8 * (b + 1)].float()).all()) for b in range(4)]
    assert nan_rows(ref) == [False, True, False, False]
    assert nan_rows(got) == nan_rows(ref)
    keep = torch.ones(32, dtype=torch.bool, device=DEV)
    keep[8:16] = False
    assert torch.equal(got[keep].view(torch.int16), ref[keep].view(torch.int16))


def test_graph_replay_equals_eager():
    c = _case(19, [MAX_KV, 800, 300, 1], [8] * 4)
    eager = _run(c, hpack=True).clone()
    o = torch.empty_like(eager)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _run(c, hpack=True, o=o)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(c, hpack=True, o=o)
    o.fill_(7.0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(o.view(torch.int16), eager.view(torch.int16))


def test_batch_one_is_the_per_head_pass():
    _assert_bitwise(_case(23, [MAX_KV], [8]))
