# SPDX-License-Identifier: Apache-2.0
"""``SGLANG_SPEC_FUSED_RELAY``: FutureMap relay rows stashed in one scatter
launch and, for a single request, resolved as row views of the relay buffers.

* A single request whose future indices are its own ``req_pool_indices``
  resolves views of its relay rows (copies otherwise) with the right values.
* Switch on vs off, ``stash`` leaves the same relay buffers (fields a payload
  leaves out keep their rows) and ``_resolve_spec_extras`` hands the draft
  input the same values, dtypes, shapes and strides.

Under ``SGLANG_IS_IN_CI`` overlap_utils turns on its debug asserts, which turn
the view path off and invalidate the rows a resolve reads, so every test
switches them off and each arm uses a fresh FutureMap.
"""

import sys
from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers import overlap_utils
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

SLOTS = 11
HIDDEN = 96
DSA = 2048
FIELDS = [
    ("bonus_tokens", "output_tokens_buf"),
    ("topk_p", "topk_p_buf"),
    ("topk_index", "topk_index_buf"),
    ("hidden_states", "hidden_states_buf"),
    ("draft_probs", "draft_probs_buf"),
    ("dsa_topk_indices", "dsa_topk_indices_buf"),
]


@pytest.fixture(autouse=True)
def _no_debug_asserts(monkeypatch):
    monkeypatch.setattr(overlap_utils, "_DEBUG_ASSERT", False)


def _future_map(topk, hidden, dsa, draft_probs, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    f = object.__new__(FutureMap)
    f.spec_algo = SimpleNamespace(
        is_ngram=lambda: False, is_some=lambda: True, need_topk=lambda: topk
    )
    f.device, f.req_pool_size = "cuda", SLOTS
    f.need_topk, f.need_hidden_states = topk, hidden
    f.output_tokens_buf = torch.randint(0, 1000, (SLOTS,), device="cuda", generator=g)
    f.topk_p_buf = torch.rand(SLOTS, 1, device="cuda", generator=g)
    f.topk_index_buf = torch.randint(0, 99, (SLOTS, 1), device="cuda", generator=g)
    f.hidden_states_buf = torch.randn(SLOTS, HIDDEN, device="cuda", generator=g)
    f.hidden_states_buf = f.hidden_states_buf.bfloat16()
    f.draft_probs_buf = (
        torch.rand(SLOTS, 33, device="cuda", generator=g) if draft_probs else None
    )
    f.dsa_topk_indices_buf = (
        torch.randint(-1, 1 << 20, (SLOTS, DSA), device="cuda", generator=g).int()
        if dsa
        else None
    )
    return f


def _batch(rows, dsa, same):
    reqs = []
    for i, row in enumerate(rows.tolist()):
        req = Req(
            rid=f"r{i}",
            origin_input_text="",
            origin_input_ids=array("q", [1]),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=4),
        )
        req.kv.req_pool_idx = row
        reqs.append(req)
    draft_input = SimpleNamespace(
        future_indices=rows if same else rows.clone(),
        draft_probs=None,
        hidden_states=None,
        future_dsa_topk_indices_available=dsa,
    )
    return SimpleNamespace(spec_info=draft_input, req_pool_indices=rows, reqs=reqs)


def _same(a, b):
    assert (a.dtype, a.shape, a.stride()) == (b.dtype, b.shape, b.stride())
    assert torch.equal(a, b)


@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("topk", [False, True])
@pytest.mark.parametrize("dsa", [False, True])
@pytest.mark.parametrize("same", [False, True])
def test_single_request_views(hidden, topk, dsa, same):
    f = _future_map(topk, hidden, dsa, draft_probs=topk)
    batch = _batch(torch.tensor([3], device="cuda"), dsa, same)
    d = batch.spec_info
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), envs.SGLANG_SPEC_FUSED_RELAY.override(True):
        f._resolve_spec_extras(batch)
        fields = [(k, b) for k, b in FIELDS if getattr(d, k, None) is not None]
        assert len(fields) == 1 + 3 * topk + hidden + dsa
        for k, b in fields:
            assert (getattr(d, k).data_ptr() == getattr(f, b)[3:4].data_ptr()) == same
        copies = [getattr(d, k).clone() for k, _ in fields]
        references = [getattr(f, b)[3:4].clone() for _, b in fields]
        for _, b in fields:
            getattr(f, b).fill_(-1)
    stream.synchronize()
    for x, y in zip(copies, references):
        assert torch.equal(x, y)


@pytest.mark.parametrize("bs", [1, 8])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("staging", [False, True])
def test_stash_matches_stock(bs, full, staging, monkeypatch):
    # Lazy buffer init asks the speculative config whether hidden states relay.
    monkeypatch.setattr(
        "sglang.srt.speculative.spec_utils.spec_need_hidden_states", lambda: full
    )
    g = torch.Generator(device="cuda").manual_seed(7)
    extra = full and not staging  # staging transitions carry the bonus token only
    payload = RelayPayload(
        bonus_tokens=torch.randint(0, 1000, (bs,), device="cuda", generator=g).int(),
        topk_p=torch.rand(bs, 1, device="cuda", generator=g) if extra else None,
        topk_index=(
            torch.randint(0, 99, (bs, 1), device="cuda", generator=g) if extra else None
        ),
        hidden_states=(
            torch.randn(bs, HIDDEN, device="cuda", generator=g).bfloat16()
            if extra
            else None
        ),
        draft_probs=torch.rand(bs, 33, device="cuda", generator=g) if extra else None,
        dsa_topk_indices=(
            torch.randint(0, 1 << 20, (bs, DSA), device="cuda", generator=g).int()
            if extra
            else None
        ),
    )
    indices = torch.randperm(SLOTS, device="cuda")[:bs]
    buffers = []
    for on in (False, True):
        f = _future_map(full, full, full, full, seed=bs)
        with envs.SGLANG_SPEC_FUSED_RELAY.override(on):
            f.stash(indices, payload)
        buffers.append(
            {b: getattr(f, b) for _, b in FIELDS if getattr(f, b) is not None}
        )
    torch.cuda.synchronize()
    assert buffers[0].keys() == buffers[1].keys()
    for b in buffers[0]:
        _same(buffers[0][b], buffers[1][b])


@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("same", [False, True])
def test_resolve_matches_stock(bs, same):
    rows = torch.randperm(SLOTS, device="cuda")[:bs]
    resolved, views = [], None
    for on in (False, True):
        f = _future_map(True, True, True, True, seed=3)
        batch = _batch(rows, True, same)
        with envs.SGLANG_SPEC_FUSED_RELAY.override(on):
            f._resolve_spec_extras(batch)
        resolved.append({k: getattr(batch.spec_info, k) for k, _ in FIELDS})
        if on:
            views = [
                v.untyped_storage().data_ptr()
                == getattr(f, b).untyped_storage().data_ptr()
                for (k, b), v in zip(FIELDS, resolved[-1].values())
            ]
    torch.cuda.synchronize()
    for k, _ in FIELDS:
        _same(resolved[0][k], resolved[1][k])
    assert views == [bs == 1 and same] * len(FIELDS)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
