from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang.srt.speculative.eagle_utils as eagle
from sglang.srt.managers.schedule_batch import ReqKvInfo


@pytest.mark.parametrize("bs", [0, 1, 3, 128])
@pytest.mark.parametrize("needed", [False, True])
@pytest.mark.parametrize("page", [1, 64])
def test_skip_allocation(monkeypatch, bs, needed, page):
    reqs = []
    for i in range(bs):
        kv = ReqKvInfo()
        kv.kv_allocated_len = 128
        kv.kv_committed_len = 125 if needed and i == 0 else 112
        reqs.append(SimpleNamespace(kv=kv, decode_batch_idx=0))
    batch = SimpleNamespace(
        reqs=reqs,
        batch_size=lambda: bs,
        maybe_evict_swa=Mock(),
        cumulate_penalty_output_tokens=Mock(),
        sampling_info=SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=True)
        ),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=page),
        device="cpu",
        tree_cache=object(),
        req_to_token_pool=object(),
        req_pool_indices=torch.arange(bs),
    )
    monkeypatch.setattr(eagle, "get_alloc_reserve_per_decode", lambda: 8)
    monkeypatch.setattr(
        eagle, "get_spec", lambda: SimpleNamespace(speculative_eagle_topk=1)
    )
    alloc = Mock()
    monkeypatch.setattr(eagle, "alloc_for_spec_decode", alloc)
    tensor = Mock(wraps=torch.tensor)
    monkeypatch.setattr(eagle.torch, "tensor", tensor)
    eagle.eagle_prepare_for_decode(batch)
    assert all(r.decode_batch_idx == 1 for r in reqs)
    batch.maybe_evict_swa.assert_called_once()
    batch.cumulate_penalty_output_tokens.assert_called_once()
    if needed and bs:
        assert tensor.call_count == 2
        assert alloc.call_count == 1
        expected = ((133 + page - 1) // page) * page - 128
        assert alloc.call_args.kwargs["num_needed_tokens"] == expected
    else:
        assert tensor.call_count == 0
        alloc.assert_not_called()


def test_preserve_capacity_assertion(monkeypatch):
    kv = ReqKvInfo()
    kv.kv_allocated_len = 256
    kv.kv_committed_len = 100
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(kv=kv, decode_batch_idx=0)],
        batch_size=lambda: 1,
        maybe_evict_swa=Mock(),
        sampling_info=SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=False)
        ),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=64),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.empty(1, 128)),
    )
    monkeypatch.setattr(eagle, "get_alloc_reserve_per_decode", lambda: 8)
    monkeypatch.setattr(
        eagle, "get_spec", lambda: SimpleNamespace(speculative_eagle_topk=4)
    )
    with pytest.raises(AssertionError, match="exceeds req_to_token"):
        eagle.eagle_prepare_for_decode(batch)
