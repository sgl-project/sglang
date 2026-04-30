import types

from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class _FakeTreeCache:
    def __init__(self):
        self.unfinished_reqs = []

    def cache_unfinished_req(self, req):
        self.unfinished_reqs.append(req)


class _FakeScheduler(SchedulerOutputProcessorMixin):
    def __init__(self):
        self.tree_cache = _FakeTreeCache()


def test_ngram_prefill_does_not_publish_unfinished_req_to_radix_cache():
    scheduler = _FakeScheduler()
    batch = types.SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.NGRAM)
    req = object()

    scheduler._cache_unfinished_req_after_prefill(batch, req)

    assert scheduler.tree_cache.unfinished_reqs == []


def test_non_ngram_prefill_publishes_unfinished_req_to_radix_cache():
    scheduler = _FakeScheduler()
    batch = types.SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.NONE)
    req = object()

    scheduler._cache_unfinished_req_after_prefill(batch, req)

    assert scheduler.tree_cache.unfinished_reqs == [req]


def test_missing_spec_algorithm_keeps_previous_cache_behavior():
    scheduler = _FakeScheduler()
    batch = types.SimpleNamespace(spec_algorithm=None)
    req = object()

    scheduler._cache_unfinished_req_after_prefill(batch, req)

    assert scheduler.tree_cache.unfinished_reqs == [req]
