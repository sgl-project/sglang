from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.io_struct import MMEmbeddingCacheAcquireReqInput
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import _can_omit_mm_features
from sglang.srt.mem_cache.multimodal_cache import (
    MM_EMBEDDING_CACHE_LEASE_ID_KEY,
    EmbeddingResult,
    MultiModalStaticCache,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _embedding(value: int) -> EmbeddingResult:
    return EmbeddingResult(embedding=torch.tensor([value], dtype=torch.int64))


def _scheduler(*, world_size: int = 1) -> Scheduler:
    scheduler = object.__new__(Scheduler)
    scheduler.ps = SimpleNamespace(pp_rank=0, dp_rank=2)
    scheduler.dp_tp_cpu_group = object() if world_size > 1 else None
    scheduler.dp_tp_group = SimpleNamespace(world_size=world_size)
    return scheduler


def _featureless_item(mm_hash: int, lease_id: str) -> MultimodalDataItem:
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=None,
        model_specific_data={MM_EMBEDDING_CACHE_LEASE_ID_KEY: lease_id},
    )
    item.hash = mm_hash
    return item


def test_acquire_requires_every_tp_rank_and_unpins_rejected_items():
    cache = MultiModalStaticCache(max_size=1024)
    cache.set(11, _embedding(11))
    cache.set(22, _embedding(22))
    scheduler = _scheduler(world_size=2)

    def reject_second_rank(tensor, **_kwargs):
        tensor[1] = 0

    request = MMEmbeddingCacheAcquireReqInput(
        rid="lease",
        feature_hashes=[11, 22],
        input_ids=[1, 2],
    )
    with (
        patch.object(mm_schedule, "embedding_cache", cache),
        patch("torch.distributed.all_reduce", side_effect=reject_second_rank),
    ):
        output = scheduler.acquire_mm_embedding_cache(request)

    assert output.hit_mask == [True, False]
    assert output.lease_id == "lease"
    assert output.routed_dp_rank == 2
    assert cache.lease_contains("lease", 11)
    assert not cache.lease_contains("lease", 22)


def test_scheduler_admits_then_request_release_drops_lease():
    cache = MultiModalStaticCache(max_size=1024)
    cache.set(11, _embedding(11))
    cache.acquire_many("lease", [11], ttl_s=300)
    scheduler = _scheduler()
    item = _featureless_item(11, "lease")
    request = SimpleNamespace(rid="request", mm_inputs=SimpleNamespace(mm_items=[item]))

    with patch.object(mm_schedule, "embedding_cache", cache):
        assert scheduler._validate_mm_embedding_leases(request) is None
        assert cache.lease_contains("lease", 11)
        MultimodalInputs(mm_items=[item]).release_features()

    assert not cache.lease_contains("lease", 11)


def test_chunk_lookup_keeps_lease_owned_by_request():
    cache = MultiModalStaticCache(max_size=1024)
    cache.set(11, _embedding(11))
    cache.acquire_many("lease", [11], ttl_s=300)
    cache.admit_lease("lease")
    item = _featureless_item(11, "lease")

    with patch.object(mm_schedule, "embedding_cache", cache):
        assert mm_schedule._get_cached_embedding(item).embedding.item() == 11
        assert mm_schedule._get_cached_embedding(item).embedding.item() == 11

    assert item.model_specific_data[MM_EMBEDDING_CACHE_LEASE_ID_KEY] == "lease"
    assert cache.lease_contains("lease", 11)


def test_missing_lease_requests_one_internal_cold_retry():
    cache = MultiModalStaticCache(max_size=1024)
    scheduler = _scheduler()
    request = SimpleNamespace(
        rid="request",
        mm_inputs=SimpleNamespace(mm_items=[_featureless_item(11, "lost")]),
    )

    with patch.object(mm_schedule, "embedding_cache", cache):
        output = scheduler._validate_mm_embedding_leases(request)

    assert output.rid == "request"
    assert output.lease_id == "lost"
    assert output.routed_dp_rank == 2


def test_non_owner_pipeline_stage_never_omits_features():
    scheduler = _scheduler()
    scheduler.ps.pp_rank = 1
    request = MMEmbeddingCacheAcquireReqInput(
        rid="lease",
        feature_hashes=[11],
        input_ids=[1],
    )

    output = scheduler.acquire_mm_embedding_cache(request)

    assert output.hit_mask == [False]
    assert output.lease_id is None


def test_continual_session_keeps_features_for_later_turns():
    regular = SimpleNamespace(parallel_sample_num=1, session_params=None)
    continual = SimpleNamespace(
        parallel_sample_num=1, session_params=SimpleNamespace(id="session")
    )

    assert _can_omit_mm_features(regular)
    assert not _can_omit_mm_features(continual)
