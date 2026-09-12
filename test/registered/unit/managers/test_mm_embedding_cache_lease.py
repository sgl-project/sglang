import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    MMEmbeddingCacheAcquireReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import (
    ReqState,
    TokenizerManager,
    _can_omit_mm_features,
    _namespace_mm_radix_cache,
)
from sglang.srt.mem_cache.multimodal_cache import (
    MM_EMBEDDING_CACHE_HASH_KEY,
    MM_EMBEDDING_CACHE_IDENTITY_KEY,
    MM_EMBEDDING_CACHE_LEASE_ID_KEY,
    EmbeddingResult,
    MultiModalStaticCache,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _embedding(value: int, identity: str = None) -> EmbeddingResult:
    return EmbeddingResult(
        embedding=torch.tensor([value], dtype=torch.int64), identity=identity
    )


def _scheduler(*, world_size: int = 1) -> Scheduler:
    scheduler = object.__new__(Scheduler)
    scheduler.ps = SimpleNamespace(pp_rank=0, dp_rank=2)
    scheduler.dp_tp_cpu_group = object() if world_size > 1 else None
    scheduler.dp_tp_group = SimpleNamespace(world_size=world_size)
    return scheduler


def _featureless_item(
    mm_hash: int, lease_id: str, identity: str = None
) -> MultimodalDataItem:
    metadata = {MM_EMBEDDING_CACHE_LEASE_ID_KEY: lease_id}
    if identity is not None:
        metadata[MM_EMBEDDING_CACHE_IDENTITY_KEY] = identity
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=None,
        model_specific_data=metadata,
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


def test_acquire_rejects_a_matching_64_bit_hash_with_a_different_identity():
    cache = MultiModalStaticCache(max_size=1024)
    cache.set(11, _embedding(11, identity="artifact-a"))
    scheduler = _scheduler()
    request = MMEmbeddingCacheAcquireReqInput(
        rid="lease",
        feature_hashes=[11],
        feature_identities=["artifact-b"],
        input_ids=[1],
    )

    with patch.object(mm_schedule, "embedding_cache", cache):
        output = scheduler.acquire_mm_embedding_cache(request)

    assert output.hit_mask == [False]
    assert output.lease_id is None


def test_identityless_acquire_rejects_a_matching_strong_identity():
    cache = MultiModalStaticCache(max_size=1024)
    cache.set(11, _embedding(11, identity="artifact-a"))

    assert cache.acquire_many("lease", [11], 10, [None]) == [False]


def test_scheduler_lookup_rejects_a_hash_collision_after_feature_materialization():
    cache = MultiModalStaticCache(max_size=1024)
    cache.set(11, _embedding(11, identity="artifact-a"))
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=torch.tensor([1]),
        model_specific_data={MM_EMBEDDING_CACHE_IDENTITY_KEY: "artifact-b"},
    )
    item.hash = 11

    with patch.object(mm_schedule, "embedding_cache", cache):
        assert mm_schedule._get_cached_embedding(item) is None


def test_admission_rejects_a_lease_with_a_different_strong_identity():
    cache = MultiModalStaticCache(max_size=1024)
    cache.set(11, _embedding(11, identity="artifact-a"))
    cache.acquire_many("lease", [11], ttl_s=300, identities=["artifact-a"])
    scheduler = _scheduler()
    item = _featureless_item(11, "lease", identity="artifact-b")
    request = SimpleNamespace(rid="request", mm_inputs=SimpleNamespace(mm_items=[item]))

    with patch.object(mm_schedule, "embedding_cache", cache):
        output = scheduler._validate_mm_embedding_leases(request)

    assert output.lease_id == "lease"
    assert not cache.lease_contains("lease", 11)


def test_caller_router_hash_cannot_redirect_an_embedding_lease():
    identity = "sha256:" + "01" * 32
    cache = MultiModalStaticCache(max_size=1024)
    cache.set(11, _embedding(11, identity=identity))
    cache.acquire_many("lease", [11], ttl_s=300, identities=[identity])
    scheduler = _scheduler()
    item = _featureless_item(11, "lease", identity=identity)
    item.model_specific_data[MM_EMBEDDING_CACHE_HASH_KEY] = 11

    # The legacy API may replace the router/pad key after lease acquisition.
    item.set_hash(22)
    request = SimpleNamespace(rid="request", mm_inputs=SimpleNamespace(mm_items=[item]))

    with patch.object(mm_schedule, "embedding_cache", cache):
        assert scheduler._validate_mm_embedding_leases(request) is None
        assert mm_schedule._get_cached_embedding(item).embedding.item() == 11

    assert item.hash == 22
    assert item.model_specific_data[MM_EMBEDDING_CACHE_HASH_KEY] == 11


def test_cache_replaces_an_unpinned_hash_collision_with_the_new_identity():
    cache = MultiModalStaticCache(max_size=1024)
    cache.set(11, _embedding(11, identity="artifact-a"))

    assert cache.set(11, _embedding(22, identity="artifact-b"))
    replacement = cache.get_single(11)
    assert replacement.identity == "artifact-b"
    assert replacement.embedding.item() == 22


def test_cache_replaces_a_strong_identity_with_an_identityless_entry():
    cache = MultiModalStaticCache(max_size=1024)
    cache.set(11, _embedding(11, identity="artifact-a"))

    assert cache.set(11, _embedding(22))
    replacement = cache.get_single(11)
    assert replacement.identity is None
    assert replacement.embedding.item() == 22


def test_radix_namespace_prevents_a_30_bit_pad_collision():
    first_hash = 11
    second_hash = first_hash + (1 << 30)
    first_identity = "sha256:" + "01" * 32
    second_identity = "sha256:" + "02" * 32
    first = _featureless_item(first_hash, "lease", identity=first_identity)
    second = _featureless_item(second_hash, "lease", identity=second_identity)

    first.set_hash(first_hash)
    second.set_hash(second_hash)
    assert first.pad_value == second.pad_value
    assert _namespace_mm_radix_cache(
        None, SimpleNamespace(mm_items=[first])
    ) != _namespace_mm_radix_cache(None, SimpleNamespace(mm_items=[second]))


def test_radix_namespace_fails_closed_on_partial_strong_identities():
    strong = _featureless_item(11, "lease", identity="sha256:" + "01" * 32)
    legacy = _featureless_item(22, "lease")

    with pytest.raises(ValueError, match="cover every media item"):
        _namespace_mm_radix_cache("caller", SimpleNamespace(mm_items=[strong, legacy]))

    assert (
        _namespace_mm_radix_cache("caller", SimpleNamespace(mm_items=[legacy]))
        == "caller"
    )


@pytest.mark.parametrize("request_type", [GenerateReqInput, EmbeddingReqInput])
@pytest.mark.parametrize("with_media", [False, True])
def test_tokenized_request_namespaces_only_generation(request_type, with_media):
    request = request_type(input_ids=[1, 2], rid="request")
    request.normalize_batch_and_arguments()
    if isinstance(request, GenerateReqInput):
        request.extra_key = "caller"
        request.bootstrap_room = 1
    mm_inputs = (
        MultimodalInputs(
            mm_items=[_featureless_item(11, "lease", identity="sha256:" + "01" * 32)]
        )
        if with_media
        else None
    )
    time_stats = APIServerReqTimeStats()
    manager = object.__new__(TokenizerManager)
    manager.preferred_sampling_params = {}
    manager.sampling_params_class = SamplingParams
    manager.tokenizer = None
    manager.model_config = SimpleNamespace(vocab_size=256)
    manager.rid_to_state = {
        request.rid: ReqState(
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=request,
            time_stats=time_stats,
        )
    }

    tokenized = manager._create_tokenized_object(
        request, None, request.input_ids, mm_inputs=mm_inputs
    )

    assert list(tokenized.input_ids) == request.input_ids
    assert tokenized.mm_inputs is mm_inputs
    assert tokenized.time_stats is time_stats
    assert time_stats.tokenize_finish_time > 0
    if isinstance(request, GenerateReqInput):
        assert isinstance(tokenized, TokenizedGenerateReqInput)
        assert tokenized.extra_key == _namespace_mm_radix_cache("caller", mm_inputs)
    else:
        assert isinstance(tokenized, TokenizedEmbeddingReqInput)
        assert tokenized.sampling_params.max_new_tokens == 0


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


def test_batch_subrequest_preserves_embedding_lease_mode():
    regular = GenerateReqInput(
        text=["first", "second"],
        image_data=[[b"first"], [b"second"]],
        sampling_params=[{}, {}],
    )
    regular.normalize_batch_and_arguments()

    assert regular[0].parallel_sample_num == 1
    assert _can_omit_mm_features(regular[0])

    parallel = GenerateReqInput(
        text="prompt",
        image_data=b"image",
        sampling_params={"n": 2},
    )
    parallel.normalize_batch_and_arguments()

    assert parallel[0].parallel_sample_num == 2
    assert not _can_omit_mm_features(parallel[0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
