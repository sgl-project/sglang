import asyncio
import pickle
import sys
import threading
import time
from array import array
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import torch
import zmq
import zmq.asyncio
from fastapi import HTTPException
from PIL import Image

from sglang.srt.disaggregation.encode_receiver import (
    EmbeddingData,
    MMReceiverHTTP,
    MultiModalEmbeddingData,
    _encoder_media_item,
    _select_mm_processor_prompt,
)
from sglang.srt.disaggregation.encode_server import (
    InternalError,
    MMEncoder,
    _get_mm_grid_dim,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.managers.tokenizer_manager import (
    _reject_missing_dispatched_encoder_embedding,
)
from sglang.srt.mem_cache.multimodal_cache import (
    EmbeddingResult,
    MultiModalStaticCache,
)
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
from sglang.srt.multimodal.cache import (
    EncoderMediaLookup,
    EncoderPreprocessArtifact,
    MultimodalPreprocessCache,
    snapshot_media,
)
from sglang.srt.multimodal.encoder_preprocessing import (
    LOCAL_PREPROCESSED_KEY,
    EncoderMediaProcessorConfig,
    EncoderPreprocessOutput,
    get_encoder_preprocessed_items,
    hash_raw_encoder_item,
    invoke_encoder_preprocessor,
)
from sglang.srt.multimodal.kimi_k3_image_processing import (
    DEFERRED_PREPROCESSING_KEY,
    materialize_kimi_k3_cpu_features,
    prepare_kimi_k3_encoder_inputs,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import resolve_encoder_transfer_backend
from sglang.srt.utils import ImageData
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_kimi_k3_encoder_transfer_backend_auto_avoids_tp_fanout():
    assert (
        resolve_encoder_transfer_backend("auto", "KimiK3ForConditionalGeneration", 8)
        == "zmq_to_tokenizer"
    )
    assert (
        resolve_encoder_transfer_backend("auto", "KimiK3ForConditionalGeneration", 1)
        == "zmq_to_scheduler"
    )
    assert (
        resolve_encoder_transfer_backend("auto", "Qwen3VLForConditionalGeneration", 8)
        == "zmq_to_scheduler"
    )
    assert (
        resolve_encoder_transfer_backend(
            "zmq_to_scheduler", "KimiK3ForConditionalGeneration", 8
        )
        == "zmq_to_scheduler"
    )
    assert (
        resolve_encoder_transfer_backend(
            "mooncake", "KimiK3ForConditionalGeneration", 8
        )
        == "mooncake"
    )


def test_epd_language_only_rejects_missing_dispatched_embedding():
    server_args = SimpleNamespace(
        language_only=True,
        encoder_transfer_backend="zmq_to_tokenizer",
    )
    request = SimpleNamespace(need_wait_for_mm_inputs=True)

    with pytest.raises(HTTPException) as exc_info:
        _reject_missing_dispatched_encoder_embedding(server_args, request, None)

    assert getattr(exc_info.value, "status_code", None) == 503


def test_epd_allows_local_processing_when_request_was_not_dispatched():
    server_args = SimpleNamespace(
        language_only=True,
        encoder_transfer_backend="zmq_to_tokenizer",
    )
    request = SimpleNamespace(need_wait_for_mm_inputs=False)

    _reject_missing_dispatched_encoder_embedding(server_args, request, None)


def _encoder(model_type="kimi_k3"):
    encoder = MMEncoder.__new__(MMEncoder)
    encoder.model_type = model_type
    encoder.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            vision_config=SimpleNamespace(merge_kernel_size=(2, 2))
        )
    )
    encoder.encoder_media_processor_config = (
        KimiK3ForConditionalGeneration.encoder_media_processor_config
        if model_type == "kimi_k3"
        else EncoderMediaProcessorConfig()
    )
    encoder.encoder_artifact_cache_configs = (
        encoder.encoder_media_processor_config.artifact_caches
    )
    return encoder


def test_kimi_k3_encoder_normalizes_pillow_images_to_media_dicts():
    image = Image.new("RGB", (2, 2))
    encoder = _encoder()

    assert encoder._grid_count_per_leaf(
        [image, {"type": "image", "image": [image, image]}], Modality.IMAGE
    ) == [1, 2]

    normalized = encoder._normalize_kimi_encoder_images(
        [image, {"type": "image", "image": [image, image]}]
    )
    assert len(normalized) == 3
    assert all(item["type"] == "image" for item in normalized)
    assert all(item["image"] is image for item in normalized)


def test_kimi_k3_encoder_passes_media_dicts_to_image_processor():
    image = Image.new("RGB", (3, 2))
    processor_calls = []

    def image_processor(*, images, **kwargs):
        processor_calls.append((images, kwargs))
        return {"pixel_values": torch.ones(1, 3), "grid_thws": [[1, 1, 1]]}

    encoder = _encoder()
    encoder.image_processor = image_processor
    encoder.vision_config = {"image": {"return_tensors": "pt"}}
    encoder._flatten_and_load_images = AsyncMock(return_value=[image])
    encoder.preproc_executor = ThreadPoolExecutor(max_workers=1)
    try:
        output = asyncio.run(encoder._process_image_items([image], None))
    finally:
        encoder.preproc_executor.shutdown()

    assert "pixel_values" in output
    assert output["original_image_sizes"] == [[3, 2]]
    assert len(processor_calls) == 1
    images, kwargs = processor_calls[0]
    assert images[0]["type"] == "image"
    assert images[0]["image"] is image
    assert kwargs == {"return_tensors": "pt"}


def _kimi_k3_image_processor():
    return SimpleNamespace(
        media_proc_cfg={
            "patch_size": 2,
            "merge_kernel_size": 2,
            "in_patch_limit": 1024,
            "patch_limit_on_one_side": 64,
            "fixed_output_tokens": None,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "transparent_bg_config": {"type": "white"},
        }
    )


def test_kimi_k3_epd_preprocess_preserves_raw_per_image_items():
    first = Image.new("RGB", (8, 6), color=(1, 2, 3))
    second = Image.new("RGB", (5, 9), color=(4, 5, 6))

    output = prepare_kimi_k3_encoder_inputs(
        [
            {"type": "image", "image": first},
            {"type": "image", "image": second},
        ],
        _kimi_k3_image_processor(),
    )

    items = get_encoder_preprocessed_items(output)
    assert isinstance(output, EncoderPreprocessOutput)
    assert len(items) == 2
    assert output["original_image_sizes"] == [[8, 6], [5, 9]]
    assert output["grid_thws"].tolist() == [[1, 4, 4], [1, 6, 4]]
    for item, image in zip(items, (first, second)):
        assert item.modality == Modality.IMAGE
        assert item.feature is image
        assert item.hash is not None
        assert item.pad_value is not None
        deferred = item.model_specific_data[DEFERRED_PREPROCESSING_KEY]
        assert deferred["image_mean"] == [0.5, 0.5, 0.5]
        assert deferred["image_std"] == [0.5, 0.5, 0.5]


def test_kimi_k3_epd_preserves_verified_content_identity():
    image = Image.new("RGB", (8, 6), color=(1, 2, 3))
    digest = "sha256:" + "ab" * 32

    output = prepare_kimi_k3_encoder_inputs(
        [{"type": "image", "image": image, "content_hash": digest}],
        _kimi_k3_image_processor(),
    )

    item = get_encoder_preprocessed_items(output)[0]
    assert item.model_specific_data["content_digest"] == digest


def test_kimi_k3_epd_model_preprocessor_receives_image_processor():
    image = Image.new("RGB", (8, 6), color=(1, 2, 3))
    image_processor = _kimi_k3_image_processor()
    image_processor.preprocess = lambda medias, return_tensors: {
        "pixel_values": torch.zeros(16, 12),
        "grid_thws": torch.tensor([[1, 4, 4]]),
    }
    calls = []

    def model_preprocessor(
        mm_data,
        modality,
        config,
        *,
        image_processor=None,
        use_gpu_preprocessing=False,
    ):
        calls.append(
            (mm_data, modality, config, image_processor, use_gpu_preprocessing)
        )
        return prepare_kimi_k3_encoder_inputs(mm_data, image_processor)

    encoder = _encoder()
    encoder.image_processor = image_processor
    encoder.use_image_processor_gpu = False
    encoder.vision_config = {"image": {"return_tensors": "pt"}}
    encoder._flatten_and_load_images = AsyncMock(return_value=[image])
    encoder.preproc_executor = ThreadPoolExecutor(max_workers=1)
    try:
        with patch(
            "sglang.srt.disaggregation.encode_server.get_parallel",
            return_value=SimpleNamespace(attn_tp_rank=0, attn_tp_size=1),
        ):
            output = asyncio.run(
                encoder._process_image_items([image], model_preprocessor)
            )
    finally:
        encoder.preproc_executor.shutdown()

    assert len(calls) == 1
    assert calls[0][0][0] == {"type": "image", "image": image}
    assert calls[0][1:] == (
        Modality.IMAGE,
        encoder.vision_config,
        image_processor,
        False,
    )
    assert len(get_encoder_preprocessed_items(output)) == 1


def test_encoder_preprocessor_context_keeps_legacy_hooks_compatible():
    calls = []

    def legacy_hook(mm_data, modality, config):
        calls.append((mm_data, modality, config))
        return {"ok": True}

    result = invoke_encoder_preprocessor(
        legacy_hook,
        ["image"],
        Modality.IMAGE,
        {"image": {}},
        image_processor=object(),
        use_gpu_preprocessing=True,
    )

    assert result == {"ok": True}
    assert calls == [(["image"], Modality.IMAGE, {"image": {}})]


def test_kimi_k3_epd_default_cpu_materialization_is_owner_only_and_exact():
    class RecordingImageProcessor:
        media_proc_cfg = _kimi_k3_image_processor().media_proc_cfg

        def __init__(self):
            self.calls = []

        def preprocess(self, medias, return_tensors):
            self.calls.append(medias)
            features = [
                torch.full((4, 3, 2, 2), media["image"].getpixel((0, 0))[0])
                for media in medias
            ]
            grids = torch.tensor([[1, 2, 2]] * len(medias))
            return {"pixel_values": torch.cat(features), "grid_thws": grids}

    processor = RecordingImageProcessor()
    images = [Image.new("RGB", (4, 4), color=(value, 0, 0)) for value in (7, 11)]
    output = prepare_kimi_k3_encoder_inputs(images, processor)
    items = get_encoder_preprocessed_items(output)

    materialized = materialize_kimi_k3_cpu_features([items[1]], processor)

    assert len(processor.calls) == 1
    assert len(processor.calls[0]) == 1
    assert processor.calls[0][0]["image"].getpixel((0, 0)) == (11, 0, 0)
    assert torch.all(materialized == 11)
    assert items[0].model_specific_data[DEFERRED_PREPROCESSING_KEY]["backend"] == "cpu"


def test_encoder_preprocess_materializes_only_local_size_balanced_items():
    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.tensor([value], dtype=torch.uint8),
        )
        for value in (3, 5, 7)
    ]
    calls = []

    def materialize(selected):
        calls.append(selected)
        return [item.feature.float() + 10 for item in selected]

    output = EncoderPreprocessOutput(
        {"pixel_values": [item.feature for item in items]},
        mm_items=items,
        item_sizes=[8, 5, 3],
        materialize_local_items=materialize,
    )

    output.materialize_for_rank(rank=1, world_size=2)

    assert calls == [[items[1], items[2]]]
    assert items[0].feature.tolist() == [3]
    assert items[1].feature.tolist() == [15.0]
    assert items[2].feature.tolist() == [17.0]
    assert LOCAL_PREPROCESSED_KEY not in items[0].model_specific_data
    assert items[1].model_specific_data[LOCAL_PREPROCESSED_KEY]
    assert items[2].model_specific_data[LOCAL_PREPROCESSED_KEY]


def test_raw_encoder_hash_includes_shape_and_dtype():
    flat = torch.arange(12, dtype=torch.uint8)

    assert hash_raw_encoder_item(flat.reshape(2, 2, 3)) != hash_raw_encoder_item(
        flat.reshape(3, 2, 2)
    )
    assert hash_raw_encoder_item(flat) != hash_raw_encoder_item(flat.to(torch.int16))


@pytest.mark.parametrize(
    ("use_image_processor_gpu", "expected_decode_mode"),
    [(False, False), (True, "nvjpeg_fancy")],
)
def test_kimi_k3_epd_selects_matching_jpeg_decode_mode(
    use_image_processor_gpu, expected_decode_mode
):
    expected = torch.zeros((3, 2, 3), dtype=torch.uint8)
    encoder = _encoder()
    encoder.use_image_processor_gpu = use_image_processor_gpu

    with patch(
        "sglang.srt.disaggregation.encode_server.load_image",
        return_value=(expected, None),
    ) as load:
        output = encoder._load_single_item(b"jpeg", Modality.IMAGE)

    assert output is expected
    load.assert_called_once_with(b"jpeg", expected_decode_mode)


def test_kimi_k3_epd_verifies_content_hash_before_decode():
    payload = b"jpeg"
    digest = snapshot_media(payload).content_digest
    expected = torch.zeros((3, 2, 3), dtype=torch.uint8)
    encoder = _encoder()
    encoder.use_image_processor_gpu = False

    with patch(
        "sglang.srt.disaggregation.encode_server.load_image",
        return_value=(expected, None),
    ) as load:
        output = encoder._load_single_item(
            {"url": payload, "content_hash": digest}, Modality.IMAGE
        )

    assert output == {
        "type": "image",
        "image": expected,
        "content_hash": digest,
    }
    load.assert_called_once_with(payload, False)


def _cache_enabled_encoder(*, trust_content_hashes=False):
    encoder = _encoder()
    encoder.mm_processor_fingerprint = "processor"
    encoder.mm_preprocess_cache = MultimodalPreprocessCache(max_size_bytes=1024 * 1024)
    encoder.mm_cache = MultiModalStaticCache(max_size=1024 * 1024)
    encoder.mm_cache_lock = asyncio.Lock()
    encoder.encoder_artifact_inflight = {}
    encoder.encoder_artifact_inflight_lock = asyncio.Lock()
    encoder.encoder_artifact_cache_generation = 0
    encoder.io_executor = ThreadPoolExecutor(max_workers=2)
    encoder.trust_mm_content_hashes = trust_content_hashes
    encoder.profiler = None
    return encoder


def test_kimi_k3_epd_trusted_metadata_hit_skips_media_read():
    payload = b"jpeg"
    digest = snapshot_media(payload).content_digest
    encoder = _cache_enabled_encoder(trust_content_hashes=True)
    try:
        key = encoder._encoder_artifact_key(digest, {"url": payload}, Modality.IMAGE)
        artifact = EncoderPreprocessArtifact(
            content_digest=digest,
            artifact_key=key,
            feature_identity="sha256:" + "11" * 32,
            feature_hash=17,
            grid_thw=(1, 2, 2),
            metadata={"original_image_sizes": (8, 6)},
        )
        encoder.mm_preprocess_cache.put(key, artifact)

        with patch(
            "sglang.srt.disaggregation.encode_server.snapshot_media",
            side_effect=AssertionError("trusted metadata hit read media"),
        ):
            lookups = asyncio.run(
                encoder._lookup_encoder_artifacts(
                    [{"url": payload, "content_hash": digest}], Modality.IMAGE
                )
            )

        assert lookups[0].snapshot is None
        assert lookups[0].artifact is artifact
    finally:
        encoder.io_executor.shutdown()


def test_kimi_k3_epd_embedding_hit_skips_decode_processor_and_vit():
    payload = b"jpeg"
    digest = snapshot_media(payload).content_digest
    encoder = _cache_enabled_encoder()
    try:
        key = encoder._encoder_artifact_key(digest, payload, Modality.IMAGE)
        artifact = EncoderPreprocessArtifact(
            content_digest=digest,
            artifact_key=key,
            feature_identity="sha256:" + "22" * 32,
            feature_hash=23,
            grid_thw=(1, 2, 2),
            metadata={"original_image_sizes": (8, 6)},
        )
        embedding = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        encoder.mm_preprocess_cache.put(key, artifact)
        encoder.mm_cache.set(
            23, EmbeddingResult(embedding=embedding, identity=artifact.feature_identity)
        )
        encoder._process_mm_items = AsyncMock(
            side_effect=AssertionError("embedding hit ran processor")
        )

        grid, output, aux = asyncio.run(
            encoder._encode_with_preprocess_cache(
                [payload], Modality.IMAGE, log_metrics=False
            )
        )

        torch.testing.assert_close(grid, torch.tensor([[1, 2, 2]]))
        torch.testing.assert_close(output, embedding)
        assert aux == {
            "original_image_sizes": [[8, 6]],
            "mm_feature_identities": [artifact.feature_identity],
        }
        encoder._process_mm_items.assert_not_awaited()
    finally:
        encoder.io_executor.shutdown()


def test_kimi_k3_epd_coalesces_concurrent_encoder_misses():
    async def run():
        encoder = _cache_enabled_encoder()
        digest = "sha256:" + "12" * 32
        artifact = EncoderPreprocessArtifact(
            content_digest=digest,
            artifact_key="shared-key",
            feature_identity="sha256:" + "33" * 32,
            feature_hash=29,
            grid_thw=(1, 2, 2),
            metadata={"original_image_sizes": (8, 6)},
        )
        lookup = EncoderMediaLookup(
            "image", None, digest, artifact.artifact_key, None, None
        )
        embedding = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        owner_started = asyncio.Event()
        release_owner = asyncio.Event()
        owner_calls = 0

        encoder._lookup_encoder_artifacts = AsyncMock(return_value=[lookup])

        async def encode_owner(*args):
            nonlocal owner_calls
            owner_calls += 1
            owner_started.set()
            await release_owner.wait()
            return {artifact.artifact_key: (artifact, embedding)}

        encoder._encode_preprocess_cache_owners = encode_owner
        try:
            first = asyncio.create_task(
                encoder._encode_with_preprocess_cache(
                    ["image"], Modality.IMAGE, log_metrics=False
                )
            )
            await owner_started.wait()
            second = asyncio.create_task(
                encoder._encode_with_preprocess_cache(
                    ["image"], Modality.IMAGE, log_metrics=False
                )
            )
            await asyncio.sleep(0)
            release_owner.set()
            outputs = await asyncio.gather(first, second)

            assert owner_calls == 1
            assert encoder.mm_preprocess_cache.stats()["singleflight_joins"] == 1
            for grid, output, aux in outputs:
                torch.testing.assert_close(grid, torch.tensor([[1, 2, 2]]))
                torch.testing.assert_close(output, embedding)
                assert aux == {
                    "original_image_sizes": [[8, 6]],
                    "mm_feature_identities": [artifact.feature_identity],
                }
        finally:
            encoder.io_executor.shutdown()

    asyncio.run(run())


def test_kimi_k3_epd_flush_invalidates_inflight_encoder_work():
    async def run():
        encoder = _cache_enabled_encoder()
        digest = "sha256:" + "34" * 32
        artifact = EncoderPreprocessArtifact(
            content_digest=digest,
            artifact_key="inflight-key",
            feature_identity="sha256:" + "44" * 32,
            feature_hash=31,
            grid_thw=(1, 2, 2),
            metadata={"original_image_sizes": (8, 6)},
        )
        lookup = EncoderMediaLookup(
            "image", None, digest, artifact.artifact_key, None, None
        )
        embedding = torch.ones((1, 4))
        owner_started = asyncio.Event()
        release_owner = asyncio.Event()
        encoder._lookup_encoder_artifacts = AsyncMock(return_value=[lookup])

        async def encode_owner(*args):
            owner_started.set()
            await release_owner.wait()
            return {artifact.artifact_key: (artifact, embedding)}

        encoder._encode_preprocess_cache_owners = encode_owner
        try:
            task = asyncio.create_task(
                encoder._encode_with_preprocess_cache(
                    ["image"], Modality.IMAGE, log_metrics=False
                )
            )
            await owner_started.wait()
            encoder.clear_mm_caches()
            release_owner.set()
            with pytest.raises(InternalError, match="flushed during preprocessing"):
                await task
            assert not encoder.encoder_artifact_inflight
            assert len(encoder.mm_cache) == 0
        finally:
            encoder.io_executor.shutdown()

    asyncio.run(run())


def test_kimi_k3_epd_global_cache_preprocesses_only_metadata_misses():
    encoder = _cache_enabled_encoder()
    encoder.rank = 0
    hit = EncoderPreprocessArtifact(
        content_digest="sha256:" + "11" * 32,
        artifact_key="hit-key",
        feature_identity="sha256:" + "55" * 32,
        feature_hash=11,
        grid_thw=(1, 2, 2),
        metadata={"original_image_sizes": (8, 6)},
    )
    miss = EncoderPreprocessArtifact(
        content_digest="sha256:" + "22" * 32,
        artifact_key="miss-key",
        feature_identity="sha256:" + "66" * 32,
        feature_hash=22,
        grid_thw=(1, 4, 4),
        metadata={"original_image_sizes": (16, 12)},
    )
    lookups = [
        EncoderMediaLookup("hit", None, hit.content_digest, "hit-key", None, hit),
        EncoderMediaLookup("miss", None, miss.content_digest, "miss-key", None, None),
    ]
    mm_inputs = {
        "pixel_values": torch.ones(16, 3),
        "grid_thws": torch.tensor([[1, 4, 4]]),
        "original_image_sizes": [[16, 12]],
    }
    feature_fn = object()
    encoder._lookup_encoder_artifacts = AsyncMock(return_value=lookups)
    encoder._materialize_encoder_media = AsyncMock(return_value=["miss-media"])
    encoder._process_mm_items = AsyncMock(return_value=(mm_inputs, feature_fn))
    encoder._store_encoder_artifacts = lambda *args, **kwargs: [miss]
    try:
        ctx = asyncio.run(
            encoder._prepare_global_cache_context(
                ["hit-source", "miss-source"], Modality.IMAGE, "request"
            )
        )
    finally:
        encoder.io_executor.shutdown()

    encoder._materialize_encoder_media.assert_awaited_once_with(
        lookups, [1], Modality.IMAGE
    )
    encoder._process_mm_items.assert_awaited_once_with(["miss-media"], Modality.IMAGE)
    assert ctx.prepared_index_map == {1: 0}
    assert ctx.str_mm_hashes == [hit.feature_identity, miss.feature_identity]
    assert ctx.grid_thw.tolist() == [[1, 2, 2], [1, 4, 4]]
    assert ctx.aux_data == {
        "original_image_sizes": [[8, 6], [16, 12]],
        "mm_feature_identities": [hit.feature_identity, miss.feature_identity],
    }


def test_kimi_k3_epd_global_cache_coalesces_concurrent_metadata_misses():
    async def run():
        encoder = _cache_enabled_encoder()
        encoder.rank = 0
        digest = "sha256:" + "56" * 32
        artifact = EncoderPreprocessArtifact(
            content_digest=digest,
            artifact_key="global-key",
            feature_identity="sha256:" + "77" * 32,
            feature_hash=41,
            grid_thw=(1, 2, 2),
            metadata={"original_image_sizes": (8, 6)},
        )
        lookup = EncoderMediaLookup(
            "image", None, digest, artifact.artifact_key, None, None
        )
        mm_inputs = {
            "pixel_values": torch.ones(4, 3),
            "grid_thws": torch.tensor([[1, 2, 2]]),
            "original_image_sizes": [[8, 6]],
        }
        owner_started = asyncio.Event()
        release_owner = asyncio.Event()
        process_calls = 0
        encoder._lookup_encoder_artifacts = AsyncMock(
            side_effect=lambda items, modality: [lookup]
        )
        encoder._materialize_encoder_media = AsyncMock(return_value=["image"])

        async def process(*args, **kwargs):
            nonlocal process_calls
            process_calls += 1
            owner_started.set()
            await release_owner.wait()
            return mm_inputs, object()

        encoder._process_mm_items = process
        encoder._store_encoder_artifacts = lambda *args, **kwargs: [artifact]
        try:
            first = asyncio.create_task(
                encoder._prepare_global_cache_context(
                    ["image"], Modality.IMAGE, "first"
                )
            )
            await owner_started.wait()
            second = asyncio.create_task(
                encoder._prepare_global_cache_context(
                    ["image"], Modality.IMAGE, "second"
                )
            )
            await asyncio.sleep(0)
            release_owner.set()
            first_ctx, second_ctx = await asyncio.gather(first, second)

            assert process_calls == 1
            assert (
                first_ctx.str_mm_hashes
                == second_ctx.str_mm_hashes
                == [artifact.feature_identity]
            )
            assert first_ctx.prepared_index_map == {0: 0}
            assert second_ctx.prepared_index_map is None
            assert encoder.mm_preprocess_cache.stats()["singleflight_joins"] == 1
        finally:
            encoder.io_executor.shutdown()

    asyncio.run(run())


def test_epd_receiver_keeps_content_hash_aligned_with_image():
    digest = "sha256:" + "cd" * 32
    receiver = MMReceiverHTTP.__new__(MMReceiverHTTP)
    request = SimpleNamespace(
        image_data=[
            ImageData(
                url="image",
                detail="high",
                max_dynamic_patch=12,
                preprocess_kwargs={"crop": False},
                content_hash=digest,
            )
        ],
        video_data=None,
        audio_data=None,
        mm_content_hashes=[digest],
    )

    assert receiver._extract_url_data(request) == [
        {
            "url": "image",
            "modality": Modality.IMAGE,
            "detail": "high",
            "max_dynamic_patch": 12,
            "preprocess_kwargs": {"crop": False},
            "content_hash": digest,
        }
    ]

    assert _encoder_media_item(receiver._extract_url_data(request)[0]) == {
        "url": "image",
        "detail": "high",
        "max_dynamic_patch": 12,
        "preprocess_kwargs": {"crop": False},
        "content_hash": digest,
    }


def test_kimi_k3_epd_aggregates_image_metadata_in_part_order():
    first = EmbeddingData(
        req_id="request",
        num_parts=2,
        part_idx=0,
        grid_dim=torch.tensor([[1, 2, 6]]),
        modality=Modality.IMAGE,
        embedding=torch.ones(3, 4),
        original_image_sizes=[[1536, 1024]],
        mm_feature_identities=["sha256:" + "11" * 32],
    )
    second = EmbeddingData(
        req_id="request",
        num_parts=2,
        part_idx=1,
        grid_dim=torch.tensor([[1, 2, 4]]),
        modality=Modality.IMAGE,
        embedding=torch.ones(2, 4),
        original_image_sizes=[[1024, 1536]],
        mm_feature_identities=["sha256:" + "22" * 32],
    )

    combined = MultiModalEmbeddingData.from_embedding_data(first, model_type="kimi_k3")
    combined.add(second)

    assert combined.ready
    assert combined.get_mm_extra_meta()["original_image_sizes"] == [
        [1536, 1024],
        [1024, 1536],
    ]
    assert combined.get_mm_extra_meta()["mm_feature_identities"] == [
        "sha256:" + "11" * 32,
        "sha256:" + "22" * 32,
    ]


def test_kimi_k3_encoder_prefers_grid_thws_and_uses_temporal_pool_length():
    grid_thws = torch.tensor([[3, 8, 12]])
    stale_grid = torch.tensor([[1, 2, 2]])
    mm_inputs = {"grid_thws": grid_thws, "image_grid_thw": stale_grid}

    assert _get_mm_grid_dim(mm_inputs, Modality.IMAGE, "kimi_k3") is grid_thws
    assert _encoder().get_num_tokens(grid_thws[0], Modality.IMAGE) == 24


def test_kimi_k3_encoder_splits_cross_request_batch_into_single_grid_items():
    encoder = _encoder()
    grid_thws = torch.tensor([[1, 2, 2], [2, 2, 4], [1, 4, 2]])
    feature = torch.arange(56, dtype=torch.float32).reshape(28, 2)
    embeddings = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    captured = {}

    def get_feature_fn(items):
        captured["items"] = items
        return embeddings

    output = encoder._encode_missing(
        feature,
        {"pixel_values": feature, "grid_thws": grid_thws},
        indices=[2, 0, 1],
        modality=Modality.IMAGE,
        get_feature_fn=get_feature_fn,
        grid_thw=grid_thws,
        keep_on_gpu=True,
    )

    items = captured["items"]
    assert len(items) == 3
    expected_feature_slices = [feature[20:28], feature[0:4], feature[4:20]]
    expected_grids = [grid_thws[2:3], grid_thws[0:1], grid_thws[1:2]]
    for item, expected_feature, expected_grid in zip(
        items, expected_feature_slices, expected_grids
    ):
        torch.testing.assert_close(item.feature, expected_feature)
        torch.testing.assert_close(item.model_specific_data["grid_thws"], expected_grid)

    assert [embedding.shape[0] for embedding in output] == [2, 1, 2]
    torch.testing.assert_close(torch.cat(output), embeddings)


def test_encoder_preprocessed_items_follow_dp_owner_selection_order():
    encoder = _encoder()
    grid_thws = torch.tensor([[1, 2, 2], [1, 2, 4], [1, 4, 2]])
    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.full((3, i + 2, i + 3), i, dtype=torch.uint8),
            model_specific_data={"grid_thws": grid_thws[i : i + 1]},
        )
        for i in range(3)
    ]
    mm_inputs = EncoderPreprocessOutput(
        {"pixel_values": [item.feature for item in items], "grid_thws": grid_thws},
        mm_items=items,
    )
    embeddings = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    captured = {}

    def get_feature_fn(selected_items):
        captured["items"] = selected_items
        return embeddings

    output = encoder._encode_missing(
        mm_inputs["pixel_values"],
        mm_inputs,
        indices=[2, 0],
        modality=Modality.IMAGE,
        get_feature_fn=get_feature_fn,
        grid_thw=grid_thws,
        keep_on_gpu=True,
    )

    assert captured["items"] == [items[2], items[0]]
    assert [part.shape[0] for part in output] == [2, 1]
    torch.testing.assert_close(torch.cat(output), embeddings[:3])


def test_encoder_preprocessed_items_hash_individually():
    encoder = _encoder()
    grid_thws = torch.tensor([[1, 2, 2], [1, 2, 4]])
    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.full((3, 2, 2), value, dtype=torch.uint8),
            model_specific_data={"grid_thws": grid_thws[i : i + 1]},
        )
        for i, value in enumerate((17, 29))
    ]
    mm_inputs = EncoderPreprocessOutput(
        {"pixel_values": [item.feature for item in items], "grid_thws": grid_thws},
        mm_items=items,
    )

    hashes = encoder._calculate_hashes_from_features(
        mm_inputs["pixel_values"], grid_thws, Modality.IMAGE, mm_inputs
    )

    assert hashes == [item.hash for item in items]
    assert hashes[0] != hashes[1]


def test_kimi_k3_encoder_only_wrapper_guards_language_tower_hooks():
    model = SimpleNamespace(language_model=None)

    KimiK3ForConditionalGeneration.post_load_weights(model)
    with pytest.raises(AttributeError, match="lm_head"):
        KimiK3ForConditionalGeneration.lm_head.fget(model)
    with pytest.raises(AttributeError, match="DSPARK"):
        KimiK3ForConditionalGeneration.set_dspark_layers_to_capture(model, [0])


def test_epd_scheduler_uses_token_ids_for_tokenized_mm_processors():
    recv_req = SimpleNamespace(
        input_text="unexpanded prompt", input_ids=array("q", [11, 22, 33])
    )

    prompt = _select_mm_processor_prompt(
        recv_req, SimpleNamespace(prefer_tokenized_input=True)
    )

    assert prompt == [11, 22, 33]
    assert isinstance(prompt, list)
    assert (
        _select_mm_processor_prompt(
            recv_req, SimpleNamespace(prefer_tokenized_input=False)
        )
        == "unexpanded prompt"
    )


def test_epd_scheduler_routes_many_requests_over_one_receive_socket():
    context = zmq.Context()
    receiver = MMReceiverHTTP.__new__(MMReceiverHTTP)
    receiver.scheduler_recv_socket = context.socket(zmq.PULL)
    port = receiver.scheduler_recv_socket.bind_to_random_port("tcp://127.0.0.1")
    received = []

    class Sink:
        def consume_parts(self, parts):
            received.append(pickle.loads(parts[0]).req_id)

    receiver.waiting_by_rid = {f"rid-{i}": Sink() for i in range(32)}
    sender = context.socket(zmq.PUSH)
    try:
        sender.connect(f"tcp://127.0.0.1:{port}")
        for i in range(32):
            mm_data = EmbeddingData(
                req_id=f"rid-{i}_local_part_0",
                num_parts=1,
                part_idx=0,
                grid_dim=None,
                modality=Modality.IMAGE,
                error_msg="probe",
                error_code=599,
            )
            sender.send_multipart([pickle.dumps(mm_data)])

        deadline = time.monotonic() + 2
        while len(received) < 32 and time.monotonic() < deadline:
            receiver._drain_scheduler_embeddings()
            time.sleep(0.01)
        assert received == [f"rid-{i}_local_part_0" for i in range(32)]
    finally:
        sender.close(linger=0)
        receiver.scheduler_recv_socket.close(linger=0)
        context.term()


def test_epd_encoder_reuses_scheduler_zmq_peer():
    async def send_twice():
        context = zmq.asyncio.Context()
        receiver = context.socket(zmq.PULL)
        port = receiver.bind_to_random_port("tcp://127.0.0.1")
        encoder = MMEncoder.__new__(MMEncoder)
        config_override = get_context().override_server_args(
            encoder_transfer_backend="zmq_to_scheduler"
        )
        with config_override as server_args:
            encoder.server_args = server_args
            encoder.send_timeout = 3
            encoder.context = context
            encoder.scheduler_send_sockets = {}
            encoder.scheduler_send_locks = {}
            mm_data = EmbeddingData(
                req_id="test-rid_local_part_0",
                num_parts=1,
                part_idx=0,
                grid_dim=None,
                modality=Modality.IMAGE,
                error_msg="probe",
                error_code=599,
            )
            try:
                for _ in range(2):
                    await encoder._send(None, mm_data, url=f"127.0.0.1:{port}")
                    parts = await asyncio.wait_for(receiver.recv_multipart(), timeout=1)
                    assert pickle.loads(parts[0]).req_id == mm_data.req_id
                assert len(encoder.scheduler_send_sockets) == 1
            finally:
                for socket in encoder.scheduler_send_sockets.values():
                    socket.close(linger=0)
                receiver.close(linger=0)
                context.term()

    asyncio.run(send_twice())


def test_epd_encoder_pipelines_zero_copy_sends_per_peer():
    class FakeTracker:
        def __init__(self, release):
            self.release = release

        def wait(self, timeout):
            assert self.release.wait(timeout)

    class FakeSocket:
        def __init__(self, release, second_queued):
            self.release = release
            self.second_queued = second_queued
            self.send_count = 0

        def setsockopt(self, *_args):
            pass

        def connect(self, _endpoint):
            pass

        def close(self, **_kwargs):
            pass

        async def send_multipart(self, _frames, **_kwargs):
            self.send_count += 1
            if self.send_count == 2:
                self.second_queued.set()
            return FakeTracker(self.release)

    class FakeContext:
        def __init__(self, socket):
            self.socket_instance = socket

        def socket(self, _socket_type):
            return self.socket_instance

    async def run_test():
        release = threading.Event()
        second_queued = asyncio.Event()
        socket = FakeSocket(release, second_queued)
        encoder = MMEncoder.__new__(MMEncoder)
        config_override = get_context().override_server_args(
            encoder_transfer_backend="zmq_to_scheduler"
        )
        with config_override as server_args:
            encoder.server_args = server_args
            encoder.send_timeout = 1
            encoder.context = FakeContext(socket)
            encoder.scheduler_send_sockets = {}
            encoder.scheduler_send_locks = {}
            mm_data = EmbeddingData(
                req_id="test-rid_local_part_0",
                num_parts=1,
                part_idx=0,
                grid_dim=None,
                modality=Modality.IMAGE,
                error_msg="probe",
                error_code=599,
            )

            first = asyncio.create_task(
                encoder._send(None, mm_data, url="127.0.0.1:12345")
            )
            while socket.send_count < 1:
                await asyncio.sleep(0)
            second = asyncio.create_task(
                encoder._send(None, mm_data, url="127.0.0.1:12345")
            )
            try:
                await asyncio.wait_for(second_queued.wait(), timeout=0.5)
            finally:
                release.set()
            await asyncio.gather(first, second)

            assert socket.send_count == 2

    asyncio.run(run_test())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
