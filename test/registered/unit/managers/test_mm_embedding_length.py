from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers import mm_schedule as mm_utils
from sglang.srt.managers import mm_utils as mm_embedding
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult
from sglang.srt.multimodal.transport.cuda_ipc import CUDA_IPC_FEATURE_COPY_EVENT_KEY
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-test-cpu-intel")


@pytest.mark.parametrize(
    (
        "prefix_length",
        "extend_length",
        "items_offset_list",
        "expected",
    ),
    [
        ([8], [16], [[(2, 5), (9, 14), (20, 24)]], 10),
        ([30], [0], [[(2, 5), (9, 14), (20, 24)]], 0),
        (
            [4, 0, 10],
            [4, 10, 10],
            [[(2, 5)], [], [(5, 12), (18, 25)]],
            7,
        ),
    ],
)
def test_count_mm_tokens_in_extend(
    prefix_length, extend_length, items_offset_list, expected
):
    input_ids = []
    for prefix, extend, item_offsets in zip(
        prefix_length, extend_length, items_offset_list
    ):
        seq_len = max(
            prefix + extend,
            max((item_end + 1 for _, item_end in item_offsets), default=0),
        )
        req_input_ids = torch.zeros(seq_len, dtype=torch.long)
        for item_start, item_end in item_offsets:
            req_input_ids[item_start : item_end + 1] = 1
        input_ids.append(req_input_ids[prefix : prefix + extend])

    actual = torch.isin(torch.cat(input_ids), torch.tensor([1])).sum().item()
    derived = mm_utils._count_mm_tokens_in_extend(
        prefix_length=prefix_length,
        extend_length=extend_length,
        items_offset_list=items_offset_list,
    )
    assert actual == derived == expected


def test_get_embedding_and_mask_uses_offset_count_without_readback():
    input_ids = torch.zeros(8, dtype=torch.long)
    input_ids[2:5] = 1
    embedding = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    mask = Mock()
    mask.sum.side_effect = AssertionError("mask count must stay on device")

    with (
        envs.SGLANG_ENABLE_ASYNC_ASSERT.override(False),
        patch.object(mm_utils, "_get_precomputed_embedding", return_value=embedding),
        patch.object(mm_utils, "_get_multimodal_mask", return_value=mask),
    ):
        result, result_mask, result_input_ids = mm_utils.get_embedding_and_mask(
            data_embedding_func=Mock(),
            embedding_items=[],
            placeholder_tensor=torch.tensor([1]),
            input_ids=input_ids,
            items_size=[0, 1],
            prefix_length=[0],
            extend_length=[8],
            items_offset_list=[[(2, 4)]],
        )

    mask.sum.assert_not_called()
    assert result is embedding
    assert result_mask is mask
    assert result_input_ids is input_ids


def test_get_embedding_and_mask_async_asserts_offset_count():
    input_ids = torch.zeros(8, dtype=torch.long)
    input_ids[2:5] = 1
    embedding = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    with (
        envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True),
        patch.object(mm_utils, "_get_precomputed_embedding", return_value=embedding),
        patch.object(mm_utils.torch, "_assert_async") as assert_async,
    ):
        mm_utils.get_embedding_and_mask(
            data_embedding_func=Mock(),
            embedding_items=[],
            placeholder_tensor=torch.tensor([1]),
            input_ids=input_ids,
            items_size=[0, 1],
            prefix_length=[0],
            extend_length=[8],
            items_offset_list=[[(2, 4)]],
        )

    assert_async.assert_called_once()
    condition, message = assert_async.call_args.args
    assert condition.item()
    assert "derived from offsets" in message


def test_adjust_embedding_length_crops_overlong_embedding():
    embedding = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    server_args = Mock(chunked_prefill_size=-1)

    with patch.object(mm_utils, "get_schedule", return_value=server_args):
        result = mm_utils._adjust_embedding_length(embedding, 3, Mock())

    torch.testing.assert_close(result, embedding[-3:], rtol=0, atol=0)


def test_adjust_embedding_length_rejects_short_embedding():
    embedding = torch.zeros(2, 4)

    with pytest.raises(RuntimeError, match="Insufficient multimodal embedding length"):
        mm_utils._adjust_embedding_length(embedding, 3, Mock())


def test_get_embedding_and_mask_falls_back_after_input_ids_rewrite():
    input_ids = torch.zeros(8, dtype=torch.long)
    rewritten_input_ids = input_ids.clone()
    embedding = torch.zeros(2, 4)
    mask_sum = Mock()
    mask_sum.item.return_value = 2
    mask = Mock()
    mask.sum.return_value = mask_sum

    with (
        patch.object(mm_utils, "_get_precomputed_embedding", return_value=None),
        patch.object(
            mm_utils,
            "_get_chunked_prefill_embedding",
            return_value=(embedding, rewritten_input_ids),
        ),
        patch.object(mm_utils, "_get_multimodal_mask", return_value=mask),
    ):
        result, result_mask, result_input_ids = mm_utils.get_embedding_and_mask(
            data_embedding_func=Mock(),
            embedding_items=[],
            placeholder_tensor=torch.tensor([1]),
            input_ids=input_ids,
            items_size=[0, 1],
            prefix_length=[0],
            extend_length=[8],
            items_offset_list=[[(2, 4)]],
        )

    mask.sum.assert_called_once_with()
    mask_sum.item.assert_called_once_with()
    assert result is embedding
    assert result_mask is mask
    assert result_input_ids is rewritten_input_ids


def _encode_feature_items(route, items, encoder):
    offsets = [(i, i) for i in range(len(items))]
    device = torch.device("cpu")
    if route == "full":
        return mm_utils._get_chunked_embedding_full(
            encoder, items, offsets, 0, len(items), torch.zeros(len(items)), device
        )[0]
    if route == "per_item":
        return mm_utils._get_chunked_embedding_by_item(
            encoder, items, offsets, 0, len(items), device
        )
    request = mm_utils.PerImageRequestInfo(0, items, offsets, 0, len(items))
    embeddings = mm_utils._batch_encode_per_image_misses(encoder, [request], device)
    return torch.cat([embeddings[(item.hash, 1)] for item in items])


@pytest.mark.parametrize("route", ["full", "per_item", "batched"])
@pytest.mark.parametrize("cached", [False, True])
def test_embedding_waits_for_host_features_only_on_cache_miss(route, cached):
    ready = Mock()
    feature = torch.ones(1, 4)
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=1,
        feature=feature,
        model_specific_data={CUDA_IPC_FEATURE_COPY_EVENT_KEY: ready},
    )

    def encode(items):
        ready.synchronize.assert_called_once_with()
        assert CUDA_IPC_FEATURE_COPY_EVENT_KEY not in items[0].model_specific_data
        return [items[0].feature.clone()]

    encoder = Mock(side_effect=encode)
    cache_entry = EmbeddingResult(embedding=feature) if cached else None
    cache = Mock()
    cache.get.return_value = cache.get_single.return_value = cache_entry
    with (
        patch.object(mm_utils, "embedding_cache", cache),
        patch.object(mm_utils, "_can_skip_pre_embed_feature_move", return_value=True),
        patch.object(mm_utils, "_acknowledge_deferred_cuda_ipc_cache_hits"),
    ):
        result = _encode_feature_items(route, [item], encoder)
    assert torch.equal(result, feature)
    if cached:
        ready.synchronize.assert_not_called()
        encoder.assert_not_called()
        assert item.model_specific_data[CUDA_IPC_FEATURE_COPY_EVENT_KEY] is ready


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("buffer_mb", [0, 1])
def test_feature_offload_waits_for_copy_without_draining_stream(dtype, buffer_mb):
    sources = [
        torch.full(shape, value, device="cuda", dtype=dtype)
        for shape, value in (((3, 28, 28), 1), ((2, 3, 28, 28), 2))
    ]
    items = [
        MultimodalDataItem(modality=modality, hash=i + 1, feature=source)
        for i, (modality, source) in enumerate(
            zip((Modality.IMAGE, Modality.VIDEO), sources)
        )
    ]
    expected = [source.cpu().to(torch.bfloat16) for source in sources]
    # Warm the host allocator before delaying the copy stream.
    offload_dtype = torch.float32 if buffer_mb else dtype
    hosts = [
        source.to(offload_dtype).to("cpu", non_blocking=True) for source in sources
    ]
    buffer = torch.empty(16384, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    del hosts
    copy_stream = torch.cuda.Stream()
    forward_stream = torch.cuda.Stream()
    tail = torch.cuda.Event()
    forward_tail = torch.cuda.Event()
    cache = Mock()
    cache.get.return_value = None

    def encode(ready_items):
        assert all(event.query() for event in copy_events)
        assert not tail.query()
        assert not forward_tail.query()
        staged = []
        for item, reference in zip(ready_items, expected):
            pixels = item.feature.to(torch.bfloat16)
            packed = torch.empty_like(pixels, pin_memory=True)
            packed.copy_(pixels)
            assert torch.equal(packed, reference)
            staged.append(packed.flatten()[:4].reshape(1, 4))
        return staged

    try:
        with (
            envs.SGLANG_MM_BUFFER_SIZE_MB.override(buffer_mb),
            patch.object(mm_embedding, "_GPU_FEATURE_BUFFER", buffer),
            patch.object(mm_embedding, "_BUFFER_OFFSET", 0),
            torch.cuda.stream(copy_stream),
        ):
            torch.cuda._sleep(300_000_000)
            if buffer_mb:
                MultimodalInputs.from_processor_output(
                    MultimodalProcessorOutput(mm_items=items)
                )
            else:
                batch = Mock(
                    mm_inputs=[MultimodalInputs(mm_items=items)],
                    extend_prefix_lens_cpu=[0],
                    extend_seq_lens_cpu=[2],
                    spec_algorithm=None,
                    input_embeds=None,
                )
                batch.forward_mode.is_decode.return_value = False
                batch.forward_mode.is_target_verify.return_value = False
                with (
                    patch.object(mm_embedding, "get_server_args", return_value=None),
                    patch.object(
                        mm_embedding,
                        "get_disagg",
                        return_value=Mock(language_only=False),
                    ),
                    patch.object(
                        mm_embedding,
                        "embed_mm_inputs",
                        return_value=(torch.zeros(2, 4), {}),
                    ),
                ):
                    mm_embedding.general_mm_embed_routine(torch.zeros(2), batch, Mock())
            copy_events = [
                item.model_specific_data[CUDA_IPC_FEATURE_COPY_EVENT_KEY]
                for item in items
            ]
            assert not copy_events[-1].query()
            torch.cuda._sleep(600_000_000)
            tail.record()

        with (
            patch.object(mm_utils, "embedding_cache", cache),
            patch.object(
                mm_utils, "_can_skip_pre_embed_feature_move", return_value=True
            ),
            torch.cuda.stream(forward_stream),
        ):
            # A GPU dependency alone does not make CPU packing safe.
            forward_stream.wait_event(copy_events[-1])
            torch.cuda._sleep(600_000_000)
            forward_tail.record()
            result = _encode_feature_items("full", items, encode)
        assert torch.equal(
            result, torch.tensor([[1] * 4, [2] * 4], dtype=torch.bfloat16)
        )
        assert all(
            CUDA_IPC_FEATURE_COPY_EVENT_KEY not in item.model_specific_data
            for item in items
        )
    finally:
        copy_stream.synchronize()
        forward_stream.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_feature_wait_keeps_the_host_asynchronous():
    source = torch.zeros(4, device="cuda")
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    item = MultimodalDataItem(modality=Modality.IMAGE, feature=source)
    try:
        with torch.cuda.stream(producer):
            torch.cuda._sleep(300_000_000)
            source.fill_(7)
            ready = producer.record_event()
        item.model_specific_data[CUDA_IPC_FEATURE_COPY_EVENT_KEY] = ready
        with torch.cuda.stream(consumer):
            item.offload_feature()
        assert not ready.query()
        item.wait_for_feature()
        assert torch.equal(item.feature, torch.full((4,), 7.0))
    finally:
        producer.synchronize()
        consumer.synchronize()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
