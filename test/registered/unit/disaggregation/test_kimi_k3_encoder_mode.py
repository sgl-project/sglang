import asyncio
import pickle
import sys
import threading
import time
from array import array
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
    _select_mm_processor_prompt,
)
from sglang.srt.disaggregation.encode_server import MMEncoder, _get_mm_grid_dim
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.managers.tokenizer_manager import (
    _reject_missing_dispatched_encoder_embedding,
)
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import resolve_encoder_transfer_backend
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


def test_kimi_k3_epd_aggregates_original_image_sizes_in_part_order():
    first = EmbeddingData(
        req_id="request",
        num_parts=2,
        part_idx=0,
        grid_dim=torch.tensor([[1, 2, 6]]),
        modality=Modality.IMAGE,
        embedding=torch.ones(3, 4),
        original_image_sizes=[[1536, 1024]],
    )
    second = EmbeddingData(
        req_id="request",
        num_parts=2,
        part_idx=1,
        grid_dim=torch.tensor([[1, 2, 4]]),
        modality=Modality.IMAGE,
        embedding=torch.ones(2, 4),
        original_image_sizes=[[1024, 1536]],
    )

    combined = MultiModalEmbeddingData.from_embedding_data(first, model_type="kimi_k3")
    combined.add(second)

    assert combined.ready
    assert combined.get_mm_extra_meta()["original_image_sizes"] == [
        [1536, 1024],
        [1024, 1536],
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
