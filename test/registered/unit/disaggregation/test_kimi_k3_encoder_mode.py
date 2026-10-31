import asyncio
import pickle
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import zmq
import zmq.asyncio
from fastapi import HTTPException

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.disaggregation.encoder.receiver import (
    EmbeddingData,
    MMReceiverHTTP,
    WaitingMMRequestStatus,
    WaitingZmqRequest,
)
from sglang.srt.disaggregation.encoder.server import MMEncoder
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.managers.tokenizer_manager import (
    _reject_missing_dispatched_encoder_embedding,
)
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def test_epd_rejection_reads_the_resolved_transfer_backend():
    """This guard fires on the *resolved* backend.

    The record is produced by actual resolution -- a language-only Kimi-K3
    launch at TP2, whose `encoder_transfer_backend` starts at the argument
    default `"auto"` and is filled in by `resolve_encoder_transfer_backend` to
    `"zmq_to_tokenizer"`. The guard
    reads that resolved value out of the published bags, so the rejection
    survives the record going raw: what a reader must never do is go back to
    the record for this field.
    Fixed doubles cannot trip on that change, so the record here must come
    from resolution, not a SimpleNamespace.
    """
    import json
    import os
    import shutil
    import tempfile

    from sglang.srt.server_args import ServerArgs

    def env_field_flags():
        from sglang.srt.environ import EnvField, envs

        return {
            name: field._set_to_none
            for klass in reversed(type(envs).__mro__)
            for name, field in vars(klass).items()
            if isinstance(field, EnvField)
        }

    config_dir = tempfile.mkdtemp(prefix="epd_tripwire_")
    try:
        payload = {
            "architectures": ["KimiK3ForConditionalGeneration"],
            "model_type": "kimi_k3",
            "text_config": {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "hidden_size": 16,
                "intermediate_size": 32,
                "moe_intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "n_routed_experts": 8,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "first_k_dense_replace": 1,
                "vocab_size": 128,
                "max_position_embeddings": 2048,
                "kv_lora_rank": 8,
                "q_lora_rank": 8,
                "qk_nope_head_dim": 8,
                "qk_rope_head_dim": 8,
                "v_head_dim": 8,
                "topk_method": "greedy",
                "scoring_func": "softmax",
            },
            "vision_config": {
                "model_type": "kimi_k3_vision",
                "hidden_size": 16,
                "num_heads": 2,
                "depth": 2,
                "patch_size": 14,
                "merge_kernel_size": [2, 2],
            },
        }
        with open(os.path.join(config_dir, "config.json"), "w") as handle:
            json.dump(payload, handle)
        environ_before = dict(os.environ)
        flags_before = env_field_flags()
        try:
            resolved = ServerArgs(
                model_path=config_dir,
                device="cuda",
                random_seed=42,
                language_only=True,
                tp_size=2,
                # Resolution branches on the host device for the hybrid
                # state-cache sizing (extra_buffer asserts a GPU stack, which
                # the CPU CI runner does not have); the guard under test reads
                # `encoder_transfer_backend`, independent of that branch, so
                # pin the strategy every host can resolve.
                mamba_radix_cache_strategy="no_buffer",
                disable_overlap_schedule=True,
            )
            resolved.resolve_once()
        finally:
            os.environ.clear()
            os.environ.update(environ_before)
            from sglang.srt.environ import envs

            for name, was_none in flags_before.items():
                getattr(type(envs), name)._set_to_none = was_none
    finally:
        shutil.rmtree(config_dir, ignore_errors=True)

    assert resolution_result(resolved, "encoder_transfer_backend") == "zmq_to_tokenizer"
    # Publish that record: the guard reads the resolved value out of the bags,
    # so a raw record does not silently disable the rejection.
    publish(resolved, role="tokenizer")
    try:
        request = SimpleNamespace(need_wait_for_mm_inputs=True)
        with pytest.raises(HTTPException) as exc_info:
            _reject_missing_dispatched_encoder_embedding(request, None)
        assert getattr(exc_info.value, "status_code", None) == 503
    finally:
        reset_context()


def test_epd_scheduler_contains_invalid_embedding_part_metadata():
    waiting = WaitingZmqRequest.__new__(WaitingZmqRequest)
    waiting.rid = "request"
    waiting.recv_req = SimpleNamespace(rid="request")
    waiting.status = WaitingMMRequestStatus.PENDING
    waiting.recv_embedding_data = None
    waiting.model_type = None
    waiting._fail_and_release = Mock()
    invalid = EmbeddingData(
        req_id="request_local_part_2",
        num_parts=2,
        part_idx=2,
        grid_dim=None,
        modality=Modality.IMAGE,
        embedding=torch.ones(1, 2),
    )

    waiting.consume_parts(
        [pickle.dumps(invalid.copy_without_embedding()), invalid.embedding.numpy()]
    )

    waiting._fail_and_release.assert_called_once()


def test_epd_tokenizer_contains_duplicate_embedding_part():
    class FakeSocket:
        def __init__(self, messages):
            self.messages = messages
            self.closed = False

        async def recv_multipart(self, copy=False):
            return self.messages.pop(0)

        def close(self):
            self.closed = True

    async def run_test():
        embedding = torch.tensor([[1.0, 2.0]])
        part = EmbeddingData(
            req_id="request_local_part_0",
            num_parts=2,
            part_idx=0,
            grid_dim=torch.tensor([[1, 2, 2]]),
            modality=Modality.IMAGE,
            embedding=embedding,
        )
        frame = [pickle.dumps(part.copy_without_embedding()), embedding.numpy()]
        socket = FakeSocket([frame, frame])
        receiver = MMReceiverHTTP.__new__(MMReceiverHTTP)
        receiver.model_type = None

        result = await receiver._recv_mm_data(
            "request", socket, SimpleNamespace(), "prompt"
        )

        assert result is None
        assert socket.closed

    asyncio.run(run_test())


def test_epd_tokenizer_ignores_foreign_part_before_current_embedding():
    class FakeSocket:
        def __init__(self, messages):
            self.messages = list(messages)
            self.closed = False

        async def recv_multipart(self, copy=False):
            return self.messages.pop(0)

        def close(self):
            self.closed = True

    async def run_test():
        stale_error = EmbeddingData(
            req_id="stale_local_part_0",
            num_parts=1,
            part_idx=0,
            grid_dim=None,
            modality=Modality.IMAGE,
            error_msg="stale failure",
            error_code=500,
        )
        embedding = torch.tensor([[1.0, 2.0]])
        current = EmbeddingData(
            req_id="current_local_part_0",
            num_parts=1,
            part_idx=0,
            grid_dim=None,
            modality=Modality.IMAGE,
            embedding=embedding,
        )
        socket = FakeSocket(
            [
                [pickle.dumps(stale_error)],
                [pickle.dumps(current.copy_without_embedding()), embedding.numpy()],
            ]
        )
        receiver = MMReceiverHTTP.__new__(MMReceiverHTTP)
        receiver.model_type = None
        processor = SimpleNamespace(
            get_mm_data=lambda _prompt, embeddings, **_kwargs: embeddings,
            get_validated_mm_data=lambda _prompt, embeddings, **_kwargs: embeddings,
        )

        result = await receiver._recv_mm_data("current", socket, processor, "prompt")

        torch.testing.assert_close(result[Modality.IMAGE], embedding)
        assert socket.closed

    asyncio.run(run_test())


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
        sender.send_multipart([b"not a pickle"])
        sender.send_multipart([pickle.dumps("not embedding data")])
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
            encoder.transfer_backend = "zmq_to_scheduler"
            encoder.use_mooncake = False
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
            encoder.transfer_backend = "zmq_to_scheduler"
            encoder.use_mooncake = False
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
