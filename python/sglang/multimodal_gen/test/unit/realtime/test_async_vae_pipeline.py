# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    realtime_video_api,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_video_api import (
    _OrderedDecodeCoordinator,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_client import (
    GatewayOutputClient,
    RealtimeVAEClient,
    RemoteFrameBatch,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    decode_message,
    encode_message,
)


def test_chunk_n_plus_one_denoises_while_chunk_n_decodes():
    async def scenario():
        timeline = []
        emitted = []
        coordinator = _OrderedDecodeCoordinator()

        async def denoise(index):
            timeline.append((f"denoise:{index}:start", asyncio.get_running_loop().time()))
            await asyncio.sleep(0.02)
            timeline.append((f"denoise:{index}:end", asyncio.get_running_loop().time()))

        async def decode(index):
            timeline.append((f"decode:{index}:start", asyncio.get_running_loop().time()))
            await asyncio.sleep(0.05)
            emitted.append(index)
            timeline.append((f"decode:{index}:end", asyncio.get_running_loop().time()))

        for index in range(3):
            await denoise(index)
            await coordinator.submit(lambda index=index: decode(index))
        await coordinator.finish()

        moments = dict(timeline)
        assert moments["denoise:1:start"] < moments["decode:0:end"]
        assert moments["denoise:2:start"] < moments["decode:1:end"]
        assert emitted == [0, 1, 2]

    asyncio.run(scenario())


def test_coordinator_cancel_stops_pending_decode():
    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()
        coordinator = _OrderedDecodeCoordinator()

        async def decode():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        await coordinator.submit(decode)
        await started.wait()
        await coordinator.cancel()
        assert cancelled.is_set()

    asyncio.run(scenario())


def test_remote_frame_handlers_keep_first_frame_trace_scoped_to_their_chunk():
    async def scenario():
        traces = []
        sends = []

        def fake_trace(_logger, _session, event, **fields):
            traces.append((event, fields))

        async def fake_send(_ws, _session, batch, frame_batch, send_stats):
            sends.append((batch.name, frame_batch.chunk_index, send_stats))

        session = SimpleNamespace()
        chunk_zero = SimpleNamespace(index=0, request_id="request-0")
        chunk_one = SimpleNamespace(index=1, request_id="request-1")
        batch_zero = SimpleNamespace(name="batch-0", realtime_event_id=10)
        batch_one = SimpleNamespace(name="batch-1", realtime_event_id=11)
        send_stats_zero = {}
        send_stats_one = {}

        with (
            patch.object(realtime_video_api, "log_realtime_trace", fake_trace),
            patch.object(realtime_video_api, "_send_remote_frame_batch", fake_send),
        ):
            handler_zero = realtime_video_api._make_remote_frame_batch_handler(
                object(), session, chunk_zero, batch_zero, send_stats_zero
            )
            handler_one = realtime_video_api._make_remote_frame_batch_handler(
                object(), session, chunk_one, batch_one, send_stats_one
            )
            frame_zero = SimpleNamespace(frame_batch_index=0, chunk_index=0)
            frame_one = SimpleNamespace(frame_batch_index=0, chunk_index=1)
            await handler_zero(frame_zero)
            await handler_zero(frame_zero)
            await handler_one(frame_one)

        first_frame_traces = [fields for _, fields in traces]
        assert [item["chunk_index"] for item in first_frame_traces] == [0, 1]
        assert [item["request_id"] for item in first_frame_traces] == [
            "request-0",
            "request-1",
        ]
        assert [item[0] for item in sends] == ["batch-0", "batch-0", "batch-1"]

    asyncio.run(scenario())


def test_remote_frame_batch_preserves_worker_stream_sequence_metadata():
    async def scenario():
        captured = []

        class Adapter:
            async def send_output(self, _ws, _session, output, _batch):
                captured.append(output.raw_frame_metadata)
                return realtime_video_api.empty_frame_send_stats()

        session = SimpleNamespace(adapter=Adapter())
        frame_batch = RemoteFrameBatch(
            session_id="s",
            generation_id="g",
            request_id="r",
            chunk_index=2,
            event_id=7,
            payloads=(b"webp",),
            content_type="image/webp",
            width=8,
            height=8,
            frame_batch_index=3,
            is_final=True,
            encode_ms=1.0,
        )
        await realtime_video_api._send_remote_frame_batch(
            SimpleNamespace(),
            session,
            SimpleNamespace(metrics=None),
            frame_batch,
            realtime_video_api.empty_frame_send_stats(),
        )

        assert captured == [
            {
                "width": 8,
                "height": 8,
                "channels": 3,
                "bytes_per_frame": 192,
                "frame_batch_index": 3,
                "num_frame_batches": 4,
                "is_final_frame_batch": True,
            }
        ]

    asyncio.run(scenario())


def test_async_vae_open_failure_still_closes_client(monkeypatch):
    async def scenario():
        closed = []

        class Client:
            def __init__(self, *_args, **_kwargs):
                pass

            async def open(self, **_kwargs):
                raise RuntimeError("handshake failed")

            async def close(self):
                closed.append(True)

        async def ignore_error(*_args, **_kwargs):
            pass

        monkeypatch.setattr(realtime_video_api, "RealtimeVAEClient", Client)
        monkeypatch.setattr(realtime_video_api, "write_error_msg", ignore_error)
        session = SimpleNamespace(
            adapter=object(),
            request=SimpleNamespace(
                realtime_output_format="webp",
                output_compression=80,
                realtime_preview_max_width=560,
            ),
            id="s",
            generation_id="g",
            vae_client=None,
        )
        server_args = SimpleNamespace(
            realtime_vae_worker_url="ws://vae",
            realtime_vae_timeout_s=1,
            realtime_vae_max_message_mb=64,
        )

        await realtime_video_api._generate_loop_async_vae(
            SimpleNamespace(), session, server_args
        )

        assert closed == [True]
        assert session.vae_client is None

    asyncio.run(scenario())


def test_remote_vae_client_streams_batches_before_chunk_completion():
    class FakeSocket:
        def __init__(self):
            self.sent = asyncio.Queue()
            self.received = asyncio.Queue()
            self.closed = False

        async def send(self, payload):
            await self.sent.put(payload)

        async def recv(self):
            return await self.received.get()

        async def close(self):
            self.closed = True

    async def scenario():
        socket = FakeSocket()
        connect_kwargs = {}

        async def connect_factory(*args, **kwargs):
            del args
            connect_kwargs.update(kwargs)
            return socket

        await socket.received.put(
            encode_message(
                "session_accepted",
                session_id="s",
                generation_id="g",
                credit_chunk_index=0,
            )
        )
        client = RealtimeVAEClient(
            "ws://vae",
            session_id="s",
            generation_id="g",
            connect_factory=connect_factory,
        )
        await client.open(
            output_format="webp",
            quality=80,
            preview_max_width=560,
            output_url="ws://gateway/v1/internal/realtime_output/s",
            output_token="output-secret",
            trace_id="trace-a",
            coordinator_token="coordinator-secret",
            worker_epoch="vae-epoch",
        )
        assert connect_kwargs["compression"] is None
        session_open = decode_message(await socket.sent.get())
        assert session_open["type"] == "session_open"
        assert session_open["output_url"].startswith("ws://gateway/")
        assert session_open["output_token"] == "output-secret"
        assert session_open["trace_id"] == "trace-a"
        assert session_open["coordinator_token"] == "coordinator-secret"
        assert session_open["worker_epoch"] == "vae-epoch"

        received_batches = []
        submit_task = asyncio.create_task(
            client.submit(
                torch.zeros(1, 48, 1, 2, 2, dtype=torch.bfloat16),
                {
                    "session_id": "s",
                    "generation_id": "g",
                    "request_id": "r0",
                    "chunk_index": 0,
                },
                on_frame_batch=lambda batch: _append_async(received_batches, batch),
            )
        )
        latent_message = decode_message(await socket.sent.get())
        await socket.received.put(
            encode_message(
                "latent_accepted",
                session_id="s",
                generation_id="g",
                request_id="r0",
                chunk_index=0,
            )
        )
        handle = await submit_task
        await socket.received.put(
            encode_message(
                "frame_batch",
                session_id="s",
                generation_id="g",
                request_id="r0",
                chunk_index=0,
                content_type="image/webp",
                width=8,
                height=8,
                payload_lengths=[2, 3],
                num_frames=2,
                frame_batch_index=0,
                is_final_frame_batch=True,
                payload=b"aabbb",
            )
        )
        await socket.received.put(
            encode_message(
                "chunk_complete",
                session_id="s",
                generation_id="g",
                request_id="r0",
                chunk_index=0,
                num_frames=2,
                queue_wait_ms=1,
                decode_ms=2,
                encode_ms=3,
            )
        )

        result = await handle.wait()
        assert latent_message["type"] == "latent_chunk"
        assert received_batches[0].payloads == (b"aa", b"bbb")
        assert result.num_frames == 2
        await client.close()
        assert socket.closed

    async def _append_async(items, value):
        items.append(value)

    asyncio.run(scenario())


def test_gateway_output_client_binds_identity_and_sends_frames_directly():
    class FakeSocket:
        def __init__(self):
            self.sent = asyncio.Queue()
            self.received = asyncio.Queue()
            self.closed = False

        async def send(self, payload):
            await self.sent.put(payload)

        async def recv(self):
            return await self.received.get()

        async def close(self):
            self.closed = True

    async def scenario():
        socket = FakeSocket()

        async def connect_factory(*_args, **_kwargs):
            return socket

        await socket.received.put(
            encode_message(
                "session_output_accepted",
                session_id="s",
                generation_id="g",
            )
        )
        client = GatewayOutputClient(
            "ws://gateway/v1/internal/realtime_output",
            session_id="s",
            generation_id="g",
            token="secret",
            connect_factory=connect_factory,
        )
        await client.open()
        opened = decode_message(await socket.sent.get())
        assert opened == {
            "version": 1,
            "type": "session_output_open",
            "session_id": "s",
            "generation_id": "g",
            "token": "secret",
        }

        frame = encode_message(
            "frame_batch",
            session_id="s",
            generation_id="g",
            chunk_index=0,
            frame_batch_index=0,
            payload_lengths=[1],
            payload=b"x",
        )
        await client.send(frame)
        assert await socket.sent.get() == frame
        completion = encode_message(
            "media_chunk_complete",
            session_id="s",
            generation_id="g",
            request_id="r0",
            chunk_index=0,
            num_frames=1,
        )
        await socket.received.put(
            encode_message(
                "media_chunk_complete_accepted",
                session_id="s",
                generation_id="g",
                request_id="r0",
                chunk_index=0,
            )
        )
        await client.send(completion)
        assert await socket.sent.get() == completion
        assert socket.received.empty()
        await client.close()
        assert socket.closed

    asyncio.run(scenario())


def test_remote_vae_client_orders_frame_callbacks_across_chunks():
    class FakeSocket:
        def __init__(self):
            self.sent = asyncio.Queue()
            self.received = asyncio.Queue()

        async def send(self, payload):
            await self.sent.put(payload)

        async def recv(self):
            return await self.received.get()

        async def close(self):
            pass

    async def scenario():
        socket = FakeSocket()

        async def connect_factory(*args, **kwargs):
            del args, kwargs
            return socket

        await socket.received.put(
            encode_message(
                "session_accepted",
                session_id="s",
                generation_id="g",
                credit_chunk_index=0,
            )
        )
        client = RealtimeVAEClient(
            "ws://vae",
            session_id="s",
            generation_id="g",
            connect_factory=connect_factory,
        )
        await client.open(output_format="webp", quality=80, preview_max_width=560)
        await socket.sent.get()

        first_started = asyncio.Event()
        release_first = asyncio.Event()
        second_started = asyncio.Event()

        async def first_callback(_batch):
            first_started.set()
            await release_first.wait()

        async def second_callback(_batch):
            second_started.set()

        async def submit(chunk_index, request_id, callback):
            task = asyncio.create_task(
                client.submit(
                    torch.zeros(1, 48, 1, 2, 2, dtype=torch.bfloat16),
                    {
                        "session_id": "s",
                        "generation_id": "g",
                        "request_id": request_id,
                        "chunk_index": chunk_index,
                    },
                    on_frame_batch=callback,
                )
            )
            await socket.sent.get()
            await socket.received.put(
                encode_message(
                    "latent_accepted",
                    session_id="s",
                    generation_id="g",
                    request_id=request_id,
                    chunk_index=chunk_index,
                )
            )
            return await task

        first_handle = await submit(0, "r0", first_callback)
        second_handle = await submit(1, "r1", second_callback)
        for chunk_index, request_id in ((0, "r0"), (1, "r1")):
            await socket.received.put(
                encode_message(
                    "frame_batch",
                    session_id="s",
                    generation_id="g",
                    request_id=request_id,
                    chunk_index=chunk_index,
                    content_type="image/webp",
                    width=8,
                    height=8,
                    payload_lengths=[1],
                    num_frames=1,
                    frame_batch_index=0,
                    payload=b"x",
                )
            )

        await asyncio.wait_for(first_started.wait(), 1)
        await asyncio.sleep(0.01)
        assert not second_started.is_set()
        release_first.set()
        await asyncio.wait_for(second_started.wait(), 1)
        for chunk_index, request_id in ((0, "r0"), (1, "r1")):
            await socket.received.put(
                encode_message(
                    "chunk_complete",
                    session_id="s",
                    generation_id="g",
                    request_id=request_id,
                    chunk_index=chunk_index,
                    num_frames=1,
                )
            )
        await first_handle.wait()
        await second_handle.wait()
        await client.close()

    asyncio.run(scenario())


def test_remote_vae_client_fails_and_removes_pending_on_frame_callback_error():
    class FakeSocket:
        def __init__(self):
            self.sent = asyncio.Queue()
            self.received = asyncio.Queue()

        async def send(self, payload):
            await self.sent.put(payload)

        async def recv(self):
            return await self.received.get()

        async def close(self):
            pass

    async def scenario():
        socket = FakeSocket()

        async def connect_factory(*args, **kwargs):
            del args, kwargs
            return socket

        await socket.received.put(
            encode_message(
                "session_accepted",
                session_id="s",
                generation_id="g",
                credit_chunk_index=0,
            )
        )
        client = RealtimeVAEClient(
            "ws://vae",
            session_id="s",
            generation_id="g",
            timeout_s=1,
            connect_factory=connect_factory,
        )
        await client.open(output_format="webp", quality=80, preview_max_width=560)
        await socket.sent.get()

        async def fail_callback(_batch):
            raise RuntimeError("downstream send failed")

        submit_task = asyncio.create_task(
            client.submit(
                torch.zeros(1, 48, 1, 2, 2, dtype=torch.bfloat16),
                {
                    "session_id": "s",
                    "generation_id": "g",
                    "request_id": "r0",
                    "chunk_index": 0,
                },
                on_frame_batch=fail_callback,
            )
        )
        await socket.sent.get()
        await socket.received.put(
            encode_message(
                "latent_accepted",
                session_id="s",
                generation_id="g",
                request_id="r0",
                chunk_index=0,
            )
        )
        handle = await submit_task
        await socket.received.put(
            encode_message(
                "frame_batch",
                session_id="s",
                generation_id="g",
                request_id="r0",
                chunk_index=0,
                content_type="image/webp",
                width=8,
                height=8,
                payload_lengths=[1],
                num_frames=1,
                frame_batch_index=0,
                payload=b"x",
            )
        )
        await socket.received.put(
            encode_message(
                "chunk_complete",
                session_id="s",
                generation_id="g",
                request_id="r0",
                chunk_index=0,
                num_frames=1,
            )
        )

        with pytest.raises(RuntimeError, match="downstream send failed"):
            await handle.wait()
        assert not client._pending
        await client.close()

    asyncio.run(scenario())
