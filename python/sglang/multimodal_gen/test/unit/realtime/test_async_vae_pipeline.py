# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import torch

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_video_api import (
    _OrderedDecodeCoordinator,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_client import RealtimeVAEClient
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
        assert decode_message(await socket.sent.get())["type"] == "session_open"

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
