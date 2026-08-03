# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
import torch

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    LatentChunkHeader,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_worker import (
    AsyncVAEWorker,
    SessionOpen,
    VAEBackpressureError,
)


def _header(session="s1", generation="g1", chunk_index=0, **overrides):
    values = {
        "session_id": session,
        "generation_id": generation,
        "request_id": f"r{chunk_index}",
        "chunk_index": chunk_index,
        "dtype": "bfloat16",
        "shape": (1, 48, 1, 2, 2),
        "byte_length": 384,
        "checksum": "unit-test",
    }
    values.update(overrides)
    return LatentChunkHeader(**values)


class _FakeEngine:
    def __init__(self):
        self.decoder_ids = set()
        self.calls = []

    def create_decoder(self, identity):
        self.decoder_ids.add(identity)
        return identity

    async def decode(self, decoder, latents, *, first_chunk):
        self.calls.append((decoder, latents.clone(), first_chunk))
        frame_count = latents.shape[2]
        return torch.zeros((1, 3, frame_count, 8, 8), dtype=torch.float32)


class _BlockingEngine(_FakeEngine):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def decode(self, decoder, latents, *, first_chunk):
        self.started.set()
        await self.release.wait()
        return await super().decode(decoder, latents, first_chunk=first_chunk)


def test_worker_keeps_decoder_state_per_generation():
    async def scenario():
        engine = _FakeEngine()
        worker = AsyncVAEWorker(engine, max_sessions=2, queue_depth_per_session=1)
        await worker.open(SessionOpen("s1", "g1"))
        await worker.open(SessionOpen("s2", "g2"))

        await worker.decode(
            _header("s1", "g1"), torch.zeros(1, 48, 1, 2, 2, dtype=torch.bfloat16)
        )
        await worker.decode(
            _header("s2", "g2"), torch.zeros(1, 48, 1, 2, 2, dtype=torch.bfloat16)
        )

        assert engine.decoder_ids == {("s1", "g1"), ("s2", "g2")}
        await worker.close_all()

    asyncio.run(scenario())


def test_worker_rejects_second_waiting_latent():
    async def scenario():
        engine = _BlockingEngine()
        worker = AsyncVAEWorker(engine, max_sessions=1, queue_depth_per_session=1)
        await worker.open(SessionOpen("s", "g"))

        first = asyncio.create_task(
            worker.decode(
                _header("s", "g", 0),
                torch.zeros(1, 48, 1, 2, 2, dtype=torch.bfloat16),
            )
        )
        await engine.started.wait()
        second = asyncio.create_task(
            worker.decode(
                _header("s", "g", 1),
                torch.zeros(1, 48, 1, 2, 2, dtype=torch.bfloat16),
            )
        )
        await asyncio.sleep(0)

        with pytest.raises(VAEBackpressureError):
            await worker.decode(
                _header("s", "g", 2),
                torch.zeros(1, 48, 1, 2, 2, dtype=torch.bfloat16),
            )

        engine.release.set()
        await first
        await second
        await worker.close_all()

    asyncio.run(scenario())


def test_worker_t2v_reseeds_chunk_one_and_drops_duplicate_frame():
    async def scenario():
        engine = _FakeEngine()
        worker = AsyncVAEWorker(engine, max_sessions=1, queue_depth_per_session=1)
        await worker.open(SessionOpen("s", "g"))
        first_latent = torch.ones(1, 48, 1, 2, 2, dtype=torch.bfloat16)
        second_latent = torch.full(
            (1, 48, 2, 2, 2), 2.0, dtype=torch.bfloat16
        )

        first = await worker.decode(_header("s", "g", 0), first_latent)
        second = await worker.decode(
            replace(
                _header("s", "g", 1),
                shape=tuple(second_latent.shape),
                byte_length=second_latent.numel() * 2,
            ),
            second_latent,
        )

        assert first.num_frames == 1
        assert engine.calls[1][1].shape[2] == 3
        assert engine.calls[1][2] is True
        assert second.num_frames == 2
        await worker.close_all()

    asyncio.run(scenario())
