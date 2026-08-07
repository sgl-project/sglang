# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace

import pytest
import torch

from sglang.multimodal_gen.runtime.entrypoints import realtime_vae_server
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    LatentChunkHeader,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_worker import (
    AsyncVAEWorker,
    SessionOpen,
    TAEHVEngine,
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


class _StreamingEngine(_FakeEngine):
    def __init__(self):
        super().__init__()
        self.first_decoded = asyncio.Event()
        self.release_second = asyncio.Event()

    async def iter_decode(self, decoder, latents, *, first_chunk):
        del decoder, latents, first_chunk
        self.first_decoded.set()
        yield torch.zeros((1, 3, 1, 8, 8), dtype=torch.float32)
        await self.release_second.wait()
        yield torch.ones((1, 3, 1, 8, 8), dtype=torch.float32)


def test_taehv_engine_warmup_runs_a_production_shape_decode(monkeypatch):
    engine = TAEHVEngine.__new__(TAEHVEngine)
    engine.device = torch.device("cpu")
    engine.dtype = torch.bfloat16
    decoder = object()
    calls = []

    monkeypatch.setattr(engine, "create_decoder", lambda identity: decoder)

    def iter_decode(actual_decoder, latents, *, first_chunk):
        calls.append((actual_decoder, tuple(latents.shape), latents.dtype, first_chunk))
        yield torch.zeros(1, 3, 1, 480, 832)

    monkeypatch.setattr(engine, "iter_decode", iter_decode)

    engine.warmup()

    assert calls == [(decoder, (1, 48, 1, 30, 52), torch.bfloat16, True)]


def test_taehv_engine_decodes_model_space_latents_without_native_vae_denorm():
    class _StreamingDecoder:
        def __init__(self):
            self.inputs = []
            self.pending = []

        def reset(self):
            pass

        def decode(self, latents=None):
            if latents is not None:
                self.inputs.append(latents.detach().clone())
                self.pending.append(torch.zeros(1, 1, 3, 16, 16))
            if self.pending:
                return self.pending.pop(0)
            return None

    engine = TAEHVEngine.__new__(TAEHVEngine)
    engine.device = torch.device("cpu")
    engine.dtype = torch.float32
    engine.mean = torch.full((1, 48, 1, 1, 1), 10.0)
    engine.std = torch.full((1, 48, 1, 1, 1), 2.0)
    decoder = _StreamingDecoder()
    model_latents = torch.ones(1, 48, 1, 2, 2)

    frames = list(engine.iter_decode(decoder, model_latents, first_chunk=True))

    assert len(frames) == 1
    assert torch.equal(
        decoder.inputs[0],
        model_latents.permute(0, 2, 1, 3, 4),
    )


def test_realtime_vae_server_warms_engine_before_serving(monkeypatch):
    events = []

    class Engine:
        def __init__(self, *_args, **_kwargs):
            events.append("engine_created")

        def warmup(self):
            events.append("engine_warmed")

    class Worker:
        def __init__(self, *_args, **_kwargs):
            events.append("worker_created")

    monkeypatch.setattr(realtime_vae_server, "TAEHVEngine", Engine)
    monkeypatch.setattr(realtime_vae_server, "AsyncVAEWorker", Worker)
    monkeypatch.setattr(realtime_vae_server, "create_app", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        realtime_vae_server.uvicorn,
        "run",
        lambda *_args, **_kwargs: events.append("server_started"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["realtime-vae", "--checkpoint-path", "/tmp/taehv.pth"],
    )

    realtime_vae_server.main()

    assert events == [
        "engine_created",
        "engine_warmed",
        "worker_created",
        "server_started",
    ]


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


def test_worker_runtime_state_reports_processing_queue_and_service_time():
    async def scenario():
        engine = _BlockingEngine()
        worker = AsyncVAEWorker(engine, max_sessions=2, queue_depth_per_session=1)
        await worker.open(SessionOpen("s1", "g1"))
        first = await worker.submit(
            _header(chunk_index=0),
            torch.zeros(1, 48, 1, 2, 2, dtype=torch.bfloat16),
        )
        await engine.started.wait()
        second = await worker.submit(
            _header(chunk_index=1),
            torch.zeros(1, 48, 1, 2, 2, dtype=torch.bfloat16),
        )

        state = worker.runtime_state()
        assert state["runnable_sessions"] == 1
        assert state["blocked_sessions"] == 1
        assert state["queue_depth"] == 1

        engine.release.set()
        await first
        await second
        assert worker.runtime_state()["service_time_ms"] > 0
        await worker.close_all()

    asyncio.run(scenario())


def test_worker_rejects_duplicate_active_session_owner():
    async def scenario():
        worker = AsyncVAEWorker(_FakeEngine(), max_sessions=2)
        await worker.open(SessionOpen("s1", "g1"))

        with pytest.raises(ProtocolViolation, match="already active"):
            await worker.open(SessionOpen("s1", "g1"))

        await worker.close_all()

    from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
        ProtocolViolation,
    )

    asyncio.run(scenario())


def test_vae_websocket_cannot_rebind_to_another_session():
    async def scenario():
        worker = AsyncVAEWorker(_FakeEngine(), max_sessions=2)
        identity = await realtime_vae_server._bind_socket_session(
            worker,
            None,
            SessionOpen("s1", "g1"),
        )

        with pytest.raises(ProtocolViolation, match="already owns"):
            await realtime_vae_server._bind_socket_session(
                worker,
                identity,
                SessionOpen("s2", "g2"),
            )

        assert worker.active_sessions == 1
        await worker.close_all()

    from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
        ProtocolViolation,
    )

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


def test_worker_emits_first_streaming_frame_before_decode_finishes():
    async def scenario():
        engine = _StreamingEngine()
        worker = AsyncVAEWorker(engine, max_sessions=1, encoded_frames_per_batch=1)
        await worker.open(SessionOpen("s", "g"))
        first_emitted = asyncio.Event()
        batches = []

        async def on_frame_batch(batch):
            batches.append(batch)
            first_emitted.set()

        decode = asyncio.create_task(
            worker.decode(
                _header("s", "g", 0),
                torch.zeros(1, 48, 1, 2, 2, dtype=torch.bfloat16),
                on_frame_batch=on_frame_batch,
            )
        )
        await engine.first_decoded.wait()
        await asyncio.wait_for(first_emitted.wait(), timeout=1)
        assert not decode.done()

        engine.release_second.set()
        result = await decode
        assert result.num_frames == 2
        assert [batch.frame_batch_index for batch in batches] == [0, 1]
        assert [(batch.source_width, batch.source_height) for batch in batches] == [
            (8, 8),
            (8, 8),
        ]
        assert [(batch.preview_width, batch.preview_height) for batch in batches] == [
            (8, 8),
            (8, 8),
        ]
        await worker.close_all()

    asyncio.run(scenario())


def test_worker_coalesces_streaming_yields_into_configured_transport_batch():
    async def scenario():
        engine = _StreamingEngine()
        worker = AsyncVAEWorker(engine, max_sessions=1, encoded_frames_per_batch=2)
        await worker.open(SessionOpen("s", "g"))
        batches = []

        async def on_frame_batch(batch):
            batches.append(batch)

        decode = asyncio.create_task(
            worker.decode(
                _header("s", "g", 0),
                torch.zeros(1, 48, 1, 2, 2, dtype=torch.bfloat16),
                on_frame_batch=on_frame_batch,
            )
        )
        await engine.first_decoded.wait()
        await asyncio.sleep(0)
        assert batches == []

        engine.release_second.set()
        result = await decode

        assert result.num_frames == 2
        assert len(batches) == 1
        assert batches[0].num_frames == 2
        assert len(batches[0].payloads) == 2
        assert batches[0].frame_batch_index == 0
        assert batches[0].source_width == 8
        assert batches[0].source_height == 8
        assert batches[0].preview_width == 8
        assert batches[0].preview_height == 8
        await worker.close_all()

    asyncio.run(scenario())


def test_worker_grants_next_credit_only_after_the_job_enters_decode():
    async def scenario():
        engine = _BlockingEngine()
        worker = AsyncVAEWorker(engine, max_sessions=1, queue_depth_per_session=2)
        await worker.open(SessionOpen("s", "g"))
        decode_started = asyncio.Event()

        future = await worker.submit(
            _header("s", "g", 0),
            torch.zeros(1, 48, 1, 2, 2, dtype=torch.bfloat16),
            on_decode_started=decode_started.set,
        )

        await asyncio.wait_for(decode_started.wait(), timeout=1)
        assert engine.started.is_set()
        engine.release.set()
        await future
        await worker.close_all()

    asyncio.run(scenario())


def test_global_decode_actor_serves_waiting_sessions_fairly():
    class OrderedEngine(_FakeEngine):
        def __init__(self):
            super().__init__()
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.order = []

        async def decode(self, decoder, latents, *, first_chunk):
            del latents, first_chunk
            self.order.append(decoder)
            if len(self.order) == 1:
                self.first_started.set()
                await self.release_first.wait()
            await asyncio.sleep(0)
            return torch.zeros((1, 3, 1, 8, 8), dtype=torch.float32)

    async def scenario():
        engine = OrderedEngine()
        worker = AsyncVAEWorker(engine, max_sessions=2, queue_depth_per_session=2)
        await worker.open(SessionOpen("s1", "g1"))
        await worker.open(SessionOpen("s2", "g2"))
        latent = torch.zeros(1, 48, 1, 2, 2, dtype=torch.bfloat16)

        s1_first = await worker.submit(_header("s1", "g1", 0), latent)
        await engine.first_started.wait()
        s1_second = await worker.submit(_header("s1", "g1", 1), latent)
        s2_first = await worker.submit(_header("s2", "g2", 0), latent)
        await asyncio.sleep(0)
        engine.release_first.set()
        await asyncio.gather(s1_first, s1_second, s2_first)

        assert engine.order == [("s1", "g1"), ("s2", "g2"), ("s1", "g1")]
        await worker.close_all()

    asyncio.run(scenario())
