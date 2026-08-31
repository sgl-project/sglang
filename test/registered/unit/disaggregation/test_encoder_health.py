import asyncio
import sys

import pytest

from sglang.srt.disaggregation.encoder import http_server
from sglang.srt.managers.schedule_batch import Modality
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


class _FakeEncoder:
    def __init__(self):
        self.audio_processor = None
        self.image_processor = object()
        self.embedding_to_send = {}
        self.encode_dispatch_lock = asyncio.Lock()
        self.encode_calls = []

    def has_pending_embeddings(self):
        return bool(self.embedding_to_send)

    def supports_modality(self, modality):
        return modality == Modality.IMAGE

    async def encode(self, **kwargs):
        self.encode_calls.append(kwargs)
        return 1, 1, 1, None, None

    async def release_request(self, _req_id):
        return None


def _install_tp_encoder(monkeypatch, encoder):
    broadcasts = []
    monkeypatch.setattr(http_server, "dp_dispatcher", None)
    monkeypatch.setattr(http_server, "encoder", encoder)
    monkeypatch.setattr(http_server, "send_sockets", [object()])
    monkeypatch.setattr(
        http_server,
        "sock_send",
        lambda socket, payload: broadcasts.append((socket, payload)),
    )
    return broadcasts


def test_health_encode_waits_for_collective_dispatch_lock(monkeypatch):
    async def run_test():
        encoder = _FakeEncoder()
        broadcasts = _install_tp_encoder(monkeypatch, encoder)
        await encoder.encode_dispatch_lock.acquire()

        task = asyncio.create_task(http_server.health_generate())
        await asyncio.sleep(0)
        assert broadcasts == []
        assert encoder.encode_calls == []

        encoder.encode_dispatch_lock.release()
        response = await task
        assert response.status_code == 200
        assert len(broadcasts) == 1
        assert len(encoder.encode_calls) == 1

    asyncio.run(run_test())


def test_health_encode_rechecks_busy_state_after_waiting(monkeypatch):
    async def run_test():
        encoder = _FakeEncoder()
        broadcasts = _install_tp_encoder(monkeypatch, encoder)
        await encoder.encode_dispatch_lock.acquire()

        task = asyncio.create_task(http_server.health_generate())
        await asyncio.sleep(0)
        encoder.embedding_to_send["real-request"] = object()
        encoder.encode_dispatch_lock.release()

        response = await task
        assert response.status_code == 200
        assert broadcasts == []
        assert encoder.encode_calls == []

    asyncio.run(run_test())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
