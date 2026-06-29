import asyncio
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.library_api import GenerationService


def test_forward_sync_passthrough():
    called = {}

    class FakeSyncClient:
        def forward(self, payload):
            called["payload"] = payload
            return {"ok": True}

    service = GenerationService(sync_client=FakeSyncClient(), async_client=object())
    response = service.forward_sync(["req"])

    assert called["payload"] == ["req"]
    assert response == {"ok": True}


def test_forward_async_passthrough():
    called = {}

    class FakeAsyncClient:
        async def forward(self, payload):
            called["payload"] = payload
            return {"ok": True}

    service = GenerationService(sync_client=object(), async_client=FakeAsyncClient())
    response = asyncio.run(service.forward_async(["req"]))

    assert called["payload"] == ["req"]
    assert response == {"ok": True}


def test_ensure_success_raises_on_error():
    output = SimpleNamespace(error="boom")

    with pytest.raises(RuntimeError, match="Failed op: boom"):
        GenerationService.ensure_success(output, failure_message="Failed op")


def test_control_sync_uses_error_mapping():
    class FakeSyncClient:
        def forward(self, _payload):
            return SimpleNamespace(error=None, output={"status": "ok"})

    service = GenerationService(sync_client=FakeSyncClient(), async_client=object())
    output = service.control_sync(req={"method": "x"}, failure_message="Failed")

    assert output.output["status"] == "ok"


def test_build_request_delegates(monkeypatch):
    sentinel_req = object()
    called = {}

    def fake_prepare_request(server_args, sampling_params):
        called["server_args"] = server_args
        called["sampling_params"] = sampling_params
        return sentinel_req

    service = GenerationService(sync_client=object(), async_client=object())
    monkeypatch.setattr(service, "_prepare_request", fake_prepare_request)
    server_args = object()
    sampling_params = object()
    req = service.build_request(server_args, sampling_params)

    assert called["server_args"] is server_args
    assert called["sampling_params"] is sampling_params
    assert req is sentinel_req
