import asyncio
import importlib.util
import ipaddress
import os

import httpx
import pytest

if importlib.util.find_spec("triton") is None:
    pytest.skip("triton is required for multimodal imports", allow_module_level=True)
if importlib.util.find_spec("diffusers") is None:
    pytest.skip("diffusers is required for multimodal imports", allow_module_level=True)

from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.srt.utils import common as common_utils


class DummyNetworkStream:
    def __init__(self, peername):
        self._peername = peername

    def get_extra_info(self, name, default=None):
        if name == "peername":
            return self._peername
        return default


def _patch_async_client(monkeypatch, transport: httpx.MockTransport) -> None:
    class DummyAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(openai_utils.httpx, "AsyncClient", DummyAsyncClient)


def test_openai_media_url_preserves_hostname_and_peer_ip(monkeypatch, tmp_path):
    resolved_ips = [ipaddress.ip_address("93.184.216.34")]
    monkeypatch.setattr(common_utils, "_resolve_media_ips", lambda host: resolved_ips)
    monkeypatch.setenv("SGLANG_VLM_MEDIA_URL_ALLOWED_SCHEMES", "https")
    monkeypatch.setenv("SGLANG_VLM_MEDIA_URL_FETCH_ENABLED", "true")
    monkeypatch.setenv("SGLANG_VLM_MEDIA_URL_MAX_REDIRECTS", "0")

    def handler(request):
        assert request.url.host == "example.com"
        return httpx.Response(
            200,
            content=b"fake",
            headers={"content-type": "image/png"},
            extensions={"network_stream": DummyNetworkStream(("93.184.216.34", 443))},
        )

    transport = httpx.MockTransport(handler)
    _patch_async_client(monkeypatch, transport)

    target_path = tmp_path / "image"
    result_path = asyncio.run(
        openai_utils._save_url_image_to_path(
            "https://example.com/image.png", str(target_path)
        )
    )

    assert result_path.endswith(".png")
    assert os.path.exists(result_path)


def test_openai_media_url_rejects_peer_ip_mismatch(monkeypatch, tmp_path):
    resolved_ips = [ipaddress.ip_address("93.184.216.34")]
    monkeypatch.setattr(common_utils, "_resolve_media_ips", lambda host: resolved_ips)
    monkeypatch.setenv("SGLANG_VLM_MEDIA_URL_ALLOWED_SCHEMES", "https")
    monkeypatch.setenv("SGLANG_VLM_MEDIA_URL_FETCH_ENABLED", "true")
    monkeypatch.setenv("SGLANG_VLM_MEDIA_URL_MAX_REDIRECTS", "0")

    def handler(request):
        return httpx.Response(
            200,
            content=b"fake",
            headers={"content-type": "image/png"},
            extensions={"network_stream": DummyNetworkStream(("93.184.216.35", 443))},
        )

    transport = httpx.MockTransport(handler)
    _patch_async_client(monkeypatch, transport)

    target_path = tmp_path / "image"
    with pytest.raises(Exception, match="peer IP"):
        asyncio.run(
            openai_utils._save_url_image_to_path(
                "https://example.com/image.png", str(target_path)
            )
        )


def test_openai_media_url_revalidates_redirect_hops(monkeypatch, tmp_path):
    resolved_hosts = []

    def fake_resolve(host):
        resolved_hosts.append(host)
        mapping = {
            "example.com": [ipaddress.ip_address("93.184.216.34")],
            "images.example": [ipaddress.ip_address("93.184.216.35")],
        }
        return mapping.get(host, [])

    monkeypatch.setattr(common_utils, "_resolve_media_ips", fake_resolve)
    monkeypatch.setenv("SGLANG_VLM_MEDIA_URL_ALLOWED_SCHEMES", "https")
    monkeypatch.setenv("SGLANG_VLM_MEDIA_URL_FETCH_ENABLED", "true")
    monkeypatch.setenv("SGLANG_VLM_MEDIA_URL_MAX_REDIRECTS", "1")

    def handler(request):
        if request.url.host == "example.com":
                return httpx.Response(
                    302,
                    headers={"location": "https://images.example/image.png"},
                    extensions={
                        "network_stream": DummyNetworkStream(("93.184.216.34", 443))
                    },
                )
        return httpx.Response(
            200,
            content=b"fake",
            headers={"content-type": "image/png"},
            extensions={"network_stream": DummyNetworkStream(("93.184.216.35", 443))},
        )

    transport = httpx.MockTransport(handler)
    _patch_async_client(monkeypatch, transport)

    target_path = tmp_path / "image"
    result_path = asyncio.run(
        openai_utils._save_url_image_to_path(
            "https://example.com/image.png", str(target_path)
        )
    )

    assert result_path.endswith(".png")
    assert "example.com" in resolved_hosts
    assert "images.example" in resolved_hosts
