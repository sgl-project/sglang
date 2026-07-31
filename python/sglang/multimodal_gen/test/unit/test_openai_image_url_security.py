import base64
from pathlib import Path

import anyio
import httpx
import pytest

from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils

_PUBLIC_IP = "93.184.216.34"
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAusB9WlUK0sAAAAASUVORK5CYII="
)


class _ChunkedBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"12345"
        yield b"6789"


def _install_mock_client(
    monkeypatch: pytest.MonkeyPatch,
    handler,
) -> None:
    async_client = httpx.AsyncClient

    def create_client(**kwargs):
        return async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(openai_utils.httpx, "AsyncClient", create_client)


def _install_resolver(
    monkeypatch: pytest.MonkeyPatch,
    addresses_by_host: dict[str, tuple[str, ...]],
) -> None:
    async def resolve(host: str, port: int) -> tuple[str, ...]:
        del port
        return addresses_by_host[host]

    monkeypatch.setattr(
        openai_utils,
        "_resolve_hostname_addresses",
        resolve,
        raising=False,
    )


@pytest.mark.parametrize(
    "image_url",
    [
        "http://127.0.0.1/input.png",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/input.png",
        "http://[fc00::1]/input.png",
    ],
)
def test_remote_image_rejects_non_public_ip_literals(
    image_url: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="public"):
        anyio.run(
            openai_utils._save_url_image_to_path,
            image_url,
            str(tmp_path / "input"),
        )


def test_remote_image_rejects_dns_resolution_to_private_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_resolver(monkeypatch, {"images.example": ("10.0.0.8",)})
    _install_mock_client(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=_PNG_BYTES,
            request=request,
        ),
    )

    with pytest.raises(ValueError, match="public"):
        anyio.run(
            openai_utils._save_url_image_to_path,
            "https://images.example/input.png",
            str(tmp_path / "input"),
        )


def test_remote_image_revalidates_redirect_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_resolver(
        monkeypatch,
        {
            "images.example": (_PUBLIC_IP,),
            "internal.example": ("192.168.1.10",),
        },
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "images.example":
            return httpx.Response(
                302,
                headers={"location": "http://internal.example/secret.png"},
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=_PNG_BYTES,
            request=request,
        )

    _install_mock_client(monkeypatch, handler)

    with pytest.raises(ValueError, match="public"):
        anyio.run(
            openai_utils._save_url_image_to_path,
            "https://images.example/input.png",
            str(tmp_path / "input"),
        )
    assert len(requests) == 1


def test_remote_image_rejects_oversized_content_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_utils, "_MAX_IMAGE_INPUT_BYTES", 8, raising=False)
    _install_resolver(monkeypatch, {"images.example": (_PUBLIC_IP,)})
    _install_mock_client(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            headers={"content-type": "image/png", "content-length": "9"},
            content=b"123456789",
            request=request,
        ),
    )

    with pytest.raises(ValueError, match="exceeds"):
        anyio.run(
            openai_utils._save_url_image_to_path,
            "https://images.example/input.png",
            str(tmp_path / "input"),
        )


def test_remote_image_enforces_streamed_size_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_utils, "_MAX_IMAGE_INPUT_BYTES", 8, raising=False)
    _install_resolver(monkeypatch, {"images.example": (_PUBLIC_IP,)})
    _install_mock_client(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            headers={"content-type": "image/png"},
            stream=_ChunkedBody(),
            request=request,
        ),
    )

    with pytest.raises(ValueError, match="exceeds"):
        anyio.run(
            openai_utils._save_url_image_to_path,
            "https://images.example/input.png",
            str(tmp_path / "input"),
        )


def test_remote_image_downloads_public_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_resolver(monkeypatch, {"images.example": (_PUBLIC_IP,)})
    _install_mock_client(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=_PNG_BYTES,
            request=request,
        ),
    )

    output_path = anyio.run(
        openai_utils._save_url_image_to_path,
        "https://images.example/input",
        str(tmp_path / "input"),
    )

    assert output_path == str(tmp_path / "input.png")
    assert Path(output_path).read_bytes() == _PNG_BYTES


def test_data_image_enforces_decoded_size_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_utils, "_MAX_IMAGE_INPUT_BYTES", 8, raising=False)
    image_url = f"data:image/png;base64,{base64.b64encode(b'123456789').decode()}"

    with pytest.raises(ValueError, match="exceeds"):
        anyio.run(
            openai_utils._maybe_url_image,
            image_url,
            str(tmp_path / "input"),
        )
