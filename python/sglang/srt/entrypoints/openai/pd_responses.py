"""Route expanded native Responses turns through a trusted generation router."""

import asyncio
import copy
import dataclasses
import json
import uuid

import aiohttp

from sglang.srt.utils import ImageData, VideoData


class PDResponsesError(ValueError):
    """Keep a worker failure distinct from invalid client input."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def _media_reference_json(value):
    """Encode API media references, preserving their preprocessing options."""
    if isinstance(value, (ImageData, VideoData)):
        return dataclasses.asdict(value)
    if isinstance(value, list):
        return [_media_reference_json(item) for item in value]
    return value


async def _routed_turn_stream(request, router_url):
    """Send an expanded Responses turn through the ordinary PD generation router.

    The frontend retains API state; the selected P/D pair owns generation.
    Use an internal streaming leg even for nonstreaming Responses so cancellation
    closes the worker streams promptly. The frontend still formats the public API.
    """
    payload = {
        field.name: copy.deepcopy(getattr(request, field.name))
        for field in dataclasses.fields(request)
        if field.init
    }
    payload["rid"] = f"{request.rid or 'responses'}-turn-{uuid.uuid4().hex}"
    payload["stream"] = True
    payload["background"] = False
    for name in ("bootstrap_host", "bootstrap_port", "bootstrap_room"):
        payload.pop(name, None)
    for name in ("image_data", "video_data"):
        if name in payload:
            payload[name] = _media_reference_json(payload[name])
    # No request bodies are logged or persisted by this transport.
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=30)
    ) as client:
        try:
            async with client.post(
                router_url.rstrip("/") + "/generate", json=payload
            ) as response:
                if response.status >= 400:
                    raise PDResponsesError(
                        f"PD generation router returned HTTP {response.status}",
                        response.status,
                    )
                pending = b""
                last_output = None
                async for chunk in response.content.iter_chunked(65536):
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        line = line.rstrip(b"\r")
                        if not line.startswith(b"data: "):
                            continue
                        data = line[6:]
                        if data == b"[DONE]":
                            if last_output is None:
                                raise PDResponsesError(
                                    "PD generation returned no output", 502
                                )
                            if not request.stream:
                                yield last_output
                            return
                        output = json.loads(data)
                        if "error" in output:
                            raise PDResponsesError("PD generation stream failed", 502)
                        last_output = output
                        if request.stream:
                            yield output
                raise PDResponsesError("PD generation stream ended without DONE", 502)
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise PDResponsesError(
                "PD generation router transport failed", 502
            ) from exc


async def routed_turn(request, router_url, raw_request=None):
    """Keep frontend disconnection attached to the routed generation lifetime."""
    consumer = asyncio.current_task()

    async def watch_disconnect():
        while True:
            if await raw_request.is_disconnected():
                consumer.cancel()
                return
            await asyncio.sleep(0.25)

    watcher = (
        asyncio.create_task(watch_disconnect())
        if raw_request is not None and not getattr(request, "background", False)
        else None
    )
    stream = _routed_turn_stream(request, router_url)
    try:
        async for output in stream:
            yield output
    finally:
        try:
            await stream.aclose()
        finally:
            if watcher is not None:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
