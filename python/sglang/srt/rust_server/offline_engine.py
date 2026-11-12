"""Client bridge for offline Engine requests handled by the Rust frontend.

The Rust server runs with the scheduler process. Offline ``Engine`` calls use
its loopback HTTP listener so they traverse the same Rust request pipeline as
HTTP traffic and reach the existing Python scheduler through the established
handoff.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, Callable

import aiohttp


class RustOfflineEngineClient:
    """Send supported offline generation requests to the embedded Rust server."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        session_factory: Callable[..., Any] = aiohttp.ClientSession,
    ):
        if host in ("", "0.0.0.0"):
            host = "127.0.0.1"
        elif host == "::":
            host = "::1"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        self.base_url = f"http://{host}:{port}"
        self._session_factory = session_factory

    async def generate(
        self, body: Mapping[str, Any], *, stream: bool
    ) -> dict[str, Any] | list[dict[str, Any]] | AsyncIterator[dict[str, Any]]:
        """Run a unary request or return an async iterator over SSE frames."""
        if stream:
            return self._stream(body)
        timeout = aiohttp.ClientTimeout(total=None, connect=30)
        async with self._session_factory(timeout=timeout) as session:
            async with session.post(f"{self.base_url}/generate", json=body) as response:
                raw = await response.text()
                value = self._decode_json(raw)
                if response.status >= 400:
                    raise self._request_error(response.status, value)
                if not isinstance(value, (dict, list)):
                    raise RuntimeError(
                        "Rust offline Engine returned an invalid generation response"
                    )
                return value

    async def _stream(self, body: Mapping[str, Any]) -> AsyncIterator[dict[str, Any]]:
        timeout = aiohttp.ClientTimeout(total=None, connect=30)
        async with self._session_factory(timeout=timeout) as session:
            async with session.post(f"{self.base_url}/generate", json=body) as response:
                if response.status >= 400:
                    raw = await response.text()
                    raise self._request_error(response.status, self._decode_json(raw))
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    value = self._decode_json(data)
                    if not isinstance(value, dict):
                        raise RuntimeError(
                            "Rust offline Engine returned an invalid stream frame"
                        )
                    yield value

    @staticmethod
    def _decode_json(raw: str) -> Any:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Rust offline Engine returned a non-JSON response"
            ) from error

    @staticmethod
    def _request_error(status: int, body: Any) -> RuntimeError:
        if isinstance(body, dict):
            message = body.get("message") or body.get("error") or body
        else:
            message = body
        return RuntimeError(f"Rust offline Engine request failed ({status}): {message}")


def generation_request_body(request: Any) -> dict[str, Any]:
    """Convert the first offline Engine generation contract to Rust JSON.

    This slice supports text or token-ID prompts, sampling parameters,
    streaming, request IDs, logprob controls, LoRA paths, and priority. Other
    populated fields fail explicitly until their Rust request and payload
    ownership contracts are migrated.
    """
    supported = (
        "text",
        "input_ids",
        "sampling_params",
        "stream",
        "rid",
        "return_logprob",
        "logprob_start_len",
        "top_logprobs_num",
        "token_ids_logprob",
        "lora_path",
        "priority",
        "cache_salt",
        "routed_dp_rank",
    )
    # Inspect every field so a newly added Engine option cannot be dropped
    # silently. `log_metrics` is internal bookkeeping; the HTTP ingress owns
    # its own request metrics and the field is not part of GenerateRequest.
    ignored = {"log_metrics"}
    active = [
        name
        for name, value in vars(request).items()
        if name not in supported and name not in ignored and _is_non_default(value)
    ]
    if active:
        names = ", ".join(sorted(active))
        raise NotImplementedError(
            "Rust offline Engine generation does not support these options yet: "
            f"{names}"
        )

    body: dict[str, Any] = {}
    for name in supported:
        value = getattr(request, name)
        if value is not None:
            body[name] = value
    return body


def _is_non_default(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (bool, int, float)):
        return bool(value)
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return bool(value)
    return True
