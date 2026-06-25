# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp

from sglang.srt.utils import print_warning_once

EXA_API_BASE_URL = "https://api.exa.ai"
EXA_INTEGRATION_HEADER = "x-exa-integration"
EXA_INTEGRATION_NAME = "sglang"
EXA_DEFAULT_NUM_RESULTS = 10
EXA_DEFAULT_SEARCH_TYPE = "auto"
EXA_DEFAULT_HIGHLIGHTS = True

_SEARCH_TYPES = {"instant", "fast", "auto", "deep-lite", "deep", "deep-reasoning"}


class ExaClientError(RuntimeError):
    """Raised when an Exa API request fails."""


@dataclass(frozen=True)
class ExaSearchConfig:
    num_results: int = EXA_DEFAULT_NUM_RESULTS
    search_type: str = EXA_DEFAULT_SEARCH_TYPE
    include_highlights: bool = EXA_DEFAULT_HIGHLIGHTS

    @classmethod
    def from_env(cls) -> "ExaSearchConfig":
        return cls(
            num_results=_get_int_env("SGLANG_EXA_NUM_RESULTS", EXA_DEFAULT_NUM_RESULTS),
            search_type=_get_search_type_env(
                "SGLANG_EXA_SEARCH_TYPE", EXA_DEFAULT_SEARCH_TYPE
            ),
            include_highlights=_get_bool_env(
                "SGLANG_EXA_INCLUDE_HIGHLIGHTS", EXA_DEFAULT_HIGHLIGHTS
            ),
        )


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False

    print_warning_once(
        f"Ignoring invalid {name}={value!r}; expected a boolean value."
    )
    return default


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError:
        print_warning_once(f"Ignoring invalid {name}={value!r}; expected an integer.")
        return default

    if not 1 <= parsed <= 100:
        print_warning_once(
            f"Ignoring invalid {name}={value!r}; expected a value from 1 to 100."
        )
        return default
    return parsed


def _get_search_type_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip()
    if normalized in _SEARCH_TYPES:
        return normalized

    print_warning_once(
        f"Ignoring invalid {name}={value!r}; expected one of "
        f"{', '.join(sorted(_SEARCH_TYPES))}."
    )
    return default


class ExaClient:
    def __init__(
        self,
        api_key: str,
        *,
        config: Optional[ExaSearchConfig] = None,
        base_url: str = EXA_API_BASE_URL,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.config = config or ExaSearchConfig()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            EXA_INTEGRATION_HEADER: EXA_INTEGRATION_NAME,
        }

    def _search_payload(self, query: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "numResults": self.config.num_results,
            "type": self.config.search_type,
        }
        if self.config.include_highlights:
            payload["contents"] = {"highlights": True}
        return payload

    def _contents_payload(self, urls: list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {"urls": urls, "text": True}
        if self.config.include_highlights:
            payload["highlights"] = True
        return payload

    async def search(self, query: str) -> dict[str, Any]:
        return await self._post("/search", self._search_payload(query))

    async def contents(self, urls: list[str]) -> dict[str, Any]:
        return await self._post("/contents", self._contents_payload(urls))

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        url = f"{self.base_url}{path}"
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                json=payload,
                headers=self._headers(),
            ) as response:
                response_text = await response.text()
                if response.status >= 400:
                    raise ExaClientError(
                        f"Exa API request failed with status {response.status}: "
                        f"{response_text}"
                    )
                try:
                    return await response.json()
                except Exception as exc:
                    raise ExaClientError(
                        f"Failed to decode Exa API response: {response_text}"
                    ) from exc

