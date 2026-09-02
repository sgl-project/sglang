# SPDX-License-Identifier: Apache-2.0
import asyncio
from typing import Any, Optional

import aiohttp
import msgspec

from sglang.srt.environ import envs
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


class ExaSearchConfig(msgspec.Struct, frozen=True, kw_only=True):
    num_results: int = EXA_DEFAULT_NUM_RESULTS
    search_type: str = EXA_DEFAULT_SEARCH_TYPE
    include_highlights: bool = EXA_DEFAULT_HIGHLIGHTS

    @classmethod
    def from_env(cls) -> "ExaSearchConfig":
        # EnvField handles type parsing (int/bool) and falls back to its own
        # default on a parse error; here we only enforce the semantic bounds
        # that the generic descriptors cannot express.
        num_results = envs.SGLANG_EXA_NUM_RESULTS.get()
        if not 1 <= num_results <= 100:
            print_warning_once(
                f"Ignoring invalid SGLANG_EXA_NUM_RESULTS={num_results!r}; "
                f"expected a value from 1 to 100."
            )
            num_results = EXA_DEFAULT_NUM_RESULTS

        search_type = envs.SGLANG_EXA_SEARCH_TYPE.get()
        if search_type not in _SEARCH_TYPES:
            print_warning_once(
                f"Ignoring invalid SGLANG_EXA_SEARCH_TYPE={search_type!r}; "
                f"expected one of {', '.join(sorted(_SEARCH_TYPES))}."
            )
            search_type = EXA_DEFAULT_SEARCH_TYPE

        return cls(
            num_results=num_results,
            search_type=search_type,
            include_highlights=envs.SGLANG_EXA_INCLUDE_HIGHLIGHTS.get(),
        )


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
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

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

    async def _get_session(self) -> aiohttp.ClientSession:
        # Reuse a single session across requests so the connection pool and
        # DNS cache survive; aiohttp.ClientSession is intended to be long-lived.
        if self._session is None:
            async with self._session_lock:
                if self._session is None:
                    self._session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    )
        return self._session

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = await self._get_session()
        url = f"{self.base_url}{path}"
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
