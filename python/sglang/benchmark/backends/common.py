import os
from typing import Any, Dict

import aiohttp

from sglang.benchmark.utils import parse_custom_headers

_ROUTING_KEY_HEADER = "X-SMG-Routing-Key"


def _create_bench_client_session():
    # When the pressure is big, the read buffer could be full before aio thread read
    # the content. We increase the read_bufsize from 64K to 10M.
    # Define constants for timeout and buffer size for clarity and maintainability
    BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
    BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2  # 10 MB

    aiohttp_timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
    return aiohttp.ClientSession(
        timeout=aiohttp_timeout, read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES
    )


def get_auth_headers() -> Dict[str, str]:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        return {"Authorization": f"Bearer {openai_api_key}"}

    api_key = os.environ.get("API_KEY")
    if api_key:
        return {"Authorization": f"{api_key}"}

    return {}


def get_request_headers(args: Any) -> Dict[str, str]:
    headers = get_auth_headers()
    if h := getattr(args, "header", None):
        headers.update(parse_custom_headers(h))
    return headers
