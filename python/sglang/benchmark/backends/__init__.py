from typing import Any, Dict, Type

from sglang.benchmark.backends.base_client import (
    BaseBackendClient,
    RequestFuncInput,
    RequestFuncOutput,
)
from sglang.benchmark.backends.gserver_client import GServerBackendClient
from sglang.benchmark.backends.oai_chat_client import OpenAIChatBackendClient
from sglang.benchmark.backends.oai_client import OpenAIBackendClient
from sglang.benchmark.backends.oai_embedding_client import (
    OpenAIEmbeddingBackendClient,
)
from sglang.benchmark.backends.profile_client import ProfileBackendClient
from sglang.benchmark.backends.sglang_client import SGLangBackendClient
from sglang.benchmark.backends.trt_client import TRTBackendClient
from sglang.benchmark.backends.truss_client import TrussBackendClient

BACKEND_MAPPING: Dict[str, Type[BaseBackendClient]] = {
    "sglang": SGLangBackendClient,
    "sglang-native": SGLangBackendClient,
    "sglang-oai": OpenAIBackendClient,
    "sglang-oai-chat": OpenAIChatBackendClient,
    "sglang-embedding": OpenAIEmbeddingBackendClient,
    "vllm": OpenAIBackendClient,
    "vllm-chat": OpenAIChatBackendClient,
    "lmdeploy": OpenAIBackendClient,
    "lmdeploy-chat": OpenAIChatBackendClient,
    "trt": TRTBackendClient,
    "gserver": GServerBackendClient,
    "truss": TrussBackendClient,
}


def get_backend_client(backend: str, args: Any) -> BaseBackendClient:
    if backend not in BACKEND_MAPPING:
        raise ValueError(f"Unknown backend: {backend}")
    return BACKEND_MAPPING[backend](args)


async def request_backend(
    backend: str,
    args: Any,
    request_func_input: RequestFuncInput,
    pbar=None,
) -> RequestFuncOutput:
    client = get_backend_client(backend, args)
    return await client.request(request_func_input, pbar=pbar)


async def request_profile(args: Any, api_url: str) -> RequestFuncOutput:
    client = ProfileBackendClient(args)
    return await client.request_profile(api_url)
