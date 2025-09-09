from argparse import Namespace
from typing import Type

from sglang.benchmark.backends.base_client import BaseBackendClient
from sglang.benchmark.backends.sglang_client import SglangClient

BACKEND_MAPPING = {
    "sglang": SglangClient,
    "sglang-native": SglangClient,
}


def get_backend_client(args: Namespace) -> BaseBackendClient:
    backend_class: Type[BaseBackendClient] = BACKEND_MAPPING.get(args.backend)
    if not backend_class:
        raise ValueError(f"Unknown backend: {args.backend}")
    return backend_class(args)


def get_api_url(args: Namespace) -> (str, str):
    base_url = args.base_url or f"http://{args.host}:{args.port}"

    if args.backend in ["sglang", "sglang-native"]:
        api_url = f"{base_url}/generate"
    elif args.backend in ["sglang-oai", "vllm", "lmdeploy"]:
        api_url = f"{base_url}/v1/completions"
    elif args.backend in ["sglang-oai-chat", "vllm-chat", "lmdeploy-chat"]:
        api_url = f"{base_url}/v1/chat/completions"
    elif args.backend == "trt":
        api_url = f"{base_url}/v2/models/ensemble/generate_stream"
    elif args.backend == "gserver":
        api_url = args.base_url if args.base_url else f"{args.host}:{args.port}"
    elif args.backend == "truss":
        api_url = f"{base_url}/v1/models/model:predict"
    else:
        raise ValueError(f"Unknown backend for URL generation: {args.backend}")

    return api_url, base_url
