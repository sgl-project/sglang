import importlib


class LazyImport:
    def __init__(self, module_name, class_name):
        self.module_name = module_name
        self.class_name = class_name
        self._module = None

    def _load(self):
        if self._module is None:
            module = importlib.import_module(self.module_name)
            self._module = getattr(module, self.class_name)
        return self._module

    def __getattr__(self, name):
        module = self._load()
        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        module = self._load()
        return module(*args, **kwargs)


# SGL API Components
from sglang.api import (
    Runtime,
    assistant,
    assistant_begin,
    assistant_end,
    flush_cache,
    function,
    gen,
    gen_int,
    gen_string,
    get_server_args,
    image,
    select,
    set_default_backend,
    system,
    user,
    user_begin,
    user_end,
    video,
)

# Global Configurations
from sglang.global_config import global_config

# SGL Backends
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

Anthropic = LazyImport("sglang.lang.backend.anthropic", "Anthropic")
LiteLLM = LazyImport("sglang.lang.backend.litellm", "LiteLLM")
OpenAI = LazyImport("sglang.lang.backend.openai", "OpenAI")
VertexAI = LazyImport("sglang.lang.backend.vertexai", "VertexAI")

from .version import __version__

# public APIs management
__all__ = [
    "global_config",
    "Anthropic",
    "LiteLLM",
    "OpenAI",
    "RuntimeEndpoint",
    "VertexAI",
    "function",
    "Runtime",
    "set_default_backend",
    "flush_cache",
    "get_server_args",
    "gen",
    "gen_int",
    "gen_string",
    "image",
    "video",
    "select",
    "system",
    "user",
    "assistant",
    "user_begin",
    "user_end",
    "assistant_begin",
    "assistant_end",
]
