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
    system_begin,
    system_end,
    user,
    user_begin,
    user_end,
    video,
)

# SGLang DSL APIs
__all__ = [
    "Runtime",
    "assistant",
    "assistant_begin",
    "assistant_end",
    "flush_cache",
    "function",
    "gen",
    "gen_int",
    "gen_string",
    "get_server_args",
    "image",
    "select",
    "set_default_backend",
    "system",
    "system_begin",
    "system_end",
    "user",
    "user_begin",
    "user_end",
    "video",
]

# Global Configurations
from sglang.global_config import global_config

__all__ += ["global_config"]

from sglang.version import __version__

__all__ += ["__version__"]

# SGL Backends
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.utils import LazyImport

Anthropic = LazyImport("sglang.lang.backend.anthropic", "Anthropic")
LiteLLM = LazyImport("sglang.lang.backend.litellm", "LiteLLM")
OpenAI = LazyImport("sglang.lang.backend.openai", "OpenAI")
VertexAI = LazyImport("sglang.lang.backend.vertexai", "VertexAI")

__all__ += ["Anthropic", "LiteLLM", "OpenAI", "VertexAI", "RuntimeEndpoint"]
