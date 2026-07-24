# SGLang public APIs

# Install stubs early for platforms where certain dependencies are unavailable
# (e.g. macOS/MPS has no triton, and torch.mps lacks Stream / set_device /
# get_device_properties).  This must run before any downstream imports.
import importlib.abc as _importlib_abc
import importlib.machinery as _importlib_machinery
import platform as _platform
import sys as _sys


def _install_mps_stub_if_available(_torch):
    if _torch.backends.mps.is_available():
        from sglang._mps_stub import install as _install_mps_stub

        _install_mps_stub()


class _TorchMpsStubLoader(_importlib_abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        if hasattr(self.loader, "create_module"):
            return self.loader.create_module(spec)
        return None

    def exec_module(self, module):
        self.loader.exec_module(module)
        _install_mps_stub_if_available(module)


class _TorchMpsStubFinder(_importlib_abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "torch":
            return None
        spec = _importlib_machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _TorchMpsStubLoader(spec.loader)
        return spec


if _sys.platform == "darwin" and _platform.machine() == "arm64":
    from sglang._triton_stub import install as _install_triton_stub

    _install_triton_stub()
    del _install_triton_stub

    if "torch" in _sys.modules:
        _install_mps_stub_if_available(_sys.modules["torch"])
    else:
        _sys.meta_path.insert(0, _TorchMpsStubFinder())
del _platform
del _sys

# Frontend Language APIs
from sglang._lazy_import import LazyImport
from sglang.global_config import global_config
from sglang.lang.api import (
    Engine,
    Runtime,
    assistant,
    assistant_begin,
    assistant_end,
    flush_cache,
    function,
    gen,
    gen_int,
    gen_string,
    get_server_info,
    image,
    select,
    separate_reasoning,
    set_default_backend,
    system,
    system_begin,
    system_end,
    user,
    user_begin,
    user_end,
    video,
)
from sglang.lang.choices import (
    greedy_token_selection,
    token_length_normalized,
    unconditional_likelihood_normalized,
)

RuntimeEndpoint = LazyImport("sglang.lang.backend.runtime_endpoint", "RuntimeEndpoint")

# Lazy import some libraries
from sglang.version import __version__

Anthropic = LazyImport("sglang.lang.backend.anthropic", "Anthropic")
Crusoe = LazyImport("sglang.lang.backend.crusoe", "Crusoe")
LiteLLM = LazyImport("sglang.lang.backend.litellm", "LiteLLM")
OpenAI = LazyImport("sglang.lang.backend.openai", "OpenAI")
VertexAI = LazyImport("sglang.lang.backend.vertexai", "VertexAI")

# Runtime Engine APIs
ServerArgs = LazyImport("sglang.srt.server_args", "ServerArgs")
Engine = LazyImport("sglang.srt.entrypoints.engine", "Engine")

__all__ = [
    "Engine",
    "Runtime",
    "assistant",
    "assistant_begin",
    "assistant_end",
    "flush_cache",
    "function",
    "gen",
    "gen_int",
    "gen_string",
    "get_server_info",
    "image",
    "select",
    "separate_reasoning",
    "set_default_backend",
    "system",
    "system_begin",
    "system_end",
    "user",
    "user_begin",
    "user_end",
    "video",
    "RuntimeEndpoint",
    "greedy_token_selection",
    "token_length_normalized",
    "unconditional_likelihood_normalized",
    "ServerArgs",
    "Anthropic",
    "Crusoe",
    "LiteLLM",
    "OpenAI",
    "VertexAI",
    "global_config",
    "__version__",
]
