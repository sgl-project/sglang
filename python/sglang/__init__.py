"""SGLang public API."""

import os as _os
import platform as _platform
import sys as _sys

# sglang.srt.environ must run before the rest of this file's imports
# (hf_transformers_patches, lang.api, ...), which pull in torch and
# FlashInfer: those claim these cache dirs early, and the first value set is
# the one that sticks. Safe here -- environ has no heavy dependency (no torch).
from sglang.srt.environ import envs as _envs
from sglang.srt.environ import (
    redirect_third_party_caches as _redirect_third_party_caches,
)

_redirect_third_party_caches()

if _sys.platform == "darwin" and _platform.machine() == "arm64":
    from sglang._platform_stubs import install_platform_stubs as _install_platform_stubs

    _install_platform_stubs()

from sglang.srt.utils.hf_transformers_patches import apply_all as _apply_hf_patches

# The transformers compatibility patches must be in place before transformers
# is *used*, not before sglang is imported. Applying them eagerly costs ~2 s per
# process (it imports torch and transformers, and transformers.masking_utils
# pulls in torch._dynamo). The hook below applies them the moment
# `import transformers` completes, wherever that happens (or right away if it
# already has).


def _install_hf_patch_hook():
    import importlib.abc as _abc

    if "transformers" in _sys.modules:
        _apply_hf_patches()
        return

    class _Hook(_abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "transformers":
                return None
            for finder in _sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is None or spec.loader is None:
                    continue
                loader = spec.loader
                orig_exec = loader.exec_module

                def exec_module(module, _orig=orig_exec):
                    _orig(module)
                    try:
                        _sys.meta_path.remove(self)
                    except ValueError:
                        pass
                    _apply_hf_patches()

                loader.exec_module = exec_module
                return spec
            return None

    _sys.meta_path.insert(0, _Hook())


# Default: eager, exactly as before. With SGLANG_PRESPAWN_WORKERS=1 the
# launcher's import path up to the pre-spawn point must stay free of torch and
# transformers, so the patches are applied when transformers is imported and
# the frontend API resolves on first attribute access (PEP 562, below).
_LAZY_INIT = _envs.SGLANG_PRESPAWN_WORKERS.get()
if _LAZY_INIT:
    _install_hf_patch_hook()
else:
    _apply_hf_patches()

    from sglang.lang.api import (
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
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
    from sglang.lang.choices import (
        greedy_token_selection,
        token_length_normalized,
        unconditional_likelihood_normalized,
    )
    from sglang.lang.global_config import global_config

# Frontend language API (sglang.lang.*) for the lazy mode: resolved on first
# attribute access so that server-side processes do not pay for
# aiohttp/requests/the interpreter. In the eager mode the names are already
# bound above and __getattr__ is never consulted for them.
_LAZY_ATTRS = {
    **{
        name: ("sglang.lang.api", name)
        for name in (
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
        )
    },
    "RuntimeEndpoint": ("sglang.lang.backend.runtime_endpoint", "RuntimeEndpoint"),
    "greedy_token_selection": ("sglang.lang.choices", "greedy_token_selection"),
    "token_length_normalized": ("sglang.lang.choices", "token_length_normalized"),
    "unconditional_likelihood_normalized": (
        "sglang.lang.choices",
        "unconditional_likelihood_normalized",
    ),
    "global_config": ("sglang.lang.global_config", "global_config"),
}


def __getattr__(name):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module 'sglang' has no attribute {name!r}")
    import importlib as _importlib

    value = getattr(_importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


# Lazy backend clients
from sglang.utils import LazyImport
from sglang.version import __version__

Anthropic = LazyImport("sglang.lang.backend.anthropic", "Anthropic")
Crusoe = LazyImport("sglang.lang.backend.crusoe", "Crusoe")
LiteLLM = LazyImport("sglang.lang.backend.litellm", "LiteLLM")
OpenAI = LazyImport("sglang.lang.backend.openai", "OpenAI")
VertexAI = LazyImport("sglang.lang.backend.vertexai", "VertexAI")

# Runtime API
ServerArgs = LazyImport("sglang.srt.server_args", "ServerArgs")
Engine = LazyImport("sglang.srt.entrypoints.engine", "Engine")

__all__ = [
    "Anthropic",
    "Crusoe",
    "Engine",
    "LiteLLM",
    "OpenAI",
    "Runtime",
    "RuntimeEndpoint",
    "ServerArgs",
    "VertexAI",
    "__version__",
    "assistant",
    "assistant_begin",
    "assistant_end",
    "flush_cache",
    "function",
    "gen",
    "gen_int",
    "gen_string",
    "get_server_info",
    "global_config",
    "greedy_token_selection",
    "image",
    "select",
    "separate_reasoning",
    "set_default_backend",
    "system",
    "system_begin",
    "system_end",
    "token_length_normalized",
    "unconditional_likelihood_normalized",
    "user",
    "user_begin",
    "user_end",
    "video",
]
