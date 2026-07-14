# SGLang public APIs

# Bootstrap knob: SGLANG_ENABLE_MINIMAL_INIT=1 skips the public-API imports
# below (torch / transformers / frontend language), keeping `import sglang`
# cheap for orchestrators that only need import-light submodules such as
# sglang.test.ci.* (e.g. test/run_suite.py, which collects tests via AST
# parsing and runs each test file in a subprocess). `sglang.srt.environ` is
# import-light, so reading the flag through it does not defeat the purpose.
from sglang.srt.environ import envs as _envs
from sglang.version import __version__

if not _envs.SGLANG_ENABLE_MINIMAL_INIT.get():
    # Install stubs early for platforms where certain dependencies are unavailable
    # (e.g. macOS/MPS has no triton, and torch.mps lacks Stream / set_device /
    # get_device_properties).  This must run before any downstream imports.
    import platform as _platform
    import sys as _sys

    if _sys.platform == "darwin" and _platform.machine() == "arm64":
        try:
            import torch as _torch

            if _torch.backends.mps.is_available():
                from sglang._triton_stub import install as _install_triton_stub

                _install_triton_stub()
                del _install_triton_stub

                from sglang._mps_stub import install as _install_mps_stub

                _install_mps_stub()
                del _install_mps_stub
            del _torch
        except ImportError:
            pass
    del _platform
    del _sys

    from sglang.srt.utils.hf_transformers_patches import apply_all as _apply_hf_patches

    _apply_hf_patches()
    del _apply_hf_patches

    # Frontend Language APIs
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
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
    from sglang.lang.choices import (
        greedy_token_selection,
        token_length_normalized,
        unconditional_likelihood_normalized,
    )

    # Lazy import some libraries
    from sglang.utils import LazyImport

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

del _envs
