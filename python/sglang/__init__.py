"""SGLang public API."""

import platform as _platform
import sys as _sys

# sglang.srt.environ must run before the rest of this file's imports
# (hf_transformers_patches, lang.api, ...), which pull in torch and
# FlashInfer: those claim these cache dirs early, and the first value set is
# the one that sticks. Safe here -- environ has no heavy dependency (no torch).
from sglang.srt.environ import (
    redirect_third_party_caches as _redirect_third_party_caches,
)

_redirect_third_party_caches()

# Kimi-K3 may opt into an SGLang-owned AITER tuning profile. Configure it
# before any downstream import can initialize AITER_CONFIGS. Torch stays in
# the ROCm module so CUDA imports of sglang do not pay for it here.
import os as _os

if (
    _os.path.isdir("/opt/rocm")
    or _os.environ.get("ROCM_PATH")
    or "HIP_VISIBLE_DEVICES" in _os.environ
):
    from sglang.srt.models.kimi_k3_rocm_bootstrap import (
        maybe_set_aiter_m16384_profile as _maybe_set_aiter_m16384_profile,
    )

    _maybe_set_aiter_m16384_profile()
    del _maybe_set_aiter_m16384_profile
del _os

if _sys.platform == "darwin" and _platform.machine() == "arm64":
    from sglang._platform_stubs import install_platform_stubs as _install_platform_stubs

    _install_platform_stubs()

from sglang.srt.utils.hf_transformers_patches import apply_all as _apply_hf_patches

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
