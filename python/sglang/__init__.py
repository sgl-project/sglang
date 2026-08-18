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
# before any downstream import can initialize AITER_CONFIGS.
import importlib.util as _importlib_util
import os as _os
from pathlib import Path as _Path

if (
    _os.environ.get("SGLANG_K3_AITER_M16384_PROFILE", "0").lower()
    in ("1", "true")
    and "AITER_CONFIG_GEMM_BF16" not in _os.environ
):
    _aiter_spec = _importlib_util.find_spec("aiter")
    if _aiter_spec is not None and _aiter_spec.origin is not None:
        _aiter_root = _Path(_aiter_spec.origin).resolve().parent
        _base = _aiter_root / "configs" / "bf16_tuned_gemm.csv"
        _model_configs = sorted(
            (_aiter_root / "configs" / "model_configs").glob(
                "*bf16_tuned_gemm*.csv"
            )
        )
        _profile = (
            _Path(__file__).resolve().parent
            / "kernels"
            / "ops"
            / "kimi_k3"
            / "configs"
            / "kimik3_m16384_profile.csv"
        )
        _paths = [_base, *_model_configs, _profile]
        if all(_path.is_file() for _path in _paths):
            _os.environ["AITER_CONFIG_GEMM_BF16"] = _os.pathsep.join(
                map(str, _paths)
            )
del _importlib_util
del _os
del _Path

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
