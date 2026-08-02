# SPDX-License-Identifier: Apache-2.0
# Module helpers for the MiniMax H3 visual VAE (inference-only bundle).
import os


def _env_flag(name, default="0"):
    value = os.environ.get(name, default)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _env_optional_bool(name, default=""):
    value = str(os.environ.get(name, default)).strip().lower()
    if value in ("", "default", "auto", "none", "unset"):
        return None
    return value not in ("0", "false", "no", "off", "disabled")


def _vit_torch_compile_kwargs(prefix):
    kwargs = {}
    backend = os.environ.get(f"{prefix}_BACKEND", "inductor").strip()
    mode = os.environ.get(f"{prefix}_MODE", "reduce-overhead").strip()
    if backend and backend.lower() not in ("default", "none"):
        kwargs["backend"] = backend
    if mode and mode.lower() not in ("default", "none"):
        kwargs["mode"] = mode
    kwargs["fullgraph"] = _env_flag(f"{prefix}_FULLGRAPH", "0")
    dynamic = _env_optional_bool(f"{prefix}_DYNAMIC")
    if dynamic is not None:
        kwargs["dynamic"] = dynamic
    return kwargs
