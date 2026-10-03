"""Select SGLang-vendored or upstream AITER Kimi-K3 FlyDSL operators."""

import importlib
import os


def load_module(local_module: str, aiter_module: str):
    mode = os.environ.get("SGLANG_K3_FLYDSL_SOURCE", "auto").lower()
    if mode not in ("auto", "sglang", "aiter"):
        raise ValueError(
            "SGLANG_K3_FLYDSL_SOURCE must be one of auto, sglang, or aiter"
        )

    candidates = (
        ((local_module, "sglang"), (aiter_module, "aiter"))
        if mode in ("auto", "sglang")
        else ((aiter_module, "aiter"),)
    )
    errors = []
    for module_name, source in candidates:
        if mode == "sglang" and source != "sglang":
            continue
        try:
            return importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as error:
            errors.append(f"{source}: {error}")
    raise ImportError("Kimi-K3 FlyDSL source unavailable: " + "; ".join(errors))
