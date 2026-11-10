"""The in-tree Rust TreeCore backend; the factory lives in tree_core_registry."""

import os
import sys

from sglang.srt.environ import envs

_MODULE = __name__ + ".mem_cache"


def _load_prebuilt_extension(path: str) -> None:
    """Register the extension at ``path`` under the bundled module's name.

    Runs while this package imports, before any import of the module itself,
    so every later import (the loader in extension.py included) resolves to it.
    """
    import hashlib
    import importlib.util

    # Loading torch first makes its libtorch dependencies resident before dlopen.
    import torch  # noqa: F401

    expected = (envs.SGLANG_RUST_TREE_CORE_EXTENSION_SHA256.get() or "").lower()
    if not expected:
        raise ImportError(
            f"SGLANG_RUST_TREE_CORE_EXTENSION={path} needs "
            "SGLANG_RUST_TREE_CORE_EXTENSION_SHA256, the sha256 of that file"
        )
    with open(path, "rb") as file:
        digest = hashlib.sha256(file.read()).hexdigest()
    if digest != expected:
        raise ImportError(
            f"SGLANG_RUST_TREE_CORE_EXTENSION={path} has sha256 {digest}, not "
            f"the {expected} SGLANG_RUST_TREE_CORE_EXTENSION_SHA256 names"
        )
    spec = importlib.util.spec_from_file_location(_MODULE, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a Rust TreeCore extension from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_MODULE, None)
        raise
    features = module.cache_residency_features()
    # Once per process, on stderr: it can run before logging is configured.
    print(
        f"SGLANG_RUST_TREE_CORE_EXTENSION: loaded the Rust TreeCore from {path} "
        f"(sha256 {digest}, pid {os.getpid()}, residency features "
        f"{','.join(features) or 'none'})",
        file=sys.stderr,
        flush=True,
    )


if envs.SGLANG_RUST_TREE_CORE_EXTENSION.get() and _MODULE not in sys.modules:
    _load_prebuilt_extension(envs.SGLANG_RUST_TREE_CORE_EXTENSION.get())
