"""Registry for pluggable TreeCore implementations.

The unified cache constructs its TreeCore through `create_tree_core`, selected
by its per-cache override or SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND (default
"rust"). Rust selections use centralized compatibility fallbacks. To plug in a
custom implementation, register it via `register_tree_core_backend(name, factory)`.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.components import TreeComponent
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
        UnifiedTreeCoreInterface,
    )

TreeCoreFactory = Callable[
    ["CacheInitParams", "dict[ComponentType, TreeComponent]"],
    "UnifiedTreeCoreInterface",
]

_TREE_CORE_REGISTRY: dict[str, TreeCoreFactory] = {}
_RUST_TREE_CORE_MODULE = "sglang.srt.mem_cache.rust_tree_core.mem_cache"
_RUST_TREE_CORE_MANIFEST = (
    Path(__file__).resolve().parents[5] / "rust" / "sglang-radix-tree" / "Cargo.toml"
)
logger = logging.getLogger(__name__)


def _rust_fallback_reason(params: CacheInitParams) -> Optional[str]:
    if params.enable_session_radix_cache:
        return "session-aware caching requires the Python TreeCore"
    if params.tree_components is not None and set(params.tree_components) - {
        ComponentType.FULL,
        ComponentType.SWA,
        ComponentType.MAMBA,
    }:
        return "the configured components require the Python TreeCore"
    if params.component_registry_override:
        return "custom components require the Python TreeCore"
    if sys.platform != "linux":
        return "the Rust TreeCore supports Linux only"
    from sglang.srt.rust_extensions.torch_build import (
        _MAX_SUPPORTED_TORCH,
        _MIN_SUPPORTED_TORCH,
    )

    match = re.match(r"^(\d+)\.(\d+)", str(torch.__version__))
    if match is None or not (
        _MIN_SUPPORTED_TORCH
        <= (int(match.group(1)), int(match.group(2)))
        <= _MAX_SUPPORTED_TORCH
    ):
        return f"PyTorch {torch.__version__} is outside the Rust TreeCore support range"
    allocator = params.token_to_kv_pool_allocator
    device = (
        torch.device(allocator.device) if allocator is not None else torch.device("cpu")
    )
    if device.type not in ("cpu", "cuda"):
        return f"the Rust TreeCore does not support device {device.type}"
    mode = envs.SGLANG_RUST_BUILD_MODE.get()
    if sys.modules.get(_RUST_TREE_CORE_MODULE) is not None or mode not in (
        "auto",
        "never",
        "force",
    ):
        # Preserve the loader's invalid-mode and force-after-import errors.
        return None
    workspace = _RUST_TREE_CORE_MANIFEST.parent.parent
    bundled = importlib.util.find_spec(_RUST_TREE_CORE_MODULE) is not None
    source_checkout = (workspace / "Cargo.toml").is_file()
    if mode != "force" and (mode == "never" or not source_checkout) and bundled:
        return None
    if not _RUST_TREE_CORE_MANIFEST.is_file():
        if bundled:
            return "the requested Rust TreeCore source build has no sources"
        return "this installation contains neither the Rust TreeCore extension nor its sources"
    # Even a cached source build needs these version queries for its fingerprint.
    # Match the loader's commands and cwd so rustup workspace overrides apply.
    for command in (("cargo", "--version", "--verbose"), ("rustc", "-vV")):
        try:
            subprocess.run(
                command,
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return f"the Rust TreeCore toolchain is unavailable ({' '.join(command)})"
    return None


def resolve_tree_core_backend(name: str, params: CacheInitParams) -> str:
    """Resolve known Rust capability gaps before loading a backend.

    Explicit Rust selections use the same compatibility policy as the default.
    Build, import, and runtime failures remain errors when the toolchain is usable.
    """
    if name != "rust":
        return name
    reason = _rust_fallback_reason(params)
    if reason is not None:
        logger.info("Using the Python TreeCore: %s", reason)
        return "python"
    return name


def select_tree_core_backend(params: CacheInitParams) -> str:
    """Resolve the instance override or default through the shared fallback policy."""
    name = (
        params.tree_core_backend
        if params.tree_core_backend is not None
        else envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get()
    )
    return resolve_tree_core_backend(name, params)


def register_tree_core_backend(name: str, factory: TreeCoreFactory) -> None:
    """Register a TreeCore factory under `name`."""
    if not name.strip():
        raise ValueError(
            f"register_tree_core_backend: name must be non-empty, got {name!r}"
        )
    if name in _TREE_CORE_REGISTRY:
        raise ValueError(f"register_tree_core_backend: {name!r} is already registered")
    _TREE_CORE_REGISTRY[name] = factory


def get_tree_core_factory(name: str) -> Optional[TreeCoreFactory]:
    return _TREE_CORE_REGISTRY.get(name)


def registered_tree_core_backends() -> list[str]:
    return list(_TREE_CORE_REGISTRY.keys())


def _python_tree_core_factory(
    params: CacheInitParams, components: dict[ComponentType, TreeComponent]
) -> UnifiedTreeCoreInterface:
    """The pure-Python TreeCore."""
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

    return UnifiedTreeCore(params, components)


def _rust_tree_core_factory(
    params: CacheInitParams, components: dict[ComponentType, TreeComponent]
) -> UnifiedTreeCoreInterface:
    """Load and construct the in-tree Rust TreeCore only when selected."""
    from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore

    return RustUnifiedTreeCore(params)


register_tree_core_backend("python", _python_tree_core_factory)
register_tree_core_backend("rust", _rust_tree_core_factory)


def create_tree_core(
    name: str,
    params: CacheInitParams,
    components: dict[ComponentType, TreeComponent],
) -> UnifiedTreeCoreInterface:
    """Resolve compatibility and construct the registered TreeCore."""
    name = resolve_tree_core_backend(name, params)
    factory = get_tree_core_factory(name)
    if factory is None:
        raise ValueError(
            f"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND={name!r} is not registered. "
            f"Registered backends: {registered_tree_core_backends()}. "
            "External backends must call register_tree_core_backend(...) at import time."
        )
    return factory(params, components)
