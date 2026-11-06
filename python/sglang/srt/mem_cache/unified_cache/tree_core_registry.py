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
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv.plan import TREE_STATE, TREE_TOKEN, TREE_WINDOW
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

# The slots the native core stores, and the family it keeps each by.
NATIVE_SLOTS = {"FULL": TREE_TOKEN, "SWA": TREE_WINDOW, "MAMBA": TREE_STATE}


def _rust_fallback_reason(params: CacheInitParams) -> Optional[str]:
    return _rust_config_fallback_reason(params) or _rust_runtime_fallback_reason(params)


def _rust_config_fallback_reason(params: CacheInitParams) -> Optional[str]:
    """Gaps from the cache configuration, known once the factory fills the slots."""
    if params.enable_session_radix_cache:
        return "session-aware caching requires the Python TreeCore"
    if params.tree_slots is not None and any(
        NATIVE_SLOTS.get(slot.component) != slot.family for slot in params.tree_slots
    ):
        return "the configured components require the Python TreeCore"
    if params.component_registry_override:
        return "custom components require the Python TreeCore"
    return None


def _rust_runtime_fallback_reason(params: CacheInitParams) -> Optional[str]:
    """Gaps from the rank's runtime; the toolchain probe can fail on one rank only."""
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


def _requested_tree_core_backend(params: CacheInitParams) -> str:
    if params.tree_core_backend is not None:
        return params.tree_core_backend
    return envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get()


def _tree_peer_groups(params: CacheInitParams) -> list:
    """The groups whose ranks share one prefix tree (those the HiCache syncs use)."""
    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized()):
        return []
    groups = []
    for group in (
        params.tp_cache_group,
        params.attn_tp_cache_group,
        params.attn_cp_cache_group,
        params.pp_cache_group,
    ):
        if (
            group is not None
            and all(group is not seen for seen in groups)
            and dist.get_world_size(group=group) > 1
        ):
            groups.append(group)
    return groups


def _all_reduce_min_across_tree_peers(
    values: list[int], params: CacheInitParams
) -> list[int]:
    state = torch.tensor(values, dtype=torch.int64)
    for group in _tree_peer_groups(params):
        torch.distributed.all_reduce(
            state, op=torch.distributed.ReduceOp.MIN, group=group
        )
    return state.tolist()


# Tag of a rank whose cache has no TreeCore, e.g. PureSWARadixCache on a PP stage
# holding only SWA layers. It is above any crc32, so in
# check_tree_core_backend_matches such a rank takes part without winning a MIN.
_NO_TREE_CORE_TAG = 1 << 62


def agree_tree_core_backend(params: CacheInitParams) -> None:
    """Agree the backend before the cache factory; collective on every rank.

    Ranks probe the Rust toolchain on their own, and concurrent probes racing to
    install it can fail on some ranks only. Mixed cores break the HiCache syncs
    that assume identical trees, so one failed probe moves every rank sharing
    the tree to Python. Ranks requesting different backends fail here.
    """
    requested = _requested_tree_core_backend(params)
    reason = _rust_runtime_fallback_reason(params) if requested == "rust" else None
    if reason is not None:
        logger.info("Using the Python TreeCore: %s", reason)
    tag = zlib.crc32(requested.encode())
    runtime_ok, min_tag, neg_max_tag = _all_reduce_min_across_tree_peers(
        [int(reason is None), tag, -tag], params
    )
    if min_tag != -neg_max_tag:
        raise ValueError(
            "Ranks sharing a prefix tree request different TreeCore backends "
            f"(this rank: {requested!r})"
        )
    if requested == "rust":
        if reason is None and not runtime_ok:
            logger.info(
                "Using the Python TreeCore: another rank sharing this prefix tree "
                "cannot use the Rust TreeCore"
            )
        params.peers_can_use_rust_tree_core = bool(runtime_ok)


def check_tree_core_backend_matches(params: CacheInitParams, tree_cache) -> None:
    """Check the built backends after the cache factory; collective on every rank.

    The configuration fallbacks are decided per cache, so ranks with differing
    cache configurations could still mix cores: fail startup instead.
    """
    backend = getattr(tree_cache, "tree_core_backend", None)
    tag = _NO_TREE_CORE_TAG if backend is None else zlib.crc32(backend.encode())
    min_tag, neg_max_tag = _all_reduce_min_across_tree_peers(
        [tag, _NO_TREE_CORE_TAG if backend is None else -tag], params
    )
    if min_tag != _NO_TREE_CORE_TAG and min_tag != -neg_max_tag:
        raise RuntimeError(
            "Ranks sharing a prefix tree built different TreeCore backends "
            f"(this rank: {backend!r}); make their cache configuration match or "
            "set SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python"
        )


def select_tree_core_backend(params: CacheInitParams) -> str:
    """Resolve the instance override or default through the shared fallback
    policy, reusing the runtime verdict agree_tree_core_backend recorded."""
    name = _requested_tree_core_backend(params)
    if name != "rust" or params.peers_can_use_rust_tree_core is None:
        return resolve_tree_core_backend(name, params)
    reason = _rust_config_fallback_reason(params)
    if reason is not None:
        logger.info("Using the Python TreeCore: %s", reason)
        return "python"
    return "rust" if params.peers_can_use_rust_tree_core else "python"


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
    *,
    resolve: bool = True,
) -> UnifiedTreeCoreInterface:
    """Resolve compatibility (unless the caller already did) and construct the
    registered TreeCore."""
    if resolve:
        name = resolve_tree_core_backend(name, params)
    factory = get_tree_core_factory(name)
    if factory is None:
        raise ValueError(
            f"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND={name!r} is not registered. "
            f"Registered backends: {registered_tree_core_backends()}. "
            "External backends must call register_tree_core_backend(...) at import time."
        )
    return factory(params, components)
