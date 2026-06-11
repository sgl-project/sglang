"""
Load fixed module inputs from .pt files and replace module inputs at runtime.

Usage:
    Launch server with --load-input-dir /path/to/input_dir

Directory structure expected:
    /path/to/input_dir/
        module_name_1.pt       # tensor or tuple of tensors
        module_name_2.pt
        ...

Module names use dots as separators (e.g., "model.layers.0.mlp.pt").
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _discover_input_files(input_dir: str) -> Dict[str, str]:
    """Scan input_dir for .pt files. Returns {module_name: file_path}."""
    result = {}
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logger.error(f"[input_loader] Directory not found: {input_dir}")
        return result
    for f in input_path.glob("*.pt"):
        # module name = filename without .pt extension
        module_name = f.stem
        result[module_name] = str(f)
    return result


def _validate_shape(
    replacement: torch.Tensor, original: torch.Tensor, module_name: str
) -> bool:
    """Check that replacement shape matches original. Returns True if valid."""
    if replacement.shape != original.shape:
        logger.error(
            f"[input_loader] Shape mismatch for '{module_name}': "
            f"loaded {tuple(replacement.shape)} vs "
            f"expected {tuple(original.shape)}"
        )
        return False
    return True


def _make_input_hook(
    module_name: str, replacement_path: str
) -> Any:
    """Create a forward_pre_hook that replaces the module's input."""
    # Load once at registration time
    raw = torch.load(replacement_path, map_location="cpu", weights_only=True)
    logger.info(
        f"[input_loader] Loaded replacement for '{module_name}' "
        f"from {replacement_path}"
    )

    # Cache device-aligned versions to avoid repeated .to() calls
    _aligned_cache: Dict[torch.device, Any] = {}

    def hook(module: nn.Module, args: Tuple, kwargs: Dict) -> Tuple[Tuple, Dict]:
        nonlocal _aligned_cache

        # Determine the slot to replace (first positional arg or known kwarg)
        original = None
        slot_type = "args"  # "args" or kwarg name
        slot_idx = 0

        if args and isinstance(args[0], torch.Tensor):
            original = args[0]
        else:
            for kw_name in ("hidden_states", "x", "input", "inputs_embeds"):
                if kw_name in kwargs and isinstance(kwargs[kw_name], torch.Tensor):
                    original = kwargs[kw_name]
                    slot_type = kw_name
                    break

        if original is None:
            return args, kwargs

        # Get or create aligned replacement
        device = original.device
        if device not in _aligned_cache:
            if isinstance(raw, torch.Tensor):
                repl = raw.to(device=device, dtype=original.dtype)
            else:
                repl = raw
            _aligned_cache[device] = repl

        repl = _aligned_cache[device]

        # Shape validation
        if isinstance(repl, torch.Tensor):
            if not _validate_shape(repl, original, module_name):
                return args, kwargs

        # Replace
        if slot_type == "args":
            new_args = (repl,) + args[1:]
            return new_args, kwargs
        else:
            kwargs[slot_type] = repl
            return args, kwargs

    return hook


def register_input_loaders(model: nn.Module, input_dir: str) -> int:
    """Register forward_pre_hooks for all .pt files in input_dir.

    Returns the number of hooks successfully registered.
    """
    from sglang.srt.model_executor.hook_manager import _ensure_pass_tracker

    module_files = _discover_input_files(input_dir)
    if not module_files:
        logger.warning(f"[input_loader] No .pt files found in {input_dir}")
        return 0

    name_to_module = dict(model.named_modules())
    tracker = _ensure_pass_tracker(model)
    registered = 0

    for module_name, file_path in module_files.items():
        if module_name not in name_to_module:
            logger.warning(
                f"[input_loader] Module '{module_name}' not found in model, skipping"
            )
            continue

        target_module = name_to_module[module_name]
        # Attach metadata for introspection
        target_module._forward_hook_target_name = module_name
        target_module._forward_hook_root_model = tracker

        hook = _make_input_hook(module_name, file_path)
        target_module.register_forward_pre_hook(hook, with_kwargs=True)
        registered += 1
        logger.info(f"[input_loader] Registered input loader on '{module_name}'")

    logger.info(
        f"[input_loader] Total: {registered}/{len(module_files)} hooks registered"
    )
    return registered
