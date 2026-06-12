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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_logged_mismatches: set = set()


def _discover_input_files(input_dir: str) -> Dict[str, str]:
    """Scan input_dir for .pt files. Returns {module_name: file_path}."""
    result = {}
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logger.error(f"[input_loader] Directory not found: {input_dir}")
        return result
    for f in input_path.glob("*.pt"):
        module_name = f.stem
        result[module_name] = str(f)
    return result


def _find_first_tensor(val: Any) -> Optional[torch.Tensor]:
    """Recursively find the first tensor in a nested structure."""
    if isinstance(val, torch.Tensor):
        return val
    elif isinstance(val, (list, tuple)):
        for item in val:
            t = _find_first_tensor(item)
            if t is not None:
                return t
    elif isinstance(val, dict):
        for item in val.values():
            t = _find_first_tensor(item)
            if t is not None:
                return t
    return None


def _align_tensor(val: Any, device: torch.device, dtype: torch.dtype) -> Any:
    """Recursively move tensors to the target device/dtype."""
    if isinstance(val, torch.Tensor):
        return val.to(
            device=device, dtype=dtype if val.is_floating_point() else val.dtype
        )
    elif isinstance(val, list):
        return [_align_tensor(v, device, dtype) for v in val]
    elif isinstance(val, tuple):
        return tuple(_align_tensor(v, device, dtype) for v in val)
    elif isinstance(val, dict):
        return {k: _align_tensor(v, device, dtype) for k, v in val.items()}
    return val


def _validate_shape(
    replacement: torch.Tensor, original: torch.Tensor, module_name: str
) -> bool:
    """Check that replacement shape matches original. Logs once per module."""
    if replacement.shape != original.shape:
        if module_name not in _logged_mismatches:
            logger.error(
                f"[input_loader] Shape mismatch for '{module_name}': "
                f"loaded {tuple(replacement.shape)} vs "
                f"expected {tuple(original.shape)}"
            )
            _logged_mismatches.add(module_name)
        return False
    return True


def _make_input_hook(module_name: str, replacement_path: str) -> Any:
    """Create a forward_pre_hook that replaces the module's input."""
    raw = torch.load(replacement_path, map_location="cpu", weights_only=True)
    logger.info(
        f"[input_loader] Loaded replacement for '{module_name}' "
        f"from {replacement_path}"
    )

    _aligned_cache: Dict[torch.device, Any] = {}

    def hook(module: nn.Module, args: Tuple, kwargs: Dict) -> Tuple[Tuple, Dict]:
        nonlocal _aligned_cache

        # Determine the slot to replace
        original = None
        slot_type = "args"

        if args:
            original = _find_first_tensor(args[0])

        if original is None:
            for kw_name in ("hidden_states", "x", "input", "inputs_embeds"):
                if kw_name in kwargs:
                    original = _find_first_tensor(kwargs[kw_name])
                    if original is not None:
                        slot_type = kw_name
                        break

        if original is None:
            return args, kwargs

        # Get or create aligned replacement
        device = original.device
        if device not in _aligned_cache:
            _aligned_cache[device] = _align_tensor(raw, device, original.dtype)

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
