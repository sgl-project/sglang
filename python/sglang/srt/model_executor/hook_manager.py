import fnmatch
import importlib
import logging
from typing import Any, Callable, List, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


def _ensure_pass_tracker(model: nn.Module) -> nn.Module:
    """Install a forward-pass counter on the tracker module (model.model or root).

    Returns the tracker module. Idempotent — safe to call multiple times.
    """
    tracker = None
    for name, mod in model.named_modules():
        if name == "model":
            tracker = mod
            break
    if tracker is None:
        tracker = model

    if not hasattr(tracker, "_worker_forward_pass_id"):
        tracker._worker_forward_pass_id = -1

        def _pass_tracker(_module, _args, _kwargs):
            _module._worker_forward_pass_id += 1
            return _args, _kwargs

        tracker.register_forward_pre_hook(_pass_tracker, with_kwargs=True)
        logger.info("Installed forward-pass tracker on %s", type(tracker).__name__)

    return tracker


def register_forward_hooks(model: nn.Module, hook_specs: List[dict[str, Any]]) -> None:
    """
    hook_specs is a list of dicts from server_args.forward_hooks.
    Attaches forward hooks to the matching modules.
    """
    name_to_module = dict(model.named_modules())

    for spec in hook_specs:
        spec_name = spec.get("name", "")
        target_patterns = spec.get("target_modules", [])
        if not target_patterns:
            logger.warning(f"Hook spec '{spec_name}' has no 'target_modules', skipping")
            continue

        hook_factory_path = spec.get("hook_factory")
        if not hook_factory_path:
            logger.warning(f"Hook spec '{spec_name}' has no 'hook_factory', skipping")
            continue

        config = spec.get("config") or {}
        hook_factory = resolve_callable(hook_factory_path)

        hook = hook_factory(config) if hook_factory else None
        if hook is None:
            logger.warning(
                f"Hook factory '{hook_factory_path}' for spec '{spec_name}' "
                "returned None, not registering any hook"
            )
            continue

        # Resolve patterns like "model.layers.*.mlp"
        matched = []
        for name, module in name_to_module.items():
            if any(fnmatch.fnmatch(name, pattern) for pattern in target_patterns):
                matched.append((name, module))

        if not matched:
            logger.warning(
                f"No modules matched hook spec '{spec_name}' "
                f"patterns={target_patterns}"
            )
            continue

        hook_type = spec.get("hook_type", "forward")
        with_kwargs = spec.get("with_kwargs", False)
        for module_name, module in matched:
            if hook_type == "forward_pre":
                module.register_forward_pre_hook(hook, with_kwargs=with_kwargs)
            else:
                module.register_forward_hook(hook)
            logger.info(f"Registered {hook_type} hook '{spec_name}' on {module_name}")


def resolve_callable(path: Optional[str]) -> Optional[Callable]:
    if path is None:
        return None

    if ":" in path:
        module_name, fn_name = path.split(":", 1)
    else:
        parts = path.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid hook callable path '{path}'. "
                "Expected 'module.submodule:factory' or 'module.submodule.factory'."
            )
        *mod_parts, fn_name = parts
        module_name = ".".join(mod_parts)

    module = importlib.import_module(module_name)
    try:
        return getattr(module, fn_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{fn_name}' "
            f"(from hook path '{path}')"
        ) from e
