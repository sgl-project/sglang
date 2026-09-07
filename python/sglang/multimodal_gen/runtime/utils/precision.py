from contextlib import contextmanager, nullcontext
from typing import Iterator, List, Optional, Union

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE


def precision_to_dtype(precision: str, field_name: str = "precision") -> torch.dtype:
    try:
        return PRECISION_TO_TYPE[precision]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported {field_name}={precision!r}; "
            f"expected one of {sorted(PRECISION_TO_TYPE)}"
        ) from exc


def resolve_precision(
    server_args,
    component_or_precision_attr: str,
    *,
    precision_attr: Optional[str] = None,
    field_name: Optional[str] = None,
) -> torch.dtype:
    component_precision = server_args.component_precisions.get(
        component_or_precision_attr
    )
    if component_precision is not None:
        return precision_to_dtype(
            component_precision,
            f"component_precisions.{component_or_precision_attr}",
        )
    precision_attr = precision_attr or component_or_precision_attr
    precision = getattr(server_args.pipeline_config, precision_attr)
    return precision_to_dtype(precision, field_name or precision_attr)


def resolve_decode_precision(
    server_args,
    component_name: str = "vae",
    *,
    quality: str | None = None,
) -> torch.dtype:
    component_precision = server_args.component_precisions.get(component_name)
    if component_precision is not None:
        return precision_to_dtype(
            component_precision, f"component_precisions.{component_name}"
        )

    pipeline_config = server_args.pipeline_config
    if component_name in ("audio_vae", "vocoder"):
        return resolve_precision(
            server_args,
            component_name,
            precision_attr="audio_vae_precision",
        )

    if quality == "high":
        high_precision = getattr(pipeline_config, "vae_decode_precision_high", None)
        if high_precision is not None:
            return precision_to_dtype(high_precision, "vae_decode_precision_high")

    decode_precision = getattr(pipeline_config, "vae_decode_precision", None)
    if decode_precision is not None:
        return precision_to_dtype(decode_precision, "vae_decode_precision")
    return resolve_precision(
        server_args,
        component_name,
        precision_attr="vae_precision",
    )


def resolve_component_precision_override(
    server_args, module_name: str
) -> Optional[torch.dtype]:
    exact_precision = server_args.component_precisions.get(module_name)
    if exact_precision is None:
        return None
    return precision_to_dtype(exact_precision, f"component_precisions.{module_name}")


def resolve_component_precision(server_args, module_name: str) -> Optional[torch.dtype]:
    exact_precision = resolve_component_precision_override(server_args, module_name)
    if exact_precision is not None:
        return exact_precision

    pipeline_config = server_args.pipeline_config

    if module_name in ("audio_vae", "vocoder"):
        precision_attr = "audio_vae_precision"
    elif module_name in ("vae", "video_vae", "diffusion_decoder"):
        precision_attr = "vae_precision"
    elif module_name in (
        "transformer",
        "transformer_2",
        "audio_dit",
        "video_dit",
        "connectors",
        "dual_tower_bridge",
    ):
        precision_attr = "dit_precision"
    elif module_name == "image_encoder":
        precision_attr = "image_encoder_precision"
    elif module_name == "text_encoder" or module_name.startswith("text_encoder_"):
        precisions = getattr(pipeline_config, "text_encoder_precisions", None)
        if not precisions:
            return None
        suffix = module_name.removeprefix("text_encoder")
        index = 0 if suffix == "" else int(suffix.removeprefix("_")) - 1
        if index < 0 or index >= len(precisions):
            raise ValueError(
                f"No configured precision for {module_name!r}; "
                f"text_encoder_precisions has {len(precisions)} entries"
            )
        precision = precisions[index]
        return precision_to_dtype(precision, f"text_encoder_precisions[{index}]")
    else:
        return None

    if not hasattr(pipeline_config, precision_attr):
        return None
    return resolve_precision(server_args, precision_attr)


def autocast_enabled(dtype: torch.dtype, disable_autocast: bool) -> bool:
    return (
        dtype != torch.float32
        and not disable_autocast
        and current_platform.is_amp_supported()
    )


def autocast_enabled_for_device(
    tensor: torch.Tensor, dtype: torch.dtype, disable_autocast: bool
) -> bool:
    return tensor.device.type == current_platform.device_type and autocast_enabled(
        dtype, disable_autocast
    )


def autocast_context(
    dtype: torch.dtype,
    disable_autocast: bool,
    *,
    enabled: Optional[bool] = None,
):
    autocast_is_enabled = (
        autocast_enabled(dtype, disable_autocast) if enabled is None else enabled
    )
    if not autocast_is_enabled and current_platform.is_mps():
        return nullcontext()
    return torch.autocast(
        device_type=current_platform.device_type,
        dtype=dtype,
        enabled=autocast_is_enabled,
    )


def get_module_dtype(module, default: torch.dtype = torch.float32) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except (AttributeError, StopIteration):
        dtype = getattr(module, "dtype", None)
        return dtype if isinstance(dtype, torch.dtype) else default


def align_tensor_to_module_dtype(
    tensor: torch.Tensor,
    module,
    *,
    device: Optional[Union[torch.device, str]] = None,
    default_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    dtype = get_module_dtype(module, default=default_dtype)
    if device is None:
        try:
            device = next(module.parameters()).device
        except (AttributeError, StopIteration):
            device = tensor.device
    if not tensor.is_floating_point():
        return tensor.to(device=device)
    return tensor.to(device=device, dtype=dtype)


def _module_requires_fp32_cast(module) -> bool:
    """Return whether this module contains floating state that must be temporarily cast to fp32.

    This is the guard used by the temporary fp32 helpers: if the module is already fully
    fp32, the context manager exits without mutating any tensors. Mixed-dtype modules are
    converted in-place for the duration of the scope and then restored exactly.
    """
    for param in module.parameters(recurse=True):
        if param.is_floating_point() and param.dtype != torch.float32:
            return True
    for buffer in module.buffers(recurse=True):
        if buffer.is_floating_point() and buffer.dtype != torch.float32:
            return True
    return False


def _capture_module_value_state(module):
    param_state = {
        name: param.detach().clone()
        for name, param in module.named_parameters(recurse=True)
    }
    buffer_state = {
        name: buffer.detach().clone()
        for name, buffer in module.named_buffers(recurse=True)
    }
    return param_state, buffer_state


def _restore_module_value_state(module, param_state, buffer_state):
    current_params = {
        name: param for name, param in module.named_parameters(recurse=True)
    }
    current_buffers = {
        name: buffer for name, buffer in module.named_buffers(recurse=True)
    }

    with torch.no_grad():
        for name, value in param_state.items():
            if name in current_params:
                restored = value.to(device=current_params[name].device)
                current_params[name].data = restored
        for name, value in buffer_state.items():
            if name in current_buffers:
                restored = value.to(device=current_buffers[name].device)
                current_buffers[name].data = restored


def _module_fp32_cache(module):
    params = {name: param for name, param in module.named_parameters(recurse=True)}
    buffers = {name: buffer for name, buffer in module.named_buffers(recurse=True)}
    cache = {"parameters": {}, "buffers": {}}

    with torch.no_grad():
        for name, param in params.items():
            if param.is_floating_point():
                cache["parameters"][name] = param.detach().to(dtype=torch.float32)
                param.data = cache["parameters"][name]
        for name, buffer in buffers.items():
            if buffer.is_floating_point():
                cache["buffers"][name] = buffer.detach().to(dtype=torch.float32)
                buffer.data = cache["buffers"][name]

    return cache


@contextmanager
def temporary_module_fp32_dtype(
    module,
    *,
    enabled: bool = True,
) -> Iterator:
    """Temporarily cast a module's floating parameters and buffers to fp32 for the scope.

    The module is restored to its exact original state on exit, including original dtypes and
    values. This is intended for inference-only, short-lived CPU workarounds where fp32 math is
    required but the module should not be left in a permanently altered state.
    """
    if not enabled:
        yield module
        return

    if not _module_requires_fp32_cast(module):
        yield module
        return

    param_state, buffer_state = _capture_module_value_state(module)
    cache = _module_fp32_cache(module)
    try:
        yield module
    finally:
        _restore_module_value_state(module, param_state, buffer_state)
        cache.clear()


@contextmanager
def temporary_modules_fp32_dtype(
    modules: List[nn.Module],
    *,
    enabled: Union[bool, List[bool]] = True,
) -> Iterator[List[nn.Module]]:
    """Temporarily cast a set of modules to fp32 while preserving their original state.

    This mirrors the single-module helper for multi-module workloads: each module is only
    converted if it contains non-fp32 floating values, and every restored module is returned to
    the exact original dtype/value state after the context exits.
    """
    enabled_list = [enabled] * len(modules) if isinstance(enabled, bool) else enabled
    states = []
    caches = []

    for module, is_enabled in zip(modules, enabled_list):
        if not is_enabled or not _module_requires_fp32_cast(module):
            states.append((None, None))
            caches.append(None)
            continue

        states.append(_capture_module_value_state(module))
        caches.append(_module_fp32_cache(module))

    try:
        yield modules
    finally:
        for module, state, cache, is_enabled in zip(
            modules, states, caches, enabled_list
        ):
            if not is_enabled or state == (None, None):
                continue
            param_state, buffer_state = state
            _restore_module_value_state(module, param_state, buffer_state)
            if cache is not None:
                cache.clear()


@contextmanager
def temporary_module_dtype(
    module,
    dtype: torch.dtype,
    *,
    enabled: bool = True,
    restore_dtype: Optional[torch.dtype] = None,
) -> Iterator:
    if not enabled:
        yield module
        return

    original_dtype = restore_dtype or get_module_dtype(module)
    module = module.to(dtype=dtype)
    try:
        yield module
    finally:
        module.to(dtype=original_dtype)
