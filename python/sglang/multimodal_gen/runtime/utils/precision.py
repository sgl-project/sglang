from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from typing import Iterator, Optional, Union

import torch

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    component_base_name,
    is_legacy_dit_offload_component_name,
    is_legacy_image_encoder_offload_component_name,
    is_text_encoder_component_name,
    is_vae_component_name,
)
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


def component_precision_overrides(server_args: object) -> Mapping[str, str]:
    overrides = vars(server_args).get("component_precisions")
    return {} if overrides is None else overrides


def resolve_precision(
    server_args,
    component_or_precision_attr: str,
    *,
    precision_attr: Optional[str] = None,
    field_name: Optional[str] = None,
) -> torch.dtype:
    component_precision = component_precision_overrides(server_args).get(
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


def resolve_component_precision(server_args, module_name: str) -> Optional[torch.dtype]:
    exact_precision = resolve_exact_component_precision(server_args, module_name)
    if exact_precision is not None:
        return exact_precision

    pipeline_config = vars(server_args).get("pipeline_config")
    if pipeline_config is None:
        return None

    base_name = component_base_name(module_name)
    if base_name in ("audio_vae", "vocoder"):
        precision_attr = "audio_vae_precision"
    elif is_vae_component_name(module_name):
        precision_attr = "vae_precision"
    elif is_legacy_dit_offload_component_name(module_name):
        precision_attr = "dit_precision"
    elif is_legacy_image_encoder_offload_component_name(module_name):
        precision_attr = "image_encoder_precision"
    elif module_name == "text_encoder":
        index = 0
    elif module_name.startswith("text_encoder_"):
        suffix = module_name.removeprefix("text_encoder_")
        if not suffix.isdigit():
            return None
        index = max(int(suffix) - 1, 0)
    elif is_text_encoder_component_name(module_name):
        return None
    else:
        return None

    if is_text_encoder_component_name(module_name):
        precisions = vars(pipeline_config).get("text_encoder_precisions")
        if not precisions:
            return None
        if index < 0 or index >= len(precisions):
            raise ValueError(
                f"No configured precision for {module_name!r}; "
                f"text_encoder_precisions has {len(precisions)} entries"
            )
        precision = precisions[index]
        return precision_to_dtype(precision, f"text_encoder_precisions[{index}]")

    if precision_attr not in vars(pipeline_config):
        return None
    return resolve_precision(server_args, module_name, precision_attr=precision_attr)


def resolve_exact_component_precision(
    server_args, component_name: str
) -> Optional[torch.dtype]:
    precision = component_precision_overrides(server_args).get(component_name)
    if precision is None:
        return None
    return precision_to_dtype(precision, f"component_precisions.{component_name}")


def validate_shared_component_autocast(server_args, component_names: list[str]) -> None:
    overrides = component_precision_overrides(server_args)
    if len(component_names) < 2 or not any(
        name in overrides for name in component_names
    ):
        return
    precisions = {
        resolve_component_precision(server_args, name) for name in component_names
    }
    if len(precisions) > 1:
        raise ValueError(
            "Components sharing one execution path must use one precision because "
            "they share a single autocast context"
        )


def explicit_component_autocast_context(
    server_args, component_name: str, dtype: torch.dtype
):
    overrides = component_precision_overrides(server_args)
    return autocast_context(
        dtype,
        server_args.disable_autocast,
        enabled=(
            component_name in overrides
            and autocast_enabled(dtype, server_args.disable_autocast)
        ),
    )


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
