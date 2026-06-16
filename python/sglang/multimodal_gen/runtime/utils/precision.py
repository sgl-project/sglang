from contextlib import contextmanager
from typing import Iterator, Optional, Union

import torch

from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


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
    precision_attr = precision_attr or component_or_precision_attr
    precision = getattr(server_args.pipeline_config, precision_attr)
    return precision_to_dtype(precision, field_name or precision_attr)


def resolve_component_precision(server_args, module_name: str) -> Optional[torch.dtype]:
    pipeline_config = getattr(server_args, "pipeline_config", None)
    if pipeline_config is None:
        return None

    if module_name in ("audio_vae", "vocoder"):
        precision_attr = "audio_vae_precision"
    elif module_name in ("vae", "video_vae"):
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
    return dtype != torch.float32 and not disable_autocast


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
