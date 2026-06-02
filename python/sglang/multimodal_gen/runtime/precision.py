import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import torch

from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = logging.getLogger(__name__)


_COMPONENTS_WITHOUT_PRECISION_POLICY = frozenset(
    {
        "image_processor",
        "processor",
        "scheduler",
        "tokenizer",
        "vision_language_encoder",
    }
)
_WARNED_UNKNOWN_COMPONENTS: set[str] = set()
_TEXT_ENCODER_SUFFIX_RE = re.compile(r"^text_encoder(?:_(\d+))?$")


def _text_encoder_precision_index(module_name: str, precision_count: int) -> int:
    match = _TEXT_ENCODER_SUFFIX_RE.match(module_name)
    if match is None:
        raise ValueError(f"Invalid text encoder component name: {module_name!r}")

    suffix = match.group(1)
    index = 0 if suffix is None else int(suffix) - 1
    if index < 0:
        raise ValueError(
            f"Invalid text encoder component name {module_name!r}: "
            "numeric suffix must be >= 1"
        )
    if index >= precision_count:
        raise ValueError(
            f"No configured precision for {module_name!r}; "
            f"text_encoder_precisions has {precision_count} entries"
        )
    return index


@dataclass(frozen=True)
class PrecisionSpec:
    """Resolved execution precision for one pipeline component.

    `is_user_policy=True` means the dtype came from user/config policy such as
    `vae_precision`. `is_user_policy=False` means the dtype is an execution
    constraint, e.g. a kernel/numerics/hardware requirement. Keeping that split
    explicit prevents accidental replacement of required hardcoded dtypes while
    still making policy-driven dtype plumbing deterministic.
    """

    component: str
    dtype: torch.dtype
    is_user_policy: bool
    reason: Optional[str] = None


def resolve_precision(
    server_args,
    component: str,
    *,
    precision_attr: Optional[str] = None,
    constraint_dtype: Optional[torch.dtype] = None,
    constraint_reason: Optional[str] = None,
) -> PrecisionSpec:
    if constraint_dtype is not None:
        return PrecisionSpec(
            component=component,
            dtype=constraint_dtype,
            is_user_policy=False,
            reason=constraint_reason,
        )

    precision_attr = precision_attr or f"{component}_precision"
    precision = getattr(server_args.pipeline_config, precision_attr)
    try:
        dtype = PRECISION_TO_TYPE[precision]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported {precision_attr}={precision!r}; "
            f"expected one of {sorted(PRECISION_TO_TYPE)}"
        ) from exc

    return PrecisionSpec(
        component=component,
        dtype=dtype,
        is_user_policy=True,
        reason=f"server_args.pipeline_config.{precision_attr}",
    )


def precision_from_string(
    component: str, precision: str, *, reason: Optional[str] = None
) -> PrecisionSpec:
    """Create a user-policy precision spec from an already-selected string."""
    try:
        dtype = PRECISION_TO_TYPE[precision]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported precision for {component}: {precision!r}; "
            f"expected one of {sorted(PRECISION_TO_TYPE)}"
        ) from exc
    return PrecisionSpec(
        component=component,
        dtype=dtype,
        is_user_policy=True,
        reason=reason,
    )


def resolve_component_precision(
    server_args, module_name: str
) -> Optional[PrecisionSpec]:
    """Resolve the configured precision policy for a named pipeline component."""
    pipeline_config = getattr(server_args, "pipeline_config", None)
    if pipeline_config is None:
        return None

    precision_attr = None
    if module_name == "audio_vae" or module_name == "vocoder":
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
    elif module_name.startswith("text_encoder"):
        precisions = getattr(pipeline_config, "text_encoder_precisions", None)
        if precisions:
            index = _text_encoder_precision_index(module_name, len(precisions))
            precision = precisions[index]
            try:
                dtype = PRECISION_TO_TYPE[precision]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported text_encoder_precisions[{index}]={precision!r}; "
                    f"expected one of {sorted(PRECISION_TO_TYPE)}"
                ) from exc
            return PrecisionSpec(
                component=module_name,
                dtype=dtype,
                is_user_policy=True,
                reason=f"server_args.pipeline_config.text_encoder_precisions[{index}]",
            )
        return None

    if precision_attr is None:
        if (
            module_name not in _COMPONENTS_WITHOUT_PRECISION_POLICY
            and module_name not in _WARNED_UNKNOWN_COMPONENTS
        ):
            logger.warning(
                "No precision policy is registered for component %s; "
                "loading it with the native/default dtype",
                module_name,
            )
            _WARNED_UNKNOWN_COMPONENTS.add(module_name)
        return None

    if not hasattr(pipeline_config, precision_attr):
        logger.warning(
            "Component %s maps to precision policy %s, but pipeline_config does "
            "not define it; loading it with the native/default dtype",
            module_name,
            precision_attr,
        )
        return None
    return resolve_precision(server_args, module_name, precision_attr=precision_attr)


def autocast_enabled(dtype: torch.dtype, disable_autocast: bool) -> bool:
    return dtype != torch.float32 and not disable_autocast


def get_module_dtype(module, default: torch.dtype = torch.float32) -> torch.dtype:
    """Return a module's effective dtype, preferring actual parameter dtype."""
    try:
        return next(module.parameters()).dtype
    except (AttributeError, StopIteration):
        pass

    dtype = getattr(module, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype

    return default


def align_tensor_to_module_dtype(
    tensor: torch.Tensor,
    module,
    *,
    device: Optional[Union[torch.device, str]] = None,
    default_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Move a tensor to a module's device/dtype before invoking the module.

    This prevents fp16/fp32 mismatches such as fp16 weights receiving fp32
    inputs in post-processing modules. Parameter dtype is treated as the source
    of truth, with `module.dtype` only used as a fallback.
    """
    dtype = get_module_dtype(module, default=default_dtype)
    if device is None:
        try:
            device = next(module.parameters()).device
        except (AttributeError, StopIteration):
            device = tensor.device
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return tensor.to(device=device)
    return tensor.to(device=device, dtype=dtype)


def precision_cache_key(
    model_path: str,
    device: Union[torch.device, str],
    dtype: torch.dtype,
) -> tuple[str, str, torch.dtype]:
    """Build a cache key that cannot collide across device/dtype contexts."""
    return (model_path, str(device), dtype)


@contextmanager
def temporary_module_dtype(
    module,
    dtype: torch.dtype,
    *,
    enabled: bool = True,
    restore_dtype: Optional[torch.dtype] = None,
) -> Iterator:
    """Temporarily cast a module and restore its original/effective dtype.

    When possible, restoration is per floating/complex parameter and buffer so
    mixed-dtype modules do not get collapsed to a single dtype after the context.
    Non-``nn.Module`` objects fall back to restoring the effective dtype via
    ``module.to(dtype=...)``.
    """
    if not enabled:
        yield module
        return

    original_dtype = restore_dtype or get_module_dtype(module)
    parameter_dtypes = {}
    buffer_dtypes = {}
    if isinstance(module, torch.nn.Module) and restore_dtype is None:
        parameter_dtypes = {
            name: param.dtype
            for name, param in module.named_parameters()
            if param.is_floating_point() or param.is_complex()
        }
        buffer_dtypes = {
            name: buffer.dtype
            for name, buffer in module.named_buffers()
            if buffer.is_floating_point() or buffer.is_complex()
        }

    module = module.to(dtype=dtype)
    try:
        yield module
    finally:
        if parameter_dtypes or buffer_dtypes:
            for name, param in module.named_parameters():
                param_dtype = parameter_dtypes.get(name)
                if param_dtype is not None and param.dtype != param_dtype:
                    param.data = param.data.to(dtype=param_dtype)
                    if param.grad is not None and param.grad.dtype != param_dtype:
                        param.grad.data = param.grad.data.to(dtype=param_dtype)
            for name, buffer in module.named_buffers():
                buffer_dtype = buffer_dtypes.get(name)
                if buffer_dtype is not None and buffer.dtype != buffer_dtype:
                    buffer.data = buffer.data.to(dtype=buffer_dtype)
        else:
            module.to(dtype=original_dtype)
