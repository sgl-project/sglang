"""Opt-in GLM DSpark correction for unrotated draft weights on NPU.

The configuration lives only around one draft construction. FC conversion is a
load-time candidate, not an exact inverse for a non-orthogonal stored Q.
"""

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)

_DRAFT_LOAD_FORMATS_CALLING_LOAD_WEIGHTS = frozenset(
    {"auto", "fastsafetensors", "mistral", "npcache", "pt", "safetensors"}
)


@dataclass(frozen=True)
class GlmDSparkQuaRotConfig:
    rotation_path: str
    hidden_size: int
    target_model_path: str


_config: ContextVar[Optional[GlmDSparkQuaRotConfig]] = ContextVar(
    "glm_dspark_quarot_config", default=None
)


def get_glm_dspark_quarot_config() -> Optional[GlmDSparkQuaRotConfig]:
    return _config.get()


@contextmanager
def glm_dspark_quarot_scope(config: Optional[GlmDSparkQuaRotConfig]):
    token = _config.set(config)
    try:
        yield
    finally:
        _config.reset(token)


def validate_glm_dspark_quarot_draft_loader(
    *, load_format, weight_cache_mode: str = "off"
) -> None:
    """Reject draft loaders that can bypass the required FC conversion.

    The allowed formats all use ``DefaultModelLoader`` and call the draft
    model's ``load_weights`` method. Direct state-copy loaders would otherwise
    leave the FC in the original coordinate system while the runtime consumes
    rotated target hidden states.
    """
    if weight_cache_mode != "off":
        raise ValueError(
            "GLM DSpark QuaRot correction does not support --weight-cache-mode="
            f"{weight_cache_mode!r}: the IPC draft loader bypasses "
            "DSparkDraftMixin.load_weights()."
        )

    normalized = getattr(load_format, "value", load_format)
    normalized = "auto" if normalized is None else str(normalized).lower()
    if normalized not in _DRAFT_LOAD_FORMATS_CALLING_LOAD_WEIGHTS:
        supported = ", ".join(sorted(_DRAFT_LOAD_FORMATS_CALLING_LOAD_WEIGHTS))
        raise ValueError(
            "GLM DSpark QuaRot correction requires a draft load format that "
            "calls DSparkDraftMixin.load_weights() so the FC can be folded; "
            f"got {normalized!r}. Supported formats: {supported}."
        )


def log_glm_dspark_quarot_runtime_status(*, requested: bool, draft_model) -> None:
    config = getattr(draft_model, "_glm_dspark_quarot_config", None)
    enabled = config is not None
    owns_vocab = bool(getattr(draft_model, "uses_own_vocab_modules", False))
    local_embedding = (
        enabled
        and owns_vocab
        and getattr(draft_model, "embed_tokens", None) is not None
    )
    local_lm_head = (
        enabled and owns_vocab and getattr(draft_model, "lm_head", None) is not None
    )
    logger.info(
        "GLM DSpark QuaRot runtime: requested=%s enabled=%s q_path=%s "
        "draft_local_embedding=%s draft_local_lm_head=%s",
        requested,
        enabled,
        config.rotation_path if enabled else None,
        local_embedding,
        local_lm_head,
    )


def build_glm_dspark_quarot_config(
    *, device, enabled, target_model_config, target_model
) -> Optional[GlmDSparkQuaRotConfig]:
    """Read metadata only after matching the enabled target-specific path.

    ``enabled`` declares that the dense draft checkpoint is unrotated and needs
    correction for this target. Target metadata alone cannot establish the
    draft's coordinate convention.
    """
    if not enabled or str(device).split(":", 1)[0] != "npu":
        return None
    hf_config = getattr(target_model_config, "hf_text_config", None)
    architectures = getattr(hf_config, "architectures", None)
    if (
        not isinstance(architectures, (list, tuple))
        or "GlmMoeDsaForCausalLM" not in architectures
    ):
        return None
    quant_config = getattr(target_model, "quant_config", None)
    get_name = getattr(quant_config, "get_name", None)
    if not callable(get_name) or get_name() != "modelslim":
        return None
    description = getattr(quant_config, "quant_description", None)
    if not isinstance(description, dict) or description.get("is_rot_used") is not True:
        raise ValueError(
            "GLM DSpark QuaRot correction requires a ModelSlim QuaRot target"
        )
    try:
        relative_path = description["optional"]["quarot"]["rotation_map"][
            "global_rotation"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "GLM DSpark target has no QuaRot global_rotation path"
        ) from exc
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or Path(relative_path).is_absolute()
    ):
        raise ValueError("QuaRot global_rotation must be a relative target file path")
    width = getattr(hf_config, "hidden_size", None)
    if type(width) is not int or width <= 0:
        raise ValueError("GLM DSpark target hidden_size must be a positive integer")
    target_path = Path(target_model_config.model_path).resolve(strict=True)
    rotation_path = (target_path / relative_path).resolve(strict=True)
    # get_slice reads the header, not the full matrix during draft construction.
    with safe_open(str(rotation_path), framework="pt", device="cpu") as reader:
        if "global_rotation" not in reader.keys():
            raise ValueError(f"Missing global_rotation tensor in {rotation_path}")
        if reader.get_slice("global_rotation").get_shape() != [width, width]:
            raise ValueError(
                f"QuaRot global_rotation must have shape [{width}, {width}]: {rotation_path}"
            )
    return GlmDSparkQuaRotConfig(str(rotation_path), width, str(target_path))


@torch.no_grad()
def fold_glm_dspark_fc(
    weight: torch.Tensor, config: GlmDSparkQuaRotConfig
) -> torch.Tensor:
    """Return fresh CPU FC blocks F_i @ Q in the original weight dtype.

    This never transforms an existing model Parameter in place, rescales Q,
    applies R, or changes CPU thread settings. Call it on newly read checkpoint
    weights so reloading does not rotate already converted parameters again.
    """
    width = config.hidden_size
    if (
        type(width) is not int
        or width <= 0
        or weight.ndim != 2
        or weight.shape[0] != width
        or weight.shape[1] == 0
        or weight.shape[1] % width
        or not weight.is_floating_point()
    ):
        raise ValueError(
            "GLM DSpark FC must be floating [hidden_size, K * hidden_size]"
        )

    started = time.monotonic()
    logger.info(
        "GLM DSpark QuaRot correction: folding original-coordinate FC %s "
        "with Q=%s from target=%s "
        "on CPU in FP32, storing %s; no inverse or scale correction",
        tuple(weight.shape),
        config.rotation_path,
        config.target_model_path,
        weight.dtype,
    )
    with safe_open(config.rotation_path, framework="pt", device="cpu") as reader:
        if "global_rotation" not in reader.keys():
            raise ValueError(
                f"Missing global_rotation tensor in {config.rotation_path}"
            )
        q = reader.get_tensor("global_rotation")
    if q.shape != (width, width) or not q.is_floating_point():
        raise ValueError(
            "QuaRot global_rotation must be a floating hidden_size square matrix"
        )
    q = q.to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(q).all():
        raise ValueError("QuaRot global_rotation contains non-finite FP32 values")

    folded = torch.empty(weight.shape, dtype=weight.dtype, device="cpu")
    # Keep only Q, the output, and small FP32 tiles; do not allocate a full FP32
    # copy of the 6144 x 30720 checkpoint or a block-diagonal rotation matrix.
    for start in range(0, weight.shape[1], width):
        for row in range(0, width, 128):
            source = (
                weight[row : row + 128, start : start + width]
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )
            if not torch.isfinite(source).all():
                raise ValueError("GLM DSpark FC contains non-finite FP32 values")
            converted = (source @ q).to(dtype=weight.dtype)
            if not torch.isfinite(converted).all():
                raise ValueError("GLM DSpark converted FC contains non-finite values")
            folded[row : row + 128, start : start + width].copy_(converted)
    del source, converted, q
    logger.info(
        "GLM DSpark FC load-time Q folding finished in %.3f seconds",
        time.monotonic() - started,
    )
    return folded
