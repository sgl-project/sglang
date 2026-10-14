"""Enabling and refusal: default-deny over deployment features, then the spec's own checks, every reason reported at
once. Config-stage refusals run from the server-args pipeline (``configure``); model-stage ones after weight load."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from sglang.srt.layers.mono.spec import Feature, MonoSpec

logger = logging.getLogger(__name__)


class MonoRefused(ValueError):
    pass


def mono_enabled(cfg: Any) -> bool:
    """``--layer-backend mono`` on the (resolved) server args."""
    return cfg.layer_backend == "mono"


# feature -> predicate "in use" on the resolved server args
_FEATURE_IN_USE: dict[Feature, Callable[[Any], bool]] = {
    Feature.TWO_BATCH_OVERLAP: lambda c: bool(c.enable_two_batch_overlap),
    Feature.DP_ATTENTION: lambda c: bool(c.enable_dp_attention),
    Feature.EXPERT_PARALLEL: lambda c: (c.ep_size or 1) > 1,
    Feature.PIPELINE_PARALLEL: lambda c: (c.pp_size or 1) > 1,
    Feature.PD_DISAGGREGATION: lambda c: c.disaggregation_mode not in (None, "null"),
    Feature.LORA: lambda c: bool(c.enable_lora),
    Feature.HIERARCHICAL_CACHE: lambda c: bool(c.enable_hierarchical_cache),
    Feature.EPLB: lambda c: bool(c.enable_eplb),
    Feature.TORCH_COMPILE: lambda c: bool(c.enable_torch_compile),
    Feature.DETERMINISTIC: lambda c: bool(c.enable_deterministic_inference),
    Feature.SPEC_VERIFY: lambda c: c.speculative_algorithm is not None,
    Feature.EAGER: lambda c: bool(c.disable_cuda_graph or c.disable_decode_cuda_graph),
}


def common_refusals(
    cfg: Any, spec: MonoSpec, platform_ok: Optional[bool] = None
) -> list[str]:
    """Reasons every mono model shares: unlisted features in use, unsupported TP, platform."""
    reasons = []
    for feature, in_use in _FEATURE_IN_USE.items():
        if feature in spec.supports or not in_use(cfg):
            continue
        if feature is Feature.EAGER:
            continue  # allowed, with a warning (debugging only)
        reasons.append(
            f"{feature.value} is not supported by the {spec.name} mono kernels"
        )
    tp = cfg.tp_size or 1
    if tp not in spec.supported_tp:
        reasons.append(
            f"tp_size {tp} not in {spec.name}'s supported TP sizes {spec.supported_tp}"
        )
    if platform_ok is None:
        platform_ok = _platform_ok()
    if not platform_ok:
        reasons.append("mono kernels need an AMD gfx950 GPU (MI350/MI355)")
    return reasons


def _platform_ok() -> bool:
    from sglang.srt.utils import is_gfx95_supported, is_hip

    return is_hip() and is_gfx95_supported()


def config_refusals(
    cfg: Any, hf_config: Any, spec: MonoSpec, platform_ok: Optional[bool] = None
) -> list[str]:
    return common_refusals(cfg, spec, platform_ok) + list(
        spec.refuse_config(cfg, hf_config)
    )


def raise_if_refused(stage: str, spec_name: str, reasons: list[str]) -> None:
    if reasons:
        lines = "\n  - ".join(reasons)
        raise MonoRefused(
            f"--layer-backend mono refused ({spec_name}, {stage}):\n  - {lines}"
        )


def warn_eager(cfg: Any, spec: MonoSpec) -> None:
    if _FEATURE_IN_USE[Feature.EAGER](cfg) and Feature.EAGER not in spec.supports:
        logger.warning(
            "mono kernels under eager decode: debugging only (graphs disabled)"
        )
