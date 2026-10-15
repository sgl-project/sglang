"""Enabling and refusal (RFC #43949 section 6): default-deny over deployment features, the spec's requirements, then
its own checks, every reason reported at once. Config-stage refusals run from the server-args pipeline
(``handle_monokernel``); model-stage ones in ``runtime.attach``."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

from sglang.kernels.ops.mono.spec import DeviceLimits, Feature, MonoSpec, load_spec

logger = logging.getLogger(__name__)

# the SGLang names this module reads (pinned by test_mono_contract)
SGLANG_READS = (
    "sglang.srt.arg_groups.model_override_base.model_config_of",
    "sglang.srt.arg_groups.overrides.resolving_view",
)

SWITCH = "--enable-monokernel"


class MonoRefused(ValueError):
    pass


def mono_enabled(cfg: Any) -> bool:
    """``--enable-monokernel`` on the (resolved) server args."""
    return bool(getattr(cfg, "enable_monokernel", False))


class _Unreadable(Exception):
    pass


def _read(cfg: Any, name: str):
    try:
        return getattr(cfg, name)
    except AttributeError:
        raise _Unreadable(name) from None


def _verify_uneven(c) -> bool:
    if _read(c, "speculative_algorithm") is None:
        return False
    return (_read(c, "speculative_eagle_topk") or 1) > 1


# feature -> "in use" on the resolved server args; a setting it cannot read counts as in use
_FEATURE_IN_USE: dict[Feature, Callable[[Any], bool]] = {
    Feature.PIPELINE_PARALLEL: lambda c: (_read(c, "pp_size") or 1) > 1,
    Feature.MULTI_NODE_TP: lambda c: (_read(c, "nnodes") or 1) > 1,
    Feature.DP_ATTENTION: lambda c: bool(_read(c, "enable_dp_attention")),
    Feature.EXPERT_PARALLEL: lambda c: (_read(c, "ep_size") or 1) > 1,
    Feature.EPLB: lambda c: bool(_read(c, "enable_eplb")),
    Feature.CONTEXT_PARALLEL: lambda c: (
        (_read(c, "attn_cp_size") or 1) > 1 or (_read(c, "dcp_size") or 1) > 1
    ),
    Feature.SPEC_VERIFY: lambda c: _read(c, "speculative_algorithm") is not None,
    Feature.UNEVEN_VERIFY: _verify_uneven,
    Feature.TWO_BATCH_OVERLAP: lambda c: bool(_read(c, "enable_two_batch_overlap")),
    Feature.LORA: lambda c: bool(_read(c, "enable_lora")),
    Feature.PD_DISAGGREGATION: lambda c: (
        _read(c, "disaggregation_mode") not in (None, "null")
    ),
    Feature.ROUTED_EXPERT_CAPTURE: lambda c: bool(
        _read(c, "enable_return_routed_experts")
    ),
    Feature.INDEXER_TOPK_CAPTURE: lambda c: bool(
        _read(c, "enable_return_indexer_topk")
    ),
    Feature.EAGLE3_HIDDEN_STATES: lambda c: (
        str(_read(c, "speculative_algorithm")).upper() == "EAGLE3"
    ),
    Feature.KV_OFFLOAD: lambda c: bool(_read(c, "enable_hierarchical_cache")),
    Feature.MEMORY_RELEASE: lambda c: bool(_read(c, "enable_memory_saver")),
    Feature.CPU_OFFLOAD: lambda c: (_read(c, "cpu_offload_gb") or 0) > 0,
    Feature.TORCH_COMPILE: lambda c: bool(_read(c, "enable_torch_compile")),
    Feature.DETERMINISTIC: lambda c: bool(_read(c, "enable_deterministic_inference")),
    Feature.EAGER: lambda c: bool(
        _read(c, "disable_cuda_graph") or _read(c, "disable_decode_cuda_graph")
    ),
}

# Requirements field -> the server setting it constrains
_REQUIRED_SETTING = {
    "dtypes": "dtype",
    "kv_cache": "kv_cache_dtype",
    "attention_backends": "attention_backend",
    "moe_backends": "moe_runner_backend",
}


def current_isa() -> Optional[str]:
    """The GPU's ISA (e.g. ``gfx950``), or None without a ROCm GPU."""
    if not torch.version.hip or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def device_limits(spec: MonoSpec, isa: Optional[str]) -> Optional[DeviceLimits]:
    return spec.devices.get(isa) if isa is not None else None


def _in_use(feature: Feature, cfg: Any) -> tuple[bool, str]:
    try:
        return _FEATURE_IN_USE[feature](cfg), ""
    except _Unreadable as e:
        return True, f" (cannot read {e.args[0]})"


def feature_refusals(cfg: Any, spec: MonoSpec) -> list[str]:
    reasons = []
    for feature in _FEATURE_IN_USE:
        if feature in spec.features or feature is Feature.EAGER:
            continue  # eager is allowed, with a warning (debugging only)
        used, why = _in_use(feature, cfg)
        if used:
            reasons.append(
                f"{feature.value}{why} is not supported by the {spec.name} monokernels"
            )
    return reasons


def requirement_refusals(cfg: Any, spec: MonoSpec) -> list[str]:
    reasons = []
    for field_name, setting in _REQUIRED_SETTING.items():
        accepted = getattr(spec.requires, field_name)
        if not accepted:
            continue
        try:
            value = _read(cfg, setting)
        except _Unreadable:
            reasons.append(
                f"cannot read {setting}; the {spec.name} monokernels need one of {sorted(accepted)}"
            )
            continue
        if str(value) not in accepted:
            reasons.append(
                f"{setting}={value!r}: the {spec.name} monokernels need one of {sorted(accepted)}"
            )
    return reasons


def common_refusals(cfg: Any, spec: MonoSpec, isa: Optional[str]) -> list[str]:
    """Reasons every monokernel model shares: unlisted features in use, requirements, the wait deadline, TP
    size, ISA."""
    from sglang.kernels.ops.mono.common_hip.guard import deadline_refusals

    reasons = feature_refusals(cfg, spec) + requirement_refusals(cfg, spec)
    reasons += deadline_refusals(getattr(cfg, "watchdog_timeout", None))
    limits = device_limits(spec, isa)
    if limits is None:
        reasons.append(
            f"the {spec.name} monokernels are built for {', '.join(spec.devices)}, not {isa or 'this GPU'}"
        )
    tp = getattr(cfg, "tp_size", None) or 1
    tp_sizes = (
        limits.tp_sizes
        if limits is not None
        else tuple(sorted({t for d in spec.devices.values() for t in d.tp_sizes}))
    )
    if tp not in tp_sizes:
        reasons.append(
            f"tp_size {tp} not in {spec.name}'s supported TP sizes {tp_sizes}"
        )
    return reasons


def config_refusals(
    cfg: Any, hf_config: Any, spec: MonoSpec, isa: Optional[str]
) -> list[str]:
    return common_refusals(cfg, spec, isa) + list(
        spec.config_disable_reasons(cfg, hf_config)
    )


def device_refusals(spec: MonoSpec, limits: DeviceLimits) -> list[str]:
    """The launch grid must fit the CUs present (a partitioned GPU has fewer)."""
    if not torch.cuda.is_available():
        return []
    cus = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    if cus < limits.resident_blocks:
        return [
            f"{cus} CUs present, the {spec.name} monokernels keep {limits.resident_blocks} workgroups resident"
            " (partitioned GPU?)"
        ]
    return []


def raise_if_refused(stage: str, spec_name: str, reasons: list[str]) -> None:
    if reasons:
        lines = "\n  - ".join(reasons)
        raise MonoRefused(f"{SWITCH} refused ({spec_name}, {stage}):\n  - {lines}")


def warn_eager(cfg: Any, spec: MonoSpec) -> None:
    used, _ = _in_use(Feature.EAGER, cfg)
    if used and Feature.EAGER not in spec.features:
        logger.warning(
            "monokernels under eager decode: debugging only (graphs disabled)"
        )


def handle_monokernel(server_args) -> None:
    """The config hook: refuse at startup (every reason at once) when the model has no spec or the resolved
    deployment uses a feature the spec does not support. Runs after every other resolution."""
    from sglang.srt.arg_groups.model_override_base import model_config_of
    from sglang.srt.arg_groups.overrides import resolving_view

    cfg = resolving_view(server_args)
    if not mono_enabled(cfg):
        return
    hf_config = model_config_of(server_args).hf_config
    arch = (hf_config.architectures or ["?"])[0]
    spec = load_spec(arch)
    if spec is None:
        raise MonoRefused(f"{SWITCH}: no mono spec for {arch}")
    raise_if_refused(
        "config", spec.name, config_refusals(cfg, hf_config, spec, current_isa())
    )
    warn_eager(cfg, spec)
    logger.info("%s: %s mono spec accepted the configuration", SWITCH, spec.name)
