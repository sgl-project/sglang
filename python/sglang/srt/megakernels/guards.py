# SPDX-License-Identifier: Apache-2.0
"""Startup checks for --enable-nvidia-megakernel on the resolved configuration."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any, Optional

import torch

from sglang.srt.arg_groups.model_override_base import model_config_of, resolving_view
from sglang.srt.environ import envs
from sglang.srt.megakernels.spec import (
    ENTRY_POINT,
    FLAG,
    Feature,
    MegakernelSpec,
    find_spec,
    registered_architectures,
)
from sglang.srt.plugins import GENERAL_PLUGINS_GROUP

logger = logging.getLogger(__name__)

PLUGIN_URL = "https://github.com/sgl-project/sgl-nvidia-mega"

# Leave time for peers to abort before the process watchdog fires.
MAX_DEADLINE_S, MIN_DEADLINE_S = 60.0, 5.0


def device_isa() -> Optional[str]:
    """This process's GPU as "sm_<major><minor>", or None without a visible GPU."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return f"sm_{props.major}{props.minor}"


def deadline_s(watchdog_timeout: Optional[float]) -> float:
    """The deadline of every cross-GPU wait: half the watchdog, at most MAX_DEADLINE_S."""
    if not watchdog_timeout or watchdog_timeout <= 0:
        return MAX_DEADLINE_S
    return min(MAX_DEADLINE_S, 0.5 * float(watchdog_timeout))


def _ragged_verify(cfg: Any) -> bool:
    mode = envs.SGLANG_RAGGED_VERIFY_MODE.get().lower()
    return cfg.speculative_algorithm is not None and mode != "static"


# One probe a feature, on the resolved settings; a spec accepts features by listing
# them. PARTITIONED_GPU needs the loaded device, so the runtime checks it.
FEATURE_PROBES = {
    Feature.PIPELINE_PARALLEL: lambda c: c.pp_size > 1,
    Feature.MULTI_NODE_TP: lambda c: c.nnodes > 1,
    Feature.DP_ATTENTION: lambda c: (
        bool(c.enable_dp_attention) or (c.attn_dp_size or 1) > 1
    ),
    Feature.EXPERT_PARALLEL: lambda c: (c.ep_size or 1) > 1,
    Feature.EPLB: lambda c: bool(c.enable_eplb),
    Feature.CONTEXT_PARALLEL: lambda c: (
        (c.attn_cp_size or 1) > 1 or (c.dcp_size or 1) > 1 or bool(c.enable_prefill_cp)
    ),
    Feature.SPECULATIVE_DECODING: lambda c: c.speculative_algorithm is not None,
    Feature.RAGGED_VERIFY: _ragged_verify,
    Feature.TWO_BATCH_OVERLAP: lambda c: bool(
        c.enable_two_batch_overlap or c.enable_single_batch_overlap
    ),
    Feature.LORA: lambda c: bool(c.enable_lora or c.lora_paths),
    Feature.PD_DISAGGREGATION: lambda c: (c.disaggregation_mode or "null") != "null",
    Feature.RETURN_ROUTED_EXPERTS: lambda c: bool(c.enable_return_routed_experts),
    Feature.RETURN_INDEXER_TOPK: lambda c: bool(c.enable_return_indexer_topk),
    Feature.KV_OFFLOAD: lambda c: bool(
        c.enable_hierarchical_cache
        or c.disaggregation_decode_enable_offload_kvcache
        or c.enable_lmcache
        or c.enable_flexkv
    ),
    Feature.HISPARSE: lambda c: bool(c.enable_hisparse),
    Feature.MEMORY_RELEASE: lambda c: bool(
        c.enable_memory_saver or c.enable_weights_cpu_backup
    ),
    Feature.CPU_OFFLOAD: lambda c: (
        (c.cpu_offload_gb or 0) > 0 or (c.offload_group_size or 0) > 0
    ),
    Feature.TORCH_COMPILE: lambda c: bool(c.enable_torch_compile),
    Feature.DETERMINISTIC_INFERENCE: lambda c: bool(
        c.enable_deterministic_inference or c.rl_on_policy_target
    ),
}


def features_in_use(cfg: Any) -> set[Feature]:
    return {feature for feature, probe in FEATURE_PROBES.items() if probe(cfg)}


def plugin_reasons() -> list[str]:
    """Why the plugin that registers the specs will not load ([] when it will)."""
    if ENTRY_POINT not in {ep.name for ep in entry_points(group=GENERAL_PLUGINS_GROUP)}:
        return [
            f"the {ENTRY_POINT!r} plugin is not installed; install sgl-nvidia-mega "
            f"({PLUGIN_URL})"
        ]
    allowed = envs.SGLANG_PLUGINS.get()
    if allowed and ENTRY_POINT not in {x.strip() for x in allowed.split(",")}:
        return [f"SGLANG_PLUGINS={allowed!r} does not list the {ENTRY_POINT!r} plugin"]
    return []


def megakernel_disable_reasons(
    *, cfg: Any, hf_config: Any, isa: Optional[str]
) -> tuple[Optional[MegakernelSpec], list[str]]:
    """The spec for this model and every reason it cannot serve this configuration."""
    if isa is None:
        return None, ["no GPU is visible"]
    spec, why = find_spec(architectures=hf_config.architectures, isa=isa)
    if spec is None and not registered_architectures():
        # the plugin loader logs and skips a plugin that fails, so nothing registered
        why = (
            f"the {ENTRY_POINT!r} plugin is installed but registered no spec; see "
            "'Failed to load plugin' or 'Failed to execute general plugin' above"
        )
    if spec is None:
        return None, [why]
    req = spec.requires
    reasons = [
        f"{f.value} is in use and {spec.name} does not accept it"
        for f in sorted(features_in_use(cfg) - spec.features, key=lambda f: f.name)
    ]
    one_of = [
        ("tp_size", cfg.tp_size, req.tp_sizes),
        (
            "decode CUDA graphs",
            cfg.cuda_graph_config.decode.backend,
            req.decode_graph_backends,
        ),
    ] + [(arg, getattr(cfg, arg), ok) for arg, ok in req.server_args.items()]
    exactly = [
        # another model's config may not have the field at all
        (f"config.{key}", getattr(hf_config, key, None), want)
        for key, want in req.hf_config.items()
    ] + [(name, getattr(envs, name).get(), want) for name, want in req.env.items()]
    reasons += [
        f"{what} {value!r} is not one of {sorted(map(str, allowed))}"
        for what, value, allowed in one_of
        if allowed and value not in allowed
    ]
    reasons += [
        f"{what} {value!r} is not {want!r}"
        for what, value, want in exactly
        if value != want
    ]
    watchdog = cfg.watchdog_timeout
    if watchdog and deadline_s(watchdog) < MIN_DEADLINE_S:
        reasons.append(
            f"--watchdog-timeout {watchdog:g} s leaves the waits a deadline under "
            f"{MIN_DEADLINE_S:g} s"
        )
    if spec.config_disable_reasons is not None:
        reasons += spec.config_disable_reasons(cfg, hf_config)
    return spec, reasons


def _dtype_reasons(spec: Optional[MegakernelSpec], model_config: Any) -> list[str]:
    # the resolved dtype, which the HF config alone does not give
    dtype = str(model_config.dtype).replace("torch.", "")
    if spec is None or not spec.requires.dtypes or dtype in spec.requires.dtypes:
        return []
    return [f"dtype {dtype!r} is not one of {sorted(spec.requires.dtypes)}"]


def handle_nvidia_megakernel(server_args: Any) -> None:
    """Resolution step: fail before the weights load when the flag cannot be served,
    listing every reason."""
    cfg = resolving_view(server_args)
    if not cfg.enable_nvidia_megakernel:
        return
    reasons = plugin_reasons()
    if not reasons:
        model_config = model_config_of(server_args)
        spec, reasons = megakernel_disable_reasons(
            cfg=cfg, hf_config=model_config.hf_config, isa=device_isa()
        )
        reasons += _dtype_reasons(spec, model_config)
    if cfg.monokernel_backend is not None:
        # both replace the model's forward, which would fail only after weights load
        reasons.append(
            f"--monokernel-backend {cfg.monokernel_backend} is also set; use one of the two"
        )
    if reasons:
        raise ValueError(
            f"{FLAG} cannot run this setup:\n  - " + "\n  - ".join(reasons)
        )
    logger.warning("%s is experimental. Its kernels and coverage may change.", FLAG)
    logger.info("%s: %s on %s", FLAG, spec.name, device_isa())
