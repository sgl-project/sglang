# SPDX-License-Identifier: Apache-2.0
"""Startup validation, graph-size alignment, and wait deadlines for monokernels."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.kernels.low_latency_kernels.monokernels.spec import (
    BACKEND,
    FLAG,
    Feature,
    MonoSpec,
    NumericsKind,
    find_spec,
)
from sglang.srt.arg_groups.overrides import (
    attention_backends_of,
    declare_resolution,
    model_config_of,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Leave time for peers to abort before the process watchdog fires.
MAX_DEADLINE_S, MIN_DEADLINE_S = 60.0, 5.0


def deadline_s(watchdog_timeout: Optional[float]) -> float:
    """Return half the watchdog timeout, capped at MAX_DEADLINE_S."""
    if not watchdog_timeout or watchdog_timeout <= 0:
        return MAX_DEADLINE_S
    return min(MAX_DEADLINE_S, 0.5 * float(watchdog_timeout))


def device_isa() -> Optional[str]:
    """Return the GPU ISA without feature suffixes, or None without a GPU."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if torch.version.hip:
        return props.gcnArchName.split(":")[0]
    return f"sm_{props.major}{props.minor}"


def _ragged_verify(cfg: Any) -> bool:
    mode = envs.SGLANG_RAGGED_VERIFY_MODE.get().lower()
    return cfg.speculative_algorithm is not None and mode != "static"


# PARTITIONED_GPU needs the device, so install checks it
_PROBES = {
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
    Feature.EAGLE3_HIDDEN_STATES: lambda c: (
        (c.speculative_algorithm or "").upper() == "EAGLE3"
    ),
    Feature.KV_OFFLOAD: lambda c: bool(
        c.enable_hierarchical_cache
        or c.disaggregation_decode_enable_offload_kvcache
        or c.enable_lmcache
        or c.enable_flexkv
    ),
    Feature.MEMORY_RELEASE: lambda c: bool(
        c.enable_memory_saver or c.enable_weights_cpu_backup
    ),
    Feature.CPU_OFFLOAD: lambda c: (
        (c.cpu_offload_gb or 0) > 0 or (c.offload_group_size or 0) > 0
    ),
    Feature.TORCH_COMPILE: lambda c: bool(c.enable_torch_compile),
}


def features_in_use(cfg: Any) -> set[Feature]:
    return {feature for feature, probe in _PROBES.items() if probe(cfg)}


def monokernel_disable_reasons(
    *, cfg: Any, model_config: ModelConfig, isa: Optional[str]
) -> tuple[Optional[MonoSpec], list[str]]:
    """Return the matching spec and all configuration rejection reasons."""
    if isa is None:
        return None, ["no GPU is visible"]
    hf_config = model_config.hf_config
    spec, why = find_spec(
        backend=BACKEND, architectures=hf_config.architectures, isa=isa
    )
    if spec is None:
        return None, [why]
    reasons = [
        f"{f.value} is in use and the spec does not accept it"
        for f in sorted(features_in_use(cfg) - spec.features, key=lambda f: f.name)
    ]
    req = spec.requires
    dtype = str(model_config.dtype).replace("torch.", "")  # resolved, e.g. bfloat16
    _, decode_backend = attention_backends_of(cfg)
    one_of = [
        ("tp_size", cfg.tp_size, spec.devices[isa].tp_sizes),
        ("dtype", dtype, req.dtypes),
        # Served steps run on the decode-phase backend.
        ("decode attention backend", decode_backend, req.attention_backends),
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
    if spec.numerics.kind is not NumericsKind.BITWISE:
        for flag, value in (
            ("--enable-deterministic-inference", cfg.enable_deterministic_inference),
            ("--rl-on-policy-target", cfg.rl_on_policy_target),
        ):
            if value:
                reasons.append(f"{flag} needs a BITWISE spec, and this one is not")
    if spec.config_disable_reasons is not None:
        reasons += spec.config_disable_reasons(cfg, hf_config)
    return spec, reasons


def covered_capture_sizes(
    *, widths: tuple[int, ...], tokens_per_request: int, sizes: list[int]
) -> list[int]:
    """Align supported buckets without raising the configured capture ceiling."""
    max_bs = max(sizes, default=0)
    covered = sorted(
        {
            w // tokens_per_request
            for w in widths
            if w % tokens_per_request == 0 and w // tokens_per_request <= max_bs
        }
    )
    if not covered:
        return list(sizes)
    return covered + sorted(b for b in set(sizes) if b > covered[-1])


def handle_amd_monokernel(server_args: ServerArgs) -> None:
    """Validate --enable-amd-monokernel and resolve its decode capture sizes."""
    cfg = resolving_view(server_args)
    if not cfg.enable_amd_monokernel:
        return
    isa = device_isa()
    spec, reasons = monokernel_disable_reasons(
        cfg=cfg, model_config=model_config_of(server_args), isa=isa
    )
    if reasons:
        raise ValueError(
            f"{FLAG} cannot run this setup:\n  - " + "\n  - ".join(reasons)
        )
    tokens = (
        int(cfg.speculative_num_draft_tokens or 1) if cfg.speculative_algorithm else 1
    )
    graph = cfg.cuda_graph_config
    explicit_bs = (Phase.DECODE, "bs") in getattr(
        server_args, "_cuda_graph_config_locked", set()
    )
    if graph.decode.backend != Backend.DISABLED and not explicit_bs:
        sizes = list(graph.decode.bs or [])
        new = covered_capture_sizes(
            widths=spec.widths(isa), tokens_per_request=tokens, sizes=sizes
        )
        if sizes and new != sorted(set(sizes)):
            declare_resolution(
                server_args,
                "handle_amd_monokernel",
                cuda_graph_config=with_phase(
                    graph, Phase.DECODE, bs=new, max_bs=max(new)
                ),
            )
    logger.warning(
        "%s is experimental. Its kernels, coverage and plugin API may change.", FLAG
    )
    logger.info(
        "%s: %s on %s, widths %s (%d tokens a request)",
        FLAG,
        spec.name,
        isa,
        list(spec.widths(isa)),
        tokens,
    )
