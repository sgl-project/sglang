# SPDX-License-Identifier: Apache-2.0
"""The monokernel plugin's refusals (RFC #43949), before any process starts.

``handle_monokernel(server_args)``, the config hook, finds the spec, refuses an
unsupported setup with every reason, and sets the decode capture sizes; it is a
no-op while ``--enable-monokernel`` is off. ``runtime`` refuses again after load,
with the loaded weights. The deadline of every wait is set here too: half of
``--watchdog-timeout``, and one under ``MIN_DEADLINE_S`` is refused.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sglang.kernels.ops.mono.spec import Feature, NumericsKind, find_spec

logger = logging.getLogger(__name__)

# The SGLang names this module reads, by file under python/sglang/ (``f(p)``: a
# parameter, ``"x"``: a string literal). test_mono_contract pins them, so renaming
# one in SGLang fails a CPU test that names this file, not a server's startup.
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": (
        "enable_monokernel",
        "cuda_graph_config",
        "watchdog_timeout",
        "speculative_algorithm",
        "speculative_num_draft_tokens",
        "tp_size",
        "dtype",
        "kv_cache_dtype",
        "attention_backend",
        "moe_runner_backend",
        # the probes of the refused-by-default features
        "pp_size",
        "nnodes",
        "enable_dp_attention",
        "attn_dp_size",
        "ep_size",
        "enable_eplb",
        "attn_cp_size",
        "dcp_size",
        "enable_prefill_cp",
        "enable_two_batch_overlap",
        "enable_single_batch_overlap",
        "enable_lora",
        "lora_paths",
        "disaggregation_mode",
        "enable_return_routed_experts",
        "enable_return_indexer_topk",
        "enable_hierarchical_cache",
        "disaggregation_decode_enable_offload_kvcache",
        "enable_lmcache",
        "enable_flexkv",
        "enable_memory_saver",
        "enable_weights_cpu_backup",
        "cpu_offload_gb",
        "offload_group_size",
        "enable_torch_compile",
        # the modes that accept only a BITWISE spec
        "enable_deterministic_inference",
        "rl_on_policy_target",
    ),
    "srt/arg_groups/overrides.py": (
        "declare_resolution",
        "model_config_of",
        "resolving_view",
    ),
    "srt/environ.py": ("envs", "SGLANG_RAGGED_VERIFY_MODE"),
    "srt/model_executor/cuda_graph_config.py": ("Phase", "DECODE", "with_phase"),
}

# A wait's deadline is below --watchdog-timeout, and long enough that a normal
# delay on one rank is not a failure.
MAX_DEADLINE_S, MIN_DEADLINE_S = 60.0, 5.0


def deadline_s(watchdog_timeout: Optional[float]) -> float:
    """Half the watchdog, at most ``MAX_DEADLINE_S``."""
    if not watchdog_timeout or watchdog_timeout <= 0:
        return MAX_DEADLINE_S
    return min(MAX_DEADLINE_S, 0.5 * float(watchdog_timeout))


def device_isa() -> Optional[str]:
    """The GPU's ISA name, e.g. ``gfx950`` (feature suffixes dropped); None without one."""
    import torch

    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = getattr(props, "gcnArchName", None)
    return arch.split(":")[0] if arch else f"sm_{props.major}{props.minor}"


def _on(cfg, *names) -> bool:
    return any(bool(getattr(cfg, name, False)) for name in names)


def _over(cfg, name, n) -> bool:
    return (getattr(cfg, name, None) or 0) > n


def _ragged_verify(cfg) -> bool:
    from sglang.srt.environ import envs

    mode = envs.SGLANG_RAGGED_VERIFY_MODE.get().lower()
    return cfg.speculative_algorithm is not None and mode != "static"


# A probe that fails counts as in use, so an unreadable setting refuses. Partitioned
# GPUs are refused at attach, and weight updates by SGLang's weight updater.
_PROBES = {
    Feature.PIPELINE_PARALLEL: lambda c: c.pp_size > 1,
    Feature.MULTI_NODE_TP: lambda c: c.nnodes > 1,
    Feature.DP_ATTENTION: lambda c: (
        _on(c, "enable_dp_attention") or _over(c, "attn_dp_size", 1)
    ),
    Feature.EXPERT_PARALLEL: lambda c: (c.ep_size or 1) > 1,
    Feature.EPLB: lambda c: bool(c.enable_eplb),
    Feature.CONTEXT_PARALLEL: lambda c: (
        _over(c, "attn_cp_size", 1)
        or _over(c, "dcp_size", 1)
        or _on(c, "enable_prefill_cp")
    ),
    Feature.SPECULATIVE_DECODING: lambda c: c.speculative_algorithm is not None,
    Feature.RAGGED_VERIFY: _ragged_verify,
    Feature.TWO_BATCH_OVERLAP: lambda c: _on(
        c, "enable_two_batch_overlap", "enable_single_batch_overlap"
    ),
    Feature.LORA: lambda c: _on(c, "enable_lora", "lora_paths"),
    Feature.PD_DISAGGREGATION: lambda c: (c.disaggregation_mode or "null") != "null",
    Feature.RETURN_ROUTED_EXPERTS: lambda c: bool(c.enable_return_routed_experts),
    Feature.RETURN_INDEXER_TOPK: lambda c: _on(c, "enable_return_indexer_topk"),
    Feature.EAGLE3_HIDDEN_STATES: lambda c: (
        (c.speculative_algorithm or "").upper() == "EAGLE3"
    ),
    Feature.KV_OFFLOAD: lambda c: _on(
        c,
        "enable_hierarchical_cache",
        "disaggregation_decode_enable_offload_kvcache",
        "enable_lmcache",
        "enable_flexkv",
    ),
    Feature.MEMORY_RELEASE: lambda c: _on(
        c, "enable_memory_saver", "enable_weights_cpu_backup"
    ),
    Feature.CPU_OFFLOAD: lambda c: (
        _over(c, "cpu_offload_gb", 0) or _over(c, "offload_group_size", 0)
    ),
    Feature.TORCH_COMPILE: lambda c: bool(c.enable_torch_compile),
}


def features_in_use(cfg: Any) -> set[Feature]:
    used = set()
    for feature, probe in _PROBES.items():
        try:
            if probe(cfg):
                used.add(feature)
        except Exception:
            used.add(feature)
    return used


def monokernel_disable_reasons(cfg, model_config, isa):
    """(spec, every reason the setup is refused); spec is None when none fits."""
    if isa is None:
        return None, ["no GPU is visible"]
    hf_config = model_config.hf_config
    spec, why = find_spec(getattr(hf_config, "architectures", None) or (), isa)
    if spec is None:
        return None, [why]
    req = spec.requires
    # the model dtype as resolved from the checkpoint ("auto" otherwise)
    dtype = str(getattr(model_config, "dtype", None) or cfg.dtype).replace("torch.", "")
    reasons = [
        f"{f.value} is in use and the spec does not accept it"
        for f in sorted(features_in_use(cfg) - spec.features, key=lambda f: f.name)
    ]
    for what, value, allowed in (
        ("tp_size", cfg.tp_size, spec.devices[isa].tp_sizes),
        ("dtype", dtype, req.dtypes),
        ("--kv-cache-dtype", str(cfg.kv_cache_dtype), req.kv_cache_dtypes),
        ("attention backend", cfg.attention_backend, req.attention_backends),
        ("MoE runner backend", cfg.moe_runner_backend, req.moe_runner_backends),
        (
            "quantization",
            getattr(model_config, "quantization", None),
            req.quantizations,
        ),
    ):
        if allowed and value not in allowed:
            reasons.append(
                f"{what} {value!r} is not one of {sorted(map(str, allowed))}"
            )
    watchdog = getattr(cfg, "watchdog_timeout", None)
    if watchdog and deadline_s(watchdog) < MIN_DEADLINE_S:
        reasons.append(
            f"--watchdog-timeout {watchdog:g} s leaves the waits a deadline under "
            f"{MIN_DEADLINE_S:g} s"
        )
    if spec.numerics.kind is not NumericsKind.BITWISE:
        for flag, name in (
            ("--enable-deterministic-inference", "enable_deterministic_inference"),
            ("--rl-on-policy-target", "rl_on_policy_target"),
        ):
            if _on(cfg, name):
                reasons.append(f"{flag} needs a BITWISE spec, and this one is not")
    return spec, reasons + list(spec.config_disable_reasons(cfg, hf_config))


def covered_capture_sizes(
    widths, tokens_per_request: int, sizes: list[int]
) -> list[int]:
    """Up to the largest covered size, the decode capture sizes are the widths (rows)
    divided by the tokens a request; larger sizes stay as configured."""
    covered = sorted(
        {w // tokens_per_request for w in widths if w % tokens_per_request == 0}
    )
    if not covered:
        return list(sizes)
    return covered + sorted(b for b in set(sizes) if b > covered[-1])


def handle_monokernel(server_args) -> None:
    """The config hook (``arg_groups/pipeline.py``)."""
    from sglang.srt.arg_groups.overrides import (
        declare_resolution,
        model_config_of,
        resolving_view,
    )

    cfg = resolving_view(server_args)
    if not cfg.enable_monokernel:
        return
    from sglang.srt.model_executor.cuda_graph_config import Phase, with_phase

    isa = device_isa()
    spec, reasons = monokernel_disable_reasons(cfg, model_config_of(server_args), isa)
    if reasons:
        raise ValueError(
            "--enable-monokernel cannot run this setup:\n  - " + "\n  - ".join(reasons)
        )
    tokens = (
        int(cfg.speculative_num_draft_tokens or 1) if cfg.speculative_algorithm else 1
    )
    graph = cfg.cuda_graph_config
    sizes = list(getattr(getattr(graph, "decode", None), "bs", None) or [])
    new = covered_capture_sizes(spec.widths(isa), tokens, sizes)
    if sizes and new != sorted(set(sizes)):
        declare_resolution(
            server_args,
            "handle_monokernel",
            cuda_graph_config=with_phase(graph, Phase.DECODE, bs=new, max_bs=max(new)),
        )
    logger.info(
        "--enable-monokernel: %s on %s, widths %s (%d tokens a request)",
        spec.name,
        isa,
        list(spec.widths(isa)),
        tokens,
    )
