# SPDX-License-Identifier: Apache-2.0
"""Startup refusal for ``--enable-monokernel``: every reason at once, before any weight loads (RFC #43949 section 7)."""

from __future__ import annotations

import logging
from typing import Any

from sglang.experimental.monokernels.spec import Feature, MonoSpec, load_spec
from sglang.srt.arg_groups.overrides import model_config_of, resolving_view
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

SWITCH = "--enable-monokernel"
# every cross-workgroup or cross-GPU wait stops after half the watchdog timeout, at most 60 s; under 5 s refused
MIN_DEADLINE_S = 5.0
MAX_DEADLINE_S = 60.0

SGLANG_READS = {
    "srt/arg_groups/overrides.py": ("resolving_view", "model_config_of"),
    "srt/environ.py": (
        "envs",
        "SGLANG_TEST_MONO_SPANS",
        "SGLANG_TEST_MONO_FAULT_INJECT",
        "SGLANG_TEST_MONO_CHECK_NUMERICS",
    ),
    "srt/arg_groups/fields/parallel.py": (
        "tp_size",
        "pp_size",
        "nnodes",
        "enable_dp_attention",
        "ep_size",
        "attn_cp_size",
        "dcp_size",
    ),
    "srt/arg_groups/fields/exec_.py": (
        "enable_monokernel",
        "enable_eplb",
        "enable_two_batch_overlap",
        "enable_memory_saver",
        "cpu_offload_gb",
        "enable_weights_cpu_backup",
        "enable_torch_compile",
        "enable_deterministic_inference",
        "rl_on_policy_target",
        "disable_cuda_graph",
        "cuda_graph_config",
        "debug_cuda_graph",
        "enable_return_routed_experts",
        "enable_return_indexer_topk",
        "enable_return_hidden_states",
    ),
    "srt/model_executor/cuda_graph_config.py": (
        "decode",
        "backend",
        '"disabled"',
        '"breakable"',
    ),
    "srt/arg_groups/fields/spec.py": (
        "speculative_algorithm",
        "speculative_eagle_topk",
    ),
    "srt/arg_groups/fields/lora.py": ("enable_lora", "lora_paths"),
    "srt/arg_groups/fields/disagg.py": ("disaggregation_mode", '"null"'),
    "srt/arg_groups/fields/memory.py": ("enable_hierarchical_cache",),
    "srt/arg_groups/fields/device.py": ("watchdog_timeout",),
}


class MonoRefused(ValueError):
    """The configuration or the loaded model is outside what the model's monokernels serve."""


def _decode_graph(c) -> str:
    """The resolved decode CUDA graph backend (``--disable-decode-cuda-graph`` and ``--cuda-graph-backend-decode``
    fold into it)."""
    return c.cuda_graph_config.decode.backend


# feature -> whether the resolved configuration uses it; a probe that cannot read its setting counts as in use
_PROBES: dict = {
    Feature.PIPELINE_PARALLEL: lambda c: (c.pp_size or 1) > 1,
    Feature.MULTI_NODE_TP: lambda c: (c.nnodes or 1) > 1,
    Feature.DP_ATTENTION: lambda c: bool(c.enable_dp_attention),
    Feature.EXPERT_PARALLEL: lambda c: (c.ep_size or 1) > 1,
    Feature.EPLB: lambda c: bool(c.enable_eplb),
    Feature.CONTEXT_PARALLEL: lambda c: (c.attn_cp_size or 1) > 1,
    Feature.DECODE_CONTEXT_PARALLEL: lambda c: (c.dcp_size or 1) > 1,
    Feature.SPECULATIVE: lambda c: c.speculative_algorithm is not None,
    Feature.UNEVEN_VERIFY: lambda c: (c.speculative_eagle_topk or 1) > 1,
    Feature.TWO_BATCH_OVERLAP: lambda c: bool(c.enable_two_batch_overlap),
    Feature.LORA: lambda c: bool(c.enable_lora) or bool(c.lora_paths),
    Feature.PD_DISAGGREGATION: lambda c: (c.disaggregation_mode or "null") != "null",
    Feature.KV_OFFLOAD: lambda c: bool(c.enable_hierarchical_cache),
    Feature.MEMORY_RELEASE: lambda c: bool(c.enable_memory_saver),
    Feature.CPU_OFFLOAD: lambda c: (c.cpu_offload_gb or 0) > 0,
    Feature.WEIGHT_BACKUP: lambda c: bool(c.enable_weights_cpu_backup),
    Feature.TORCH_COMPILE: lambda c: bool(c.enable_torch_compile),
    Feature.DETERMINISTIC: lambda c: bool(c.enable_deterministic_inference),
    Feature.RL_ON_POLICY: lambda c: c.rl_on_policy_target is not None,
    Feature.EAGER: lambda c: (
        bool(c.disable_cuda_graph) or _decode_graph(c) == "disabled"
    ),
    Feature.BREAKABLE_DECODE_GRAPH: lambda c: (
        bool(c.debug_cuda_graph) or _decode_graph(c) == "breakable"
    ),
    Feature.ROUTED_EXPERT_CAPTURE: lambda c: bool(c.enable_return_routed_experts),
    Feature.INDEXER_TOPK_CAPTURE: lambda c: bool(c.enable_return_indexer_topk),
    Feature.HIDDEN_STATE_CAPTURE: lambda c: bool(c.enable_return_hidden_states),
}


def feature_reasons(cfg: Any, spec: MonoSpec, probes: dict = _PROBES) -> list:
    reasons = []
    for feature, probe in probes.items():
        try:
            used = probe(cfg)
        except Exception as err:
            used, why = True, f" (its setting cannot be read: {type(err).__name__})"
        else:
            why = ""
        if used and feature not in spec.features:
            reasons.append(f"{feature.value} is not supported{why}")
    return reasons


def deadline_s(watchdog_timeout) -> float:
    """The deadline of every guarded wait: half the watchdog timeout, at most MAX_DEADLINE_S."""
    if not watchdog_timeout or watchdog_timeout <= 0:
        return MAX_DEADLINE_S
    return min(float(watchdog_timeout) / 2, MAX_DEADLINE_S)


def deadline_reasons(watchdog_timeout) -> list:
    d = deadline_s(watchdog_timeout)
    if d < MIN_DEADLINE_S:
        return [
            f"--watchdog-timeout {watchdog_timeout} gives a {d:g} s wait deadline; "
            f"monokernels need at least {MIN_DEADLINE_S:g} s (a watchdog of {2 * MIN_DEADLINE_S:g} s)"
        ]
    return []


def span_selection(spec: MonoSpec) -> tuple:
    """The spans SGLANG_TEST_MONO_SPANS selects (default all) and the reasons for unknown names."""
    raw = envs.SGLANG_TEST_MONO_SPANS.get() or ""
    names = [n.strip() for n in raw.split(",") if n.strip()]
    if not names:
        return frozenset(spec.span_names), []
    unknown = sorted(set(names) - set(spec.span_names))
    if unknown:
        return frozenset(), [
            f"SGLANG_TEST_MONO_SPANS names {unknown}; {spec.name} has {list(spec.span_names)}"
        ]
    return frozenset(names), []


def fault_injection(spec: MonoSpec, tp_size: int = 0) -> tuple:
    """SGLANG_TEST_MONO_FAULT_INJECT=<rank>:<site>[:<after>] as (rank, site, after) or None, and the reasons
    a set value is malformed: a rank outside the TP size, or a site the spec does not name."""
    raw = envs.SGLANG_TEST_MONO_FAULT_INJECT.get() or ""
    if not raw:
        return None, []
    want = f"SGLANG_TEST_MONO_FAULT_INJECT={raw!r}: want <rank>:<site>[:<after>]"
    parts = raw.split(":")
    if not (
        len(parts) in (2, 3)
        and parts[0].isdigit()
        and parts[1]
        and (len(parts) == 2 or parts[2].isdigit())
    ):
        return None, [want]
    rank, site, after = int(parts[0]), parts[1], int(parts[2]) if len(parts) == 3 else 0
    if tp_size and rank >= tp_size:
        return None, [f"{want}, with a rank below the TP size {tp_size}"]
    if spec.wait_sites and site not in spec.wait_sites:
        return None, [f"{want}; {spec.name} has the wait sites {list(spec.wait_sites)}"]
    return (rank, site, after), []


def config_refusals(cfg: Any, hf_config: Any, spec: MonoSpec) -> list:
    reasons = feature_reasons(cfg, spec)
    reasons += deadline_reasons(cfg.watchdog_timeout)
    reasons += span_selection(spec)[1]
    reasons += fault_injection(spec, cfg.tp_size or 0)[1]
    check = (envs.SGLANG_TEST_MONO_CHECK_NUMERICS.get() or "").strip()
    if check not in ("", "log", "enforce"):
        reasons.append(
            f"SGLANG_TEST_MONO_CHECK_NUMERICS={check!r}: want log or enforce"
        )
    reasons += list(spec.config_disable_reasons(cfg, hf_config))
    return reasons


def raise_refused(spec_name: str, stage: str, reasons: list) -> None:
    if reasons:
        raise MonoRefused(
            f"{SWITCH} ({spec_name}) refused at {stage}:\n  - " + "\n  - ".join(reasons)
        )


def handle_monokernel(server_args: Any) -> None:
    """The config hook. With the switch on, refuse startup when the model has no spec or the resolved
    deployment uses anything the spec does not support. Runs after every other resolution step."""
    cfg = resolving_view(server_args)
    if not cfg.enable_monokernel:
        return
    hf_config = model_config_of(server_args).hf_config
    arch = (getattr(hf_config, "architectures", None) or ["?"])[0]
    spec = load_spec(arch)
    if spec is None:
        raise MonoRefused(f"{SWITCH}: no monokernel spec for {arch}")
    raise_refused(spec.name, "startup", config_refusals(cfg, hf_config, spec))
    logger.info(
        "%s: the %s spec accepts this configuration; its kernels read %s",
        SWITCH,
        spec.name,
        spec.requires,
    )
