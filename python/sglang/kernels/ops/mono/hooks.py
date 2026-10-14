# SPDX-License-Identifier: Apache-2.0
"""The monokernel plugin's hooks (RFC #43949): the calls SGLang makes. Each is a
no-op while ``--enable-monokernel`` is off.

- ``handle_monokernel(server_args)``, the config hook: before any process starts,
  find the spec, refuse an unsupported setup with every reason, and set the
  decode capture sizes.
- ``load_tensor(model, name, tensor)``: each checkpoint tensor, for the copies of
  weights that loading rewrites.
- ``attach(model, hf_config)``: after weight load, before KV sizing. Every rank's
  reasons are gathered and refused together; then build every planned width and
  compare every rank's routing plans.
- ``route(model, forward_batch, **state)``: one plan a forward, from the graph key.
- ``kernel(plan, layer_id, point)``: the span that starts at ``point``, or None.

A failed wait stops its launch and flags every peer, whose waits stop too
(``common_hip/guard.py``): a rank finishes a step only with every peer's data for
it, so no token of a failed step is sent and SGLang stops the process tree.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

from sglang.kernels.ops.mono.spec import (
    BuildContext,
    Feature,
    NumericsKind,
    ShapeKey,
    SpanPoint,
    find_spec,
)

logger = logging.getLogger(__name__)

# The SGLang names this module reads, by file under python/sglang/ (``f(p)``: a
# parameter, ``"x"``: a string literal). test_mono_contract pins them, so renaming
# one in SGLang fails a CPU test that names this file, not a server's startup.
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": (
        "enable_monokernel",
        "disable_cuda_graph",
        "disable_decode_cuda_graph",
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
    "srt/distributed/parallel_state.py": ("world_size", "cpu_group"),
    "srt/environ.py": ("envs", "SGLANG_RAGGED_VERIFY_MODE"),
    "srt/model_executor/cuda_graph_config.py": ("Phase", "DECODE", "with_phase"),
    "srt/model_executor/forward_batch_info.py": (
        "forward_mode",
        "batch_size",
        "input_ids",
        "is_decode",
        "is_target_verify",
    ),
    "srt/model_executor/model_runner_components/weight_updater.py": (
        '"_derived_weight_cache_error"',
    ),
    "srt/runtime_context.py": (
        "get_device",
        "get_exec",
        "get_parallel",
        "get_spec",
        '"tp_group"',
    ),
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


# ---------------------------------------------------------------- config hook
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


# ---------------------------------------------------------------- in each process
def enabled() -> bool:
    from sglang.srt.runtime_context import get_exec

    return bool(get_exec().kernel.enable_monokernel)


def _gather(obj) -> list:
    """Every TP rank's ``obj``, through the CPU group."""
    import torch.distributed as dist

    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    if group.world_size == 1:
        return [obj]
    out: list[Any] = [None] * group.world_size
    dist.all_gather_object(out, obj, group=group.cpu_group)
    return out


def load_tensor(model, name: str, tensor) -> None:
    """Each checkpoint tensor, before renames and quantization: the spec may keep a
    copy of a weight that loading rewrites."""
    spec = model.__dict__.get("_mono_load_spec", False)
    if spec is False:
        archs = getattr(model.config, "architectures", None) or ()
        spec = find_spec(archs, device_isa())[0] if enabled() else None
        model.__dict__["_mono_load_spec"] = spec
    if spec is not None and spec.load_tensor is not None:
        spec.load_tensor(
            model.__dict__.setdefault("_mono_copies", {}), model, name, tensor
        )


def shape_key(forward_batch) -> Optional[ShapeKey]:
    """The graph key of a DECODE or TARGET_VERIFY step, else None."""
    mode = forward_batch.forward_mode
    if mode.is_decode():
        name = "decode"
    elif mode.is_target_verify():
        name = "target_verify"
    else:
        return None
    bs, rows = int(forward_batch.batch_size), int(forward_batch.input_ids.shape[0])
    if bs <= 0 or rows % bs:
        return None
    return ShapeKey(name, bs, rows // bs, rows)


@dataclass(frozen=True)
class Plan:
    """One forward's routing decision; ``state`` holds the forward's values for the
    spec's kernels (``route``'s keywords, and what a graph capture records)."""

    key: ShapeKey
    built: Any  # the spec's built model
    state: dict = field(default_factory=dict, compare=False)


class MonoRuntime:
    """A model's built monokernels and the spans each graph key routes to them."""

    def __init__(self, built, layer_ids, widths):
        self.built = built
        self.layer_ids, self.widths = tuple(layer_ids), tuple(widths)
        self._spans: dict = {}  # key -> the (layer, point) pairs with a kernel

    def spans(self, key: ShapeKey) -> tuple:
        if key not in self._spans:
            built = key.rows in self.widths
            self._spans[key] = tuple(
                (i, p)
                for i in self.layer_ids
                for p in SpanPoint
                if built and self.built.kernel(i, p, key) is not None
            )
        return self._spans[key]

    def decision(self, key: ShapeKey) -> str:
        """The routing decision for ``key`` as text: what every rank must agree on."""
        spans = " ".join(f"{i}.{p.name.lower()}" for i, p in self.spans(key))
        return f"{key}: {spans or 'normal'}"


def attach(model, hf_config) -> None:
    """After weight load, before KV sizing: every rank's refusal reasons are listed
    and any refusal stops startup; then build every planned width and compare every
    rank's routing plans. Sets ``model.mono``, which ``route`` reads."""
    if not enabled():
        return
    import torch

    from sglang.srt.runtime_context import get_device

    isa = device_isa()
    spec, why = find_spec(getattr(hf_config, "architectures", None) or (), isa)
    model.__dict__.pop("_mono_load_spec", None)
    copies = model.__dict__.pop("_mono_copies", {})
    reasons = (
        [why]
        if spec is None
        else list(spec.model_disable_reasons(model, hf_config, copies))
    )
    if spec is not None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if props.multi_processor_count < spec.devices[isa].resident_blocks:
            reasons.append(
                f"{Feature.PARTITIONED_GPU.value}: {props.multi_processor_count} compute "
                f"units, the launch grid needs {spec.devices[isa].resident_blocks}"
            )
    # all ranks refuse together, before any handshake
    reasons = [f"rank {r}: {x}" for r, rs in enumerate(_gather(reasons)) for x in rs]
    if reasons:
        raise ValueError("--enable-monokernel: " + "; ".join(reasons))

    # the widths the capture plan routes (every width a step can have without graphs)
    keys, tokens = _capture_keys(), _tokens_per_request()
    rows = {k.rows for k in keys}
    widths = tuple(
        w for w in spec.widths(isa) if w % tokens == 0 and (not keys or w in rows)
    )
    ctx = BuildContext(isa, widths, deadline_s(get_device().watchdog_timeout), copies)
    t0, mem0 = time.perf_counter(), torch.cuda.memory_allocated()
    rt = MonoRuntime(
        spec.build(model, hf_config, ctx), sorted(spec.spans(hf_config)), widths
    )
    lines = [rt.decision(k) for k in keys]
    if len(set(_gather(hashlib.sha256("\n".join(lines).encode()).hexdigest()))) > 1:
        raise RuntimeError("monokernel routing plans differ across ranks")
    model.mono = rt
    # SGLang's weight updater refuses each update: the kernels read copies and views
    model._derived_weight_cache_error = (
        "Online weight updates are not supported with --enable-monokernel "
        f"({spec.name})"
    )
    logger.info(
        "monokernel %s on %s: %d of %d capture keys covered; widths %s built in %.1f s, "
        "%.1f MiB; deadline %.0f s (a failed wait stops every rank with a GPU fault at "
        "address 0x0); weight copies: %s",
        spec.name,
        isa,
        sum(not line.endswith(": normal") for line in lines),
        len(lines),
        list(widths),
        time.perf_counter() - t0,
        (torch.cuda.memory_allocated() - mem0) / 2**20,
        ctx.deadline_s,
        _sizes(copies),
    )


def _sizes(copies: dict) -> str:
    kinds: dict = {}
    for key, t in copies.items():
        n, b = kinds.get(key[0], (0, 0))
        kinds[key[0]] = (n + 1, b + t.numel() * t.element_size())
    return ", ".join(
        f"{k} {n} x {b / 2**20:.1f} MiB" for k, (n, b) in sorted(kinds.items())
    )


def _tokens_per_request() -> int:
    from sglang.srt.runtime_context import get_spec

    spec_cfg = get_spec()
    if spec_cfg.speculative_algorithm is None:
        return 1
    return int(spec_cfg.speculative_num_draft_tokens or 1)


def _capture_keys() -> list[ShapeKey]:
    """The decode graph keys: DECODE, or TARGET_VERIFY with speculative decoding."""
    from sglang.srt.runtime_context import get_exec

    graph = get_exec().graph
    if graph.disable_cuda_graph or graph.disable_decode_cuda_graph:
        return []
    decode = getattr(graph.cuda_graph_config, "decode", None)
    tokens = _tokens_per_request()
    mode = "decode" if tokens == 1 else "target_verify"
    sizes = sorted(set(getattr(decode, "bs", None) or []))
    return [ShapeKey(mode, bs, tokens, bs * tokens) for bs in sizes]


def route(model, forward_batch, **state) -> Optional[Plan]:
    """One plan a forward from the graph key alone; None: the normal path.
    ``state``: the forward's values the spec's kernels read."""
    rt = vars(model).get("mono")  # no nn.Module attribute miss on the normal path
    if rt is None:
        return None
    key = shape_key(forward_batch)
    return Plan(key, rt.built, state) if key is not None and rt.spans(key) else None


def kernel(plan: Optional[Plan], layer_id: int, point):
    """The span of ``layer_id`` that starts at ``point`` (a ``SpanPoint`` or its
    lower-case name) under ``plan``, bound to the plan; None: the normal path."""
    if plan is None:
        return None
    point = SpanPoint[point.upper()] if isinstance(point, str) else point
    fn = plan.built.kernel(layer_id, point, plan.key)
    return None if fn is None else partial(fn, plan)
