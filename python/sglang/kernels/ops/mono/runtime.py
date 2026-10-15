"""The monokernel runtime (RFC #43949 sections 2 and 5): the calls SGLang makes in each process, with the switch off
each a no-op.

  load_weights(), each tensor         tensor = load_tensor(model, name, tensor); loaded(model) at the end
  precompile_kernels_after_loading()  attach(loop_module, architecture)
  the layer loop                      plan = route(loop_module, forward_batch) once a forward;
                                      span = kernel(plan, i, "layer_in") at each span start (None: the normal layer)

A step not routed to the kernels runs the model's layers exactly as before; the module tree, parameter names and weight
loading are untouched by the switch. Routing is a function of what every rank sees alike (forward mode, row count,
batch size), so under CUDA graphs the decision baked in at capture is what every replay repeats.
"""

from __future__ import annotations

import functools
import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional

import torch
import torch.distributed as dist

from sglang.kernels.ops.mono import check, guards
from sglang.kernels.ops.mono.spec import (
    Feature,
    LayerKernel,
    MonoContext,
    MonoModel,
    MonoSpec,
    MonoStep,
    Point,
    Span,
    check_spans,
    load_spec,
    span_label,
)

logger = logging.getLogger(__name__)

# the SGLang names this module reads (pinned by test_mono_contract)
SGLANG_READS = (
    "sglang.srt.model_executor.cuda_graph_config.Backend",
    "sglang.srt.runtime_context.get_exec",
    "sglang.srt.runtime_context.get_parallel",
    "sglang.srt.runtime_context.get_server_args",
)

_RUNTIME = "_mono_runtime"  # attribute of the layer-loop module that holds its runtime


class MonoUnsupported(Exception):
    """The loaded model or runtime configuration is outside what the kernels serve."""


def tp_agree(ok: bool, group) -> bool:
    """True when every rank of ``group`` (a CPU / gloo group, or None for one rank) passed ``ok`` true. A kernel waits
    in-kernel on every peer, so a rank that turned monokernels off alone, or failed to build what the others launch,
    must take every other rank with it; every rank calls this at the same point."""
    if group is None:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def _server_args():
    from sglang.srt.runtime_context import get_server_args

    return get_server_args()


@functools.cache
def _spec(architecture: str) -> Optional[MonoSpec]:
    return load_spec(architecture)


# ---- routing ----


@dataclass(frozen=True)
class Decision:
    step: Optional[MonoStep]  # None: the model's own path
    reason: str  # why not, for metrics ("" when routed)


def decide(
    spec: MonoSpec,
    ctx: MonoContext,
    *,
    is_decode: bool,
    is_verify: bool,
    rows: int,
    batch_size: int,
) -> Decision:
    if not (is_decode or is_verify):
        return Decision(None, "not decode")
    if is_verify and Feature.SPEC_VERIFY not in spec.features:
        return Decision(None, "verify unsupported")
    if rows not in ctx.limits.widths:
        return Decision(None, "width")
    if batch_size < 1 or rows % batch_size:
        return Decision(None, "rows not a multiple of the batch")
    step = MonoStep(
        rows=rows,
        batch_size=batch_size,
        is_verify=is_verify,
        query_len=rows // batch_size,
    )
    why = spec.eligible(ctx, step)
    if why:
        return Decision(None, why)
    return Decision(step, "")


def decide_for_batch(spec: MonoSpec, ctx: MonoContext, forward_batch) -> Decision:
    mode = forward_batch.forward_mode
    rows = (
        int(forward_batch.input_ids.shape[0])
        if forward_batch.input_ids is not None
        else 0
    )
    return decide(
        spec,
        ctx,
        is_decode=mode.is_decode(),
        is_verify=mode.is_target_verify(),
        rows=rows,
        batch_size=int(forward_batch.batch_size),
    )


# ---- the runtime a model holds ----


class Route(NamedTuple):
    spans: tuple[Span, ...]
    kernel: LayerKernel
    step: MonoStep
    cache: dict  # per-step memo the spec's glue may fill once and share across the step's layers
    checker: Optional[check.Checker] = (
        None  # set on the steps check mode compares against the module path
    )


class MonoRuntime:
    def __init__(self, spec: MonoSpec, ctx: MonoContext, model: MonoModel) -> None:
        self.spec, self.ctx, self.model = spec, ctx, model
        self.spans = dict(model.spans)
        check_spans(self.spans)
        self._kernels: dict[int, LayerKernel] = {}
        self.decision = Decision(None, "no step yet")
        self.step_cache: dict = {}
        self.counts: Counter = Counter()
        self.state = dict(spec.state_bindings(ctx, model))
        self.checker = check.Checker(ctx.tp_rank) if check.requested() else None
        self._checking = False
        logger.info(
            "mono kernels on: spec=%s widths=%s layers=%d (%s)",
            spec.name,
            list(ctx.limits.widths),
            len(self.spans),
            dict(Counter(span_label(s) for s in self.spans.values())),
        )

    def begin_step(self, forward_batch) -> Decision:
        self.step_cache = {}
        d = decide_for_batch(self.spec, self.ctx, forward_batch)
        if d.step is not None and d.step.rows not in self._kernels:
            d = Decision(None, "width not built")
        self.decision = d
        self._checking = (
            d.step is not None
            and self.checker is not None
            and self.checker.begin_step()
        )
        self.counts["mono" if d.step is not None else f"fallback:{d.reason}"] += 1
        if torch.cuda.is_current_stream_capturing() and self.ctx.tp_rank == 0:
            # replays never reach Python: the capture is where a graph's path is decided
            logger.info(
                "mono graph capture: %d rows -> %s",
                int(forward_batch.input_ids.shape[0]),
                "mono" if d.step is not None else f"model path ({d.reason})",
            )
        return d

    def routed(self, layer_id: int) -> Optional[Route]:
        """The route when this step runs layer ``layer_id`` on the kernels, else None."""
        step = self.decision.step
        if step is None:
            return None
        spans = self.spans.get(layer_id)
        if spans is None:
            return None
        binding = self.state.get(layer_id)
        if binding is not None:
            binding.wait_ready()
        return Route(
            spans,
            self._kernels[step.rows],
            step,
            self.step_cache,
            self.checker if self._checking else None,
        )

    def build_widths(self, widths) -> dict[int, int]:
        """Build every width now, collectively (bytes of scratch each allocated); any rank failing refuses mono."""
        scratch = {}
        for rows in widths:
            before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            kernel = None
            try:
                kernel = self.model.kernel(rows)
            except Exception:
                logger.exception(
                    "mono %s: building the %d-row kernels failed", self.spec.name, rows
                )
            if not tp_agree(kernel is not None, self.ctx.cpu_group):
                guards.raise_if_refused(
                    "model",
                    self.spec.name,
                    [f"building the {rows}-row kernels failed on a TP rank"],
                )
            self._kernels[rows] = kernel
            if torch.cuda.is_available():
                scratch[rows] = torch.cuda.memory_allocated() - before
        return scratch


def attach_spec(
    model,
    spec: MonoSpec,
    *,
    server_args,
    hf_config,
    tp_rank: int,
    tp_size: int,
    cpu_group,
    device,
) -> MonoRuntime:
    """The runtime of ``spec`` on ``model``, its widths not built yet."""
    guards.raise_if_refused("model", spec.name, list(spec.model_disable_reasons(model)))
    isa = guards.current_isa()
    limits = guards.device_limits(spec, isa)
    if limits is None:
        guards.raise_if_refused(
            "model", spec.name, [f"no {spec.name} mono kernels for {isa or 'this GPU'}"]
        )
    ctx = MonoContext(
        server_args=server_args,
        hf_config=hf_config,
        tp_size=tp_size,
        tp_rank=tp_rank,
        cpu_group=cpu_group,
        device=torch.device(device),
        limits=limits,
    )
    return MonoRuntime(spec, ctx, spec.build(model, ctx))


# ---- weight loading ----


def load_tensor(model, name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Each checkpoint tensor, before renames and quantization; returns it unchanged."""
    if not guards.mono_enabled(_server_args()):
        return tensor
    spec = _spec(type(model).__name__)
    if spec is not None and spec.load_tensor is not None:
        spec.load_tensor(model, name, tensor)
    return tensor


def loaded(model) -> None:
    """The end of ``load_weights()``."""


# ---- after loading, before KV sizing ----


def _step_widths(
    spec_rows: tuple[int, ...], widths: tuple[int, ...]
) -> tuple[int, ...]:
    """The widths steps can reach: each decode capture size times each step's rows a request; every width without
    decode graphs (eager, debugging only)."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.runtime_context import get_exec

    cg = get_exec().graph.cuda_graph_config
    if cg is None or cg.decode.backend == Backend.DISABLED:
        return tuple(widths)
    return tuple(
        sorted({bs * q for bs in cg.decode.bs for q in spec_rows} & set(widths))
    )


def attach(model, architecture: str) -> None:
    """Attach the runtime of ``architecture``'s spec to ``model`` (the module whose forward runs the layer loop):
    check the weights (all ranks refuse together), build every width a graph capture reaches, and compare the ranks'
    plans. Raises ``MonoRefused`` with every reason."""
    if not guards.mono_enabled(_server_args()):
        return
    from sglang.srt.runtime_context import get_parallel

    spec = _spec(architecture)
    if spec is None:
        raise guards.MonoRefused(f"{guards.SWITCH}: no mono spec for {architecture}")
    tp = get_parallel().tp_group
    t0 = time.perf_counter()
    rt = attach_spec(
        model,
        spec,
        server_args=_server_args(),
        hf_config=model.config,
        tp_rank=tp.rank_in_group,
        tp_size=tp.world_size,
        cpu_group=tp.cpu_group,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    guards.raise_if_refused(
        "model", spec.name, guards.device_refusals(spec, rt.ctx.limits)
    )
    widths = _step_widths(rt.model.rows_per_request, rt.ctx.limits.widths)
    scratch = rt.build_widths(widths)
    _compare_plans(rt, widths)
    setattr(model, _RUNTIME, rt)
    # the weight updater refuses online updates on this (the kernels hold views and copies of the loaded weights)
    model._derived_weight_cache_error = (
        f"Online weight updates are not supported with {guards.SWITCH}: the {spec.name} monokernels "
        "hold views and copies of the loaded weights."
    )
    if rt.ctx.tp_rank == 0:
        logger.info(
            "mono %s: attached in %.1f s; built widths %s; scratch MiB per width %s",
            spec.name,
            time.perf_counter() - t0,
            list(widths),
            {w: round(b / 2**20, 1) for w, b in scratch.items()},
        )


def _compare_plans(rt: MonoRuntime, widths: tuple[int, ...]) -> None:
    """Every rank routes the same layers at the same widths, or none starts."""
    plan = (
        rt.spec.name,
        tuple(
            sorted((i, tuple(s.name for s in spans)) for i, spans in rt.spans.items())
        ),
        widths,
    )
    group = rt.ctx.cpu_group
    if group is None:
        return
    plans = [None] * dist.get_world_size(group)
    dist.all_gather_object(plans, plan, group=group)
    if any(p != plan for p in plans):
        raise guards.MonoRefused(
            f"{guards.SWITCH} refused ({rt.spec.name}): the TP ranks' plans differ"
        )


# ---- the layer loop ----


class Plan(NamedTuple):
    runtime: MonoRuntime


def route(model, forward_batch) -> Optional[Plan]:
    """Once a forward: the step's plan, or None for the normal path (also with the switch off)."""
    rt: Optional[MonoRuntime] = getattr(model, _RUNTIME, None)
    if rt is None:
        return None
    if rt.begin_step(forward_batch).step is None:
        return None
    return Plan(rt)


def kernel(
    plan: Optional[Plan], layer_id: int, point: str
) -> Optional[Callable[..., Any]]:
    """At each span start: the span that runs layer ``layer_id`` from ``point``, or None for the normal path. The
    span takes and returns exactly what the code it replaces does."""
    if plan is None:
        return None
    route_ = plan.runtime.routed(layer_id)
    if route_ is None or route_.spans[0].start is not Point[point.upper()]:
        return None
    return plan.runtime.model.span(route_)
