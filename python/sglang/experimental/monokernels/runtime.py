# SPDX-License-Identifier: Apache-2.0
"""The monokernel runtime in each TP process: the load hooks, attach in stages that every rank agrees on,
routing from the graph key, and span lookup (RFC #43949 sections 3, 6, 8 and 9)."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist

from sglang.experimental.monokernels.guards import (
    SWITCH,
    MonoRefused,
    deadline_s,
    fault_injection,
    raise_refused,
    span_selection,
)
from sglang.experimental.monokernels.spec import (
    MonoSpec,
    Point,
    StepKind,
    check_layer_spans,
    load_spec,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_device, get_exec, get_parallel, get_spec

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/runtime_context.py": ("get_device", "get_exec", "get_parallel", "get_spec"),
    "srt/environ.py": ("envs", "SGLANG_TEST_MONO_CHECK_NUMERICS"),
    "srt/arg_groups/fields/exec_.py": ("enable_monokernel", "cuda_graph_config"),
    "srt/arg_groups/fields/device.py": ("watchdog_timeout",),
    "srt/arg_groups/fields/spec.py": (
        "speculative_algorithm",
        "speculative_num_draft_tokens",
    ),
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
}


class MonokernelError(RuntimeError):
    """A monokernel invariant broke while serving."""


@dataclass(frozen=True)
class MonoContext:
    """This process's view: its TP rank, the CPU group for agreement, the device, the wait deadline,
    the spans SGLANG_TEST_MONO_SPANS selects, and an injected fault (site, after) for this rank or None."""

    tp_rank: int
    tp_size: int
    group: Any
    device: torch.device
    isa: str
    deadline_s: float
    spans: frozenset
    fault: Optional[tuple]


@dataclass(frozen=True)
class Plan:
    """One step's routing: the step type, its rows, the covered layers' spans, and the first layer of each span
    in this step type's plan."""

    kind: StepKind
    rows: int
    layers: Mapping[int, tuple]
    first: Mapping[str, int]
    runtime: "MonoRuntime"


class LoadState:
    """A model's monokernel state between its first checkpoint tensor and attach: the spec's copies,
    with their sizes for the startup log."""

    def __init__(self, spec: MonoSpec):
        self.spec = spec
        self.stage = "loading"
        self.data: dict = {}
        self.copies: dict = {}

    def keep(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Record a weight copy the kernels keep; the log lists each with its size."""
        self.copies[name] = tensor.numel() * tensor.element_size()
        return tensor


def enabled() -> bool:
    return bool(get_exec().features.enable_monokernel)


def _load_state(owner) -> Optional[LoadState]:
    state = getattr(owner, "_monokernel_load", None)
    if state is None:
        spec = load_spec(type(owner).__name__) if enabled() else None
        state = LoadState(spec) if spec is not None else False
        owner._monokernel_load = state
    return state or None


def load_tensor(owner, name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Each checkpoint tensor before the loader renames or quantizes it: the spec may keep a copy."""
    state = _load_state(owner)
    if state is not None and state.spec.load_tensor is not None:
        if state.stage != "loading":
            raise MonokernelError(f"{SWITCH}: a checkpoint tensor after {state.stage}")
        state.spec.load_tensor(state, owner, name, tensor)
    return tensor


def loaded(owner) -> None:
    """The end of load_weights, before process_weights_after_loading rewrites weights in place."""
    state = _load_state(owner)
    if state is None:
        return
    if state.stage != "loading":
        raise MonokernelError(f"{SWITCH}: loaded() after {state.stage}")
    if state.spec.loaded is not None:
        state.spec.loaded(state, owner)
    state.stage = "loaded"


def _fault(spec: MonoSpec, tp_rank: int) -> Optional[tuple]:
    """This rank's injected fault (site, after), or None; the value was checked by the config hook."""
    fault, reasons = fault_injection(spec)
    if reasons:
        raise MonoRefused("; ".join(reasons))
    if fault is None or fault[0] != tp_rank:
        return None
    return fault[1], fault[2]


def _context(spec: MonoSpec) -> MonoContext:
    p = get_parallel()
    device = torch.device("cuda", torch.cuda.current_device())
    isa = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    return MonoContext(
        tp_rank=p.tp_rank,
        tp_size=p.tp_size,
        group=p.tp_group.cpu_group,
        device=device,
        isa=isa,
        deadline_s=deadline_s(get_device().watchdog_timeout),
        spans=span_selection(spec)[0],
        fault=_fault(spec, p.tp_rank),
    )


def _gather(ctx: MonoContext, value) -> list:
    if ctx.tp_size == 1:
        return [value]
    everyone = [None] * ctx.tp_size
    dist.all_gather_object(everyone, value, group=ctx.group)
    return everyone


def agree(ctx: MonoContext, spec_name: str, stage: str, fn):
    """Run ``fn() -> (reasons, value)`` on this rank, then exchange every rank's reasons before any rank
    raises, so no rank is left waiting in a collective that its peers abandoned."""
    try:
        reasons, value = fn()
    except Exception as err:
        reasons, value = [f"{type(err).__name__}: {err}"], None
    everyone = _gather(ctx, [str(r) for r in reasons])
    raise_refused(
        spec_name,
        stage,
        [f"rank {r}: {m}" for r, rs in enumerate(everyone) for m in rs],
    )
    return value


def widths_for(sizes, verify_tokens: Optional[int], max_rows: int) -> dict:
    """Step rows a graph capture can reach, capped at the kernels' largest width: the decode capture sizes, or
    with speculative decoding (the target then runs verify steps, not decode steps) those sizes times the tokens
    a request verifies. The kernel tests take their widths from here too."""
    sizes = sorted({int(b) for b in sizes})
    if verify_tokens is None:
        return {StepKind.DECODE: tuple(b for b in sizes if 1 <= b <= max_rows)}
    if verify_tokens < 1:
        raise MonoRefused(
            f"{SWITCH}: speculative decoding with {verify_tokens} tokens a verify step"
        )
    return {
        StepKind.TARGET_VERIFY: tuple(
            b * verify_tokens for b in sizes if 1 <= b * verify_tokens <= max_rows
        )
    }


def _verify_tokens() -> Optional[int]:
    if get_spec().speculative_algorithm is None:
        return None
    return int(get_spec().speculative_num_draft_tokens or 0)


def reachable_widths(max_rows: int) -> dict:
    return widths_for(
        get_exec().graph.cuda_graph_config.decode.bs, _verify_tokens(), max_rows
    )


def _digest(layers: Mapping[int, tuple]) -> str:
    items = sorted(
        (i, tuple((s.name, int(s.start), int(s.end)) for s in spans))
        for i, spans in layers.items()
    )
    return hashlib.sha1(repr(items).encode()).hexdigest()[:12]


class MonoRuntime:
    """A model's attached monokernels in this process."""

    def __init__(self, spec: MonoSpec, ctx: MonoContext, owner, state: LoadState):
        self.spec, self.ctx, self.owner, self.state = spec, ctx, owner, state
        self.plans: dict = {}
        self.firsts: dict = {}
        self.widths: dict = {}
        self.verify_tokens: Optional[int] = None
        self.model = None
        self._unbuilt: set = set()

    def _local_plan(self, kind: StepKind) -> tuple:
        sp = self.spec.spans(self.owner, kind)
        layers = {}
        for i, spans in sp.layers.items():
            kept = tuple(s for s in spans if s.name in self.ctx.spans)
            if kept:
                check_layer_spans(i, kept)
                layers[i] = kept
        return layers, sp.notes

    def _plan(self, kind: StepKind) -> dict:
        """This step type's spans per layer: built on each rank, then the same digest on every rank or
        startup stops."""
        layers, notes = agree(
            self.ctx,
            self.spec.name,
            f"the {kind.value} plan",
            lambda: ([], self._local_plan(kind)),
        )
        digest = _digest(layers)
        digests = _gather(self.ctx, digest)
        if len(set(digests)) != 1:
            raise_refused(
                self.spec.name,
                f"the {kind.value} plan",
                [f"rank {r}: plan {d}" for r, d in enumerate(digests)],
            )
        first: dict = {}
        for i in sorted(layers):
            for s in layers[i]:
                first.setdefault(s.name, i)
        self.firsts[kind] = first
        if self.ctx.tp_rank == 0:
            counts: dict = {}
            for spans in layers.values():
                for s in spans:
                    counts[s.name] = counts.get(s.name, 0) + 1
            logger.info(
                "%s %s: %s plan %s: %d layers, spans %s",
                SWITCH,
                self.spec.name,
                kind.value,
                digest,
                len(layers),
                counts,
            )
            for note in notes:
                logger.info(
                    "%s %s: %s plan: %s", SWITCH, self.spec.name, kind.value, note
                )
        return layers

    def _model_reasons(self, limits) -> list:
        s, ctx = self.spec, self.ctx
        if limits is None:
            return [f"no {s.name} kernels for {ctx.isa}"]
        reasons = []
        if ctx.tp_size not in limits.tp_sizes:
            reasons.append(
                f"TP {ctx.tp_size}; the kernels serve TP {list(limits.tp_sizes)}"
            )
        return reasons + list(s.model_disable_reasons(self.owner))

    def attach(self) -> None:
        t0 = time.perf_counter()
        s, ctx = self.spec, self.ctx
        limits = s.devices.get(ctx.isa)
        agree(ctx, s.name, "model checks", lambda: (self._model_reasons(limits), None))
        self.plans[StepKind.DECODE] = self._plan(StepKind.DECODE)
        self.verify_tokens = _verify_tokens()
        self.widths = reachable_widths(limits.max_rows)
        all_widths = sorted(set().union(*self.widths.values()))
        agree(
            ctx,
            s.name,
            "widths",
            lambda: (
                []
                if all_widths
                else [
                    f"no captured step width is within the kernels' {limits.max_rows} rows: every step "
                    "would take the model path"
                ],
                None,
            ),
        )
        self.model = agree(
            ctx,
            s.name,
            "weight views",
            lambda: ([], s.build(ctx, self.owner, self.state)),
        )
        memory = agree(
            ctx, s.name, "memory", lambda: ([], self.model.allocate(all_widths))
        )
        # one agreement after every width: the build loop runs no collective, so a rank that fails one width
        # cannot leave a peer waiting inside the loop
        builds = agree(
            ctx,
            s.name,
            "kernel builds",
            lambda: ([], self.model.build_widths(all_widths)),
        )
        check = (envs.SGLANG_TEST_MONO_CHECK_NUMERICS.get() or "").strip()
        if check and hasattr(self.model, "check_numerics"):
            self._check_numerics(check, all_widths)
        # the kernels hold views and copies of the loaded weights
        self.owner._derived_weight_cache_error = (
            f"Online weight updates are not supported with {SWITCH}: the {s.name} "
            "monokernels hold views and copies of the loaded weights."
        )
        self.state.stage = "attached"
        if ctx.tp_rank == 0:
            copies = self.state.copies
            logger.info(
                "%s %s: weight copies %s (total %.1f MiB)",
                SWITCH,
                s.name,
                {k: f"{v / 2**20:.1f} MiB" for k, v in sorted(copies.items())},
                sum(copies.values()) / 2**20,
            )
            logger.info(
                "%s %s: widths %s; memory %s; builds %s; attached in %.1f s",
                SWITCH,
                s.name,
                {k.value: list(v) for k, v in self.widths.items()},
                memory,
                builds,
                time.perf_counter() - t0,
            )

    def _check_numerics(self, mode: str, widths) -> None:
        """SGLANG_TEST_MONO_CHECK_NUMERICS=log|enforce: the spans against the normal path on the loaded weights at
        every built width, logged per width; enforce refuses startup above the spec's bounds."""
        values = self.model.check_numerics(self.owner, widths)
        bounds = self.spec.numerics.bounds
        over = sorted(
            f"{w} rows: {k} {v:.3g} > {bounds[k]:.3g}"
            for (w, k), v in values.items()
            if bounds.get(k) is not None and v > bounds[k]
        )
        if self.ctx.tp_rank == 0:
            for w in sorted({w for w, _ in values}):
                logger.info(
                    "%s %s: against the normal path at %d rows: %s",
                    SWITCH,
                    self.spec.name,
                    w,
                    {
                        k: float(f"{v:.3g}")
                        for (r, k), v in sorted(values.items())
                        if r == w
                    },
                )
        agree(
            self.ctx,
            self.spec.name,
            "numerics against the normal path",
            lambda: (over if mode == "enforce" else [], None),
        )

    def route(self, forward_batch) -> Optional[Plan]:
        mode = forward_batch.forward_mode
        if mode.is_decode():
            kind = StepKind.DECODE
        elif mode.is_target_verify():
            kind = StepKind.TARGET_VERIFY
        else:
            return None
        rows, bs = int(forward_batch.input_ids.shape[0]), int(forward_batch.batch_size)
        want = bs if kind is StepKind.DECODE else bs * (self.verify_tokens or 0)
        if bs <= 0 or rows != want:
            raise MonokernelError(
                f"{SWITCH}: a {kind.value} step of {rows} rows for {bs} requests (want {want})"
            )
        if rows not in self.widths.get(kind, ()):
            if (kind, rows) not in self._unbuilt and self.ctx.tp_rank == 0:
                logger.info(
                    "%s: %s steps of %d rows take the model path (no width built)",
                    SWITCH,
                    kind.value,
                    rows,
                )
            self._unbuilt.add((kind, rows))
            return None
        layers = self.plans.get(kind)
        if layers is None:
            # the verify plan waits for DSpark's capture layers, set after loading; every rank reaches this
            # point in the same capture warm-up, so its digest is compared there, before any verify step runs
            layers = self.plans[kind] = self._plan(kind)
        if (
            self.ctx.tp_rank == 0
            and torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        ):
            logger.info(
                "%s: captured %s %d rows on monokernels (%d layers)",
                SWITCH,
                kind.value,
                rows,
                len(layers),
            )
        return Plan(kind, rows, layers, self.firsts[kind], self)


def attach(owner) -> None:
    """precompile_kernels_after_loading: refuse on any rank, allocate, build every reachable width."""
    state = _load_state(owner)
    if state is None:
        return
    if state.stage != "loaded":
        raise MonokernelError(
            f"{SWITCH}: attach() before loaded() (stage {state.stage})"
        )
    rt = MonoRuntime(state.spec, _context(state.spec), owner, state)
    rt.attach()
    owner.model._monokernel = rt


def route(model, forward_batch) -> Optional[Plan]:
    """Once a forward: the plan for this step, or None for the normal path."""
    rt = getattr(model, "_monokernel", None)
    return None if rt is None else rt.route(forward_batch)


def kernel(plan: Optional[Plan], layer_id: int, point):
    """The span of layer ``layer_id`` that starts at ``point`` in this step, or None."""
    if plan is None:
        return None
    p = Point.of(point)
    for s in plan.layers.get(layer_id, ()):
        if s.start == p:
            return plan.runtime.model.span(plan, layer_id, s)
    return None


__all__ = [
    "LoadState",
    "MonoContext",
    "MonoRuntime",
    "MonokernelError",
    "Plan",
    "attach",
    "kernel",
    "load_tensor",
    "loaded",
    "route",
]
