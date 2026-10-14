"""The mono runtime a model holds: ``attach`` after weight load, ``begin_step`` once per forward, ``routed`` per layer.

A step not routed to the kernels runs the model's layers exactly as before; the module tree, parameter names and weight
loading are untouched by the switch.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, NamedTuple, Optional

import torch

from sglang.srt.layers.mono import check, failstop, guards, registry, routing
from sglang.srt.layers.mono.runtime.consensus import tp_agree
from sglang.srt.layers.mono.spec import (
    LayerKernel,
    LayerMode,
    MonoContext,
    MonoSpec,
    MonoStep,
)

logger = logging.getLogger(__name__)


class Route(NamedTuple):
    mode: LayerMode
    kernel: LayerKernel
    step: MonoStep
    cache: dict  # per-step memo the spec's glue may fill once and share across the step's layers
    checker: Optional[check.Checker] = (
        None  # set on the steps check mode compares against the module path
    )


class MonoRuntime:
    def __init__(
        self, spec: MonoSpec, ctx: MonoContext, packed: Any, modes: dict[int, LayerMode]
    ) -> None:
        self.spec, self.ctx, self.packed, self.modes = spec, ctx, packed, dict(modes)
        self._kernels: dict[int, LayerKernel] = {}
        self._refused_widths: set[int] = set()
        self.decision = routing.Decision(None, "no step yet")
        self.step_cache: dict = {}
        self.counts: Counter = Counter()
        self.bindings = routing.CacheBinding()
        self.state = dict(spec.state_bindings(ctx, packed))
        self.checker = check.Checker(ctx.tp_rank) if check.requested() else None
        self._checking = False
        logger.info(
            "mono kernels on: spec=%s widths=%s layers=%d (%s)",
            spec.name,
            list(spec.widths),
            len(self.modes),
            dict(Counter(m.value for m in self.modes.values())),
        )

    # ---- per step ----

    def begin_step(self, forward_batch) -> routing.Decision:
        self.step_cache = {}
        d = routing.decide_for_batch(self.spec, self.ctx, forward_batch)
        if d.step is not None and not self._prepare(d.step.rows):
            d = routing.Decision(None, "width refused at build")
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
        mode = self.modes.get(layer_id)
        if mode is None:
            return None
        binding = self.state.get(layer_id)
        if binding is not None:
            binding.wait_ready()
        return Route(
            mode,
            self._kernels[step.rows],
            step,
            self.step_cache,
            self.checker if self._checking else None,
        )

    def health(self) -> Optional[torch.Tensor]:
        return failstop.gather_health(k.health() for k in self._kernels.values())

    # ---- builds ----

    def _prepare(self, rows: int) -> bool:
        """Build (compile) the ``rows`` kernels on their first step, collectively: a width any rank failed to build
        is refused on every rank and its steps take the model's path. Never inside a graph capture: the capture's
        eager warmup forward reaches every width first."""
        if rows in self._kernels:
            return True
        if rows in self._refused_widths:
            return False
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"mono: width {rows} first reached inside a CUDA graph capture"
            )
        kernel = None
        try:
            kernel = self.spec.build(self.ctx, self.packed, rows)
        except Exception:
            logger.exception(
                "mono %s: building the %d-row kernels failed", self.spec.name, rows
            )
        if not tp_agree(kernel is not None, self.ctx.cpu_group):
            self._refused_widths.add(rows)
            logger.warning(
                "mono %s: %d-row steps take the model's path (build refused)",
                self.spec.name,
                rows,
            )
            return False
        self._kernels[rows] = kernel
        return True


def attach(
    model, *, server_args, hf_config, tp_rank: int, tp_size: int, cpu_group, device
) -> Optional[MonoRuntime]:
    """The model's mono runtime, or None with the switch off. Raises ``MonoRefused`` (every reason) when the switch is
    on and the loaded model cannot run the spec."""
    if not guards.mono_enabled(server_args):
        return None
    arch = (hf_config.architectures or [type(model).__name__])[0]
    spec = registry.load_spec(arch) or registry.load_spec(type(model).__name__)
    if spec is None:
        raise guards.MonoRefused(f"--layer-backend mono: no mono spec for {arch}")
    return attach_spec(
        model,
        spec,
        server_args=server_args,
        hf_config=hf_config,
        tp_rank=tp_rank,
        tp_size=tp_size,
        cpu_group=cpu_group,
        device=device,
    )


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
    """``attach`` with the spec named by the caller (a model with a second spec, e.g. its speculative draft)."""
    guards.raise_if_refused("model", spec.name, list(spec.refuse_model(model)))
    ctx = MonoContext(
        server_args=server_args,
        hf_config=hf_config,
        tp_size=tp_size,
        tp_rank=tp_rank,
        cpu_group=cpu_group,
        device=torch.device(device),
        options=dict(spec.options),
    )
    packed = spec.pack(model, ctx)
    # a spec may refine the per-layer modes from the loaded model (``packed.modes``)
    modes = getattr(packed, "modes", None) or spec.layers(hf_config)
    return MonoRuntime(spec, ctx, packed, modes)
