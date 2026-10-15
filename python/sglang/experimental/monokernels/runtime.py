# SPDX-License-Identifier: Apache-2.0
"""The monokernel plugin in each process (RFC #43949), after the config checks in
``guards``. Each call is a no-op while ``--enable-monokernel`` is off.

- ``load_tensor(model, name, tensor)``: each checkpoint tensor, for the copies of
  weights that loading rewrites.
- ``attach(model, hf_config)``: after weight load, before KV sizing. Every rank's
  reasons are gathered and refused together; then build every planned width and
  compare every rank's routing plans.
- ``route(model, forward_batch, **state)``: one plan a forward, from the graph key.
- ``kernel(plan, layer_id, point)``: the span that starts at ``point``, or None.

Routing runs none of the spec's code: ``attach`` fixes the spans of every graph
key a step can have, up to the largest width, and every rank compares them. A
forward looks its key up: any other key (prefill, extend, mixed and idle steps,
uneven rows, a width not built) takes the normal path, for every layer. A graph
replay repeats its capture's decision; each decision is logged once, on rank 0.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

from sglang.experimental.monokernels.guards import deadline_s, device_isa
from sglang.experimental.monokernels.spec import (
    BuildContext,
    Feature,
    ShapeKey,
    SpanPoint,
    find_spec,
)

logger = logging.getLogger(__name__)

# The SGLang names this module reads (as in ``guards.SGLANG_READS``)
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": (
        "enable_monokernel",
        "disable_cuda_graph",
        "disable_decode_cuda_graph",
        "cuda_graph_config",
        "watchdog_timeout",
        "speculative_algorithm",
        "speculative_num_draft_tokens",
    ),
    "srt/distributed/parallel_state.py": ("world_size", "cpu_group", "rank_in_group"),
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


def _str(key: ShapeKey) -> str:
    return f"{key.forward_mode} bs {key.batch_size} x {key.tokens_per_request}"


def _why_normal(forward_batch, key: Optional[ShapeKey], widths) -> str:
    """Why a forward takes the normal path, coarse enough to log once each."""
    if key is not None:
        top = max(widths, default=0)
        if key.rows > top:
            return f"{key.forward_mode} over {top} rows"
        if key.rows not in widths:
            return f"{key.forward_mode} of {key.rows} rows, a width not built"
        return f"{key.forward_mode} of {key.rows} rows, which the spec leaves normal"
    mode = forward_batch.forward_mode
    if mode.is_decode() or mode.is_target_verify():
        return f"{mode.name.lower()} with uneven rows a request"
    return f"{mode.name.lower()} steps"


@dataclass(frozen=True)
class Plan:
    """One forward's routing decision; ``state`` holds the forward's values for the
    spec's kernels (``route``'s keywords, and what a graph capture records)."""

    key: ShapeKey
    built: Any  # the spec's built model
    spans: dict = field(compare=False)  # (layer_id, point name) -> span
    state: dict = field(default_factory=dict, compare=False)


def _keys(widths, tokens: int):
    """Every graph key with ``widths`` rows: decode, and verify at ``tokens``."""
    for w in widths:
        yield ShapeKey("decode", w, 1, w)
        if tokens > 1 and w % tokens == 0:
            yield ShapeKey("target_verify", w // tokens, tokens, w)


class MonoRuntime:
    """A model's built monokernels, and the spans of every graph key a step can
    have, fixed here: routing a forward runs none of the spec's code."""

    def __init__(self, built, layer_ids, widths, tokens: int, rank: int = 0):
        self.built, self.widths, self.rank = built, tuple(widths), rank
        self.table: dict = {}  # ShapeKey -> {(layer_id, point name): span}
        for key in _keys(self.widths, tokens):
            spans = {}
            for i in layer_ids:
                for p in SpanPoint:
                    fn = built.kernel(i, p, key)
                    if fn is not None:
                        spans[(i, p.name.lower())] = fn
            if spans:
                self.table[key] = spans
        self.counts: Counter = Counter()  # eager forwards: "mono", or the reason
        self._logged: set = set()

    def decision(self, key: ShapeKey) -> str:
        """The routing decision for ``key`` as text: what every rank must agree on."""
        spans = " ".join(f"{i}.{p}" for i, p in sorted(self.table.get(key, ())))
        return f"{key}: {spans or 'normal'}"

    def note(self, key: Optional[ShapeKey], why: str) -> None:
        """Count an eager forward's path, and log each decision once on rank 0: a
        graph's at its capture, an eager forward's once a reason."""
        import torch

        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if not capturing:
            self.counts[why or "mono"] += 1
        entry = ("graph", key, why) if capturing else ("eager", why)
        if self.rank == 0 and entry not in self._logged:
            self._logged.add(entry)
            if not capturing:
                where = "eager forwards" + ("" if why else " of a built width")
            else:
                where = "graph" + (f" {_str(key)}" if key is not None else "")
            if why:
                logger.info("monokernel: %s take the normal path: %s", where, why)
            else:
                layers = len({i for i, _ in self.table[key]})
                logger.info(
                    "monokernel: %s run %d layers on monokernels", where, layers
                )


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
        spec.build(model, hf_config, ctx),
        sorted(spec.spans(hf_config)),
        widths,
        tokens,
        _rank(),
    )
    table = "\n".join(rt.decision(k) for k in sorted(rt.table, key=str))
    if len(set(_gather(hashlib.sha256(table.encode()).hexdigest()))) > 1:
        raise RuntimeError("monokernel routing plans differ across ranks")
    lines = [rt.decision(k) for k in keys]
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


def _rank() -> int:
    from sglang.srt.runtime_context import get_parallel

    return int(get_parallel().tp_group.rank_in_group)


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
    """One plan a forward from the graph key alone; None: the normal path, for
    every layer. ``state``: the forward's values the spec's kernels read."""
    rt = vars(model).get("mono")  # no nn.Module attribute miss on the normal path
    if rt is None:
        return None
    key = shape_key(forward_batch)
    spans = rt.table.get(key) if key is not None else None
    rt.note(key, "" if spans else _why_normal(forward_batch, key, rt.widths))
    return Plan(key, rt.built, spans, state) if spans else None


def kernel(plan: Optional[Plan], layer_id: int, point):
    """The span of ``layer_id`` that starts at ``point`` (a ``SpanPoint`` or its
    lower-case name) under ``plan``, bound to the plan; None: the normal path."""
    if plan is None:
        return None
    name = point if isinstance(point, str) else point.name.lower()
    fn = plan.spans.get((layer_id, name))
    return None if fn is None else partial(fn, plan)
