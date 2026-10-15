# SPDX-License-Identifier: Apache-2.0
"""The monokernel framework in each process, after the config checks in ``guards``.
SGLang's runtime calls it for every model; each call is a no-op while
``--enable-monokernel`` is off, and for a model no spec covers.

- ``watch_weights``: the model loader's checkpoint tensors, on their way to
  ``model.load_weights``; the spec may keep a copy of a weight that loading rewrites.
- ``attach``: after weight load, before KV sizing. Every rank's refusal reasons are
  listed together; then it builds every planned width, checks that every rank
  serves the same graph keys, and switches the model's forward.

The switch is SGLang's own pattern for execution backends (the torch.compile
trampoline): ``model.forward`` on this instance routes each call by its graph key
alone, against the keys fixed at attach. A step the spec serves runs the spec's
forward, which takes and returns what the model's forward does; any other step
(prefill, extend, mixed and idle steps, uneven rows, a width not built) runs the
model's forward as it is. A CUDA graph replays its capture's decision, so the
startup log line that lists the served steps says which steps run the monokernels.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Optional

import msgspec

from sglang.experimental.monokernels.guards import deadline_s, device_isa
from sglang.experimental.monokernels.spec import (
    BuildContext,
    Feature,
    ShapeKey,
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
    "srt/distributed/parallel_state.py": ("world_size", "cpu_group"),
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
        "get_context",
        "get_device",
        "get_exec",
        "get_parallel",
        "get_spec",
        "is_config_namespace_published",
        '"tp_group"',
    ),
}


def enabled() -> bool:
    from sglang.srt.runtime_context import get_context, get_exec

    # the model loader also runs without a server's config (tools, unit tests)
    if not get_context().is_config_namespace_published("exec"):
        return False
    return get_exec().kernel.enable_monokernel


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


def watch_weights(*, model, weights):
    """``weights`` as they are, after the spec saw each (name, tensor): the
    checkpoint's names and tensors, before the model renames or converts them."""
    if not enabled():
        return weights
    spec, _ = find_spec(architectures=model.config.architectures, isa=device_isa())
    if spec is None or spec.load_tensor is None:
        return weights
    copies = model.__dict__.setdefault("_mono_copies", {})

    def watched():
        for name, tensor in weights:
            spec.load_tensor(copies, model, name, tensor)
            yield name, tensor

    return watched()


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
    return ShapeKey(forward_mode=name, batch_size=bs, tokens_per_request=rows // bs)


class Plan(msgspec.Struct, frozen=True):
    """One routed forward: its graph key, and ``state``, where the spec's forward
    keeps the values its layers share within the step."""

    key: ShapeKey
    state: dict = {}


def attach(*, model, hf_config) -> None:
    """After weight load, before KV sizing: refuse with every rank's reasons, or
    build every planned width and switch the model's forward."""
    if not enabled():
        return
    import torch

    from sglang.srt.runtime_context import get_device

    isa = device_isa()
    spec, why = find_spec(architectures=hf_config.architectures, isa=isa)
    copies = model.__dict__.pop("_mono_copies", {})
    if spec is None:
        # the config hook refused a served model without a spec; this one (e.g. a
        # draft model) keeps its forward
        logger.info("monokernel: %s: the normal forward", why)
        return
    reasons = list(spec.model_disable_reasons(model, hf_config, copies))
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

    if spec.tokens_per_request is not None:
        tokens = spec.tokens_per_request(model)
    else:
        tokens = _tokens_per_request()
    keys = _capture_keys(tokens)
    rows = {k.rows for k in keys}
    # the widths the capture plan routes (every width a step can have without graphs)
    widths = tuple(
        w for w in spec.widths(isa) if w % tokens == 0 and (not keys or w in rows)
    )
    ctx = BuildContext(
        isa=isa,
        widths=widths,
        deadline_s=deadline_s(get_device().watchdog_timeout),
        copies=copies,
    )
    t0, mem0 = time.perf_counter(), torch.cuda.memory_allocated()
    built = spec.build(model, hf_config, ctx)
    served = _served(built=built, widths=widths, tokens=tokens)
    # a rank that routed a step differently would wait on its peers until the deadline
    if any(other != served for other in _gather(served)):
        raise RuntimeError("monokernel: the ranks' specs serve different graph keys")
    routed = [k for k in keys if k in served] if keys else served
    if not routed:
        raise ValueError(
            f"--enable-monokernel: {spec.name} serves none of this server's steps "
            f"(widths {list(widths)}, {tokens} tokens a request)"
        )
    _switch(model=model, built=built, served=served)
    # SGLang's weight updater refuses each update: the kernels read copies and views
    model._derived_weight_cache_error = (
        "Online weight updates are not supported with --enable-monokernel "
        f"({spec.name})"
    )
    logger.info(
        "monokernel %s on %s runs %s%s; every other step runs the standard path. "
        "Widths %s built in %.1f s, %.1f MiB; wait deadline %.0f s (a failed wait "
        "stops every rank with a GPU fault at address 0x0); weight copies: %s",
        spec.name,
        isa,
        _steps(routed),
        f" ({len(routed)} of {len(keys)} decode graphs)" if keys else "",
        list(widths),
        time.perf_counter() - t0,
        (torch.cuda.memory_allocated() - mem0) / 2**20,
        ctx.deadline_s,
        _sizes(copies),
    )


def _served(*, built, widths, tokens: int) -> frozenset[ShapeKey]:
    """The graph keys of ``widths`` rows that ``built`` serves: decode, and verify at
    ``tokens`` a request. Fixed here: routing runs none of the spec's code."""
    keys = [
        ShapeKey(forward_mode="decode", batch_size=w, tokens_per_request=1)
        for w in widths
    ] + [
        ShapeKey(
            forward_mode="target_verify",
            batch_size=w // tokens,
            tokens_per_request=tokens,
        )
        for w in widths
        if tokens > 1 and w % tokens == 0
    ]
    return frozenset(k for k in keys if built.serves(k))


def _switch(*, model, built, served: frozenset[ShapeKey]) -> None:
    """Route each call of ``model.forward`` on this instance by its graph key, as
    SGLang's torch.compile trampoline routes by phase. The runners call
    ``forward(input_ids, positions, forward_batch, ...)``, and a CUDA graph capture
    records the routed call."""
    original = model.forward

    # the graph runners read the forward's parameters: they see the model's own
    @functools.wraps(original)
    def forward(*args, **kwargs):
        batch = kwargs["forward_batch"] if "forward_batch" in kwargs else args[2]
        key = shape_key(batch)
        if key not in served:
            return original(*args, **kwargs)
        return built.forward(Plan(key=key), *args, **kwargs)

    model.forward = forward


def _steps(keys) -> str:
    """The steps ``keys`` cover, e.g. "decode steps of batch size [1, 2, 4]"."""
    sizes: dict = {}
    for k in keys:
        sizes.setdefault((k.forward_mode, k.tokens_per_request), []).append(
            k.batch_size
        )
    return ", ".join(
        f"{mode} steps of batch size {sorted(bs)}" + (f" x {t} tokens" if t > 1 else "")
        for (mode, t), bs in sorted(sizes.items())
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


def _capture_keys(tokens: int) -> list[ShapeKey]:
    """The decode graph keys at ``tokens`` a request: DECODE, or TARGET_VERIFY (a
    speculative verify step, or a draft's block)."""
    from sglang.srt.runtime_context import get_exec

    graph = get_exec().graph
    if graph.disable_cuda_graph or graph.disable_decode_cuda_graph:
        return []
    mode = "decode" if tokens == 1 else "target_verify"
    sizes = sorted(set(graph.cuda_graph_config.decode.bs or []))
    return [
        ShapeKey(forward_mode=mode, batch_size=bs, tokens_per_request=tokens)
        for bs in sizes
    ]
