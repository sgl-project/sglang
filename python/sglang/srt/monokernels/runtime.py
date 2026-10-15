# SPDX-License-Identifier: Apache-2.0
"""Weight observation and full-forward dispatch for the selected monokernel backend."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Optional

import msgspec

from sglang.srt.monokernels.guards import deadline_s, device_isa
from sglang.srt.monokernels.spec import (
    BuildContext,
    Feature,
    ShapeKey,
    find_spec,
)

logger = logging.getLogger(__name__)

# Source-contract inventory for test_mono_contract.
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": (
        "monokernel_backend",
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


def _backend() -> Optional[str]:
    """Return the selected backend, or None without published server configuration."""
    from sglang.srt.runtime_context import get_context, get_exec

    # the model loader also runs without a server's config (tools, unit tests)
    if not get_context().is_config_namespace_published("exec"):
        return None
    return get_exec().kernel.monokernel_backend


def _gather_tp(obj) -> list:
    """Gather a value from every TP rank through the CPU group."""
    import torch.distributed as dist

    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    if group.world_size == 1:
        return [obj]
    out: list[Any] = [None] * group.world_size
    dist.all_gather_object(out, obj, group=group.cpu_group)
    return out


def watch_weights(*, model, weights):
    """Let the spec observe checkpoint tensors before model-specific conversion."""
    backend = _backend()
    if backend is None:
        return weights
    spec, _ = find_spec(
        backend=backend, architectures=model.config.architectures, isa=device_isa()
    )
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
    """A routed forward's graph key and state shared by its layers within that step."""

    key: ShapeKey
    state: dict = {}


def attach(*, model, hf_config) -> None:
    """Build the selected spec and install its forward after loading, before KV sizing."""
    backend = _backend()
    if backend is None:
        return
    import torch

    from sglang.srt.runtime_context import get_device

    isa = device_isa()
    spec, why = find_spec(
        backend=backend, architectures=hf_config.architectures, isa=isa
    )
    copies = model.__dict__.pop("_mono_copies", {})
    if spec is None:
        # A draft model may have no registered spec.
        logger.info("monokernel: %s: the normal forward", why)
        return
    reasons = list(spec.model_disable_reasons(model, hf_config, copies))
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if props.multi_processor_count < spec.devices[isa].resident_blocks:
        reasons.append(
            f"{Feature.PARTITIONED_GPU.value}: {props.multi_processor_count} compute "
            f"units, the launch grid needs {spec.devices[isa].resident_blocks}"
        )
    # Collect validation failures before the plugin starts collective setup.
    reasons = [
        f"rank {rank}: {reason}"
        for rank, rank_reasons in enumerate(_gather_tp(reasons))
        for reason in rank_reasons
    ]
    if reasons:
        raise ValueError(f"--monokernel-backend {backend}: " + "; ".join(reasons))

    if spec.tokens_per_request is not None:
        tokens = spec.tokens_per_request(model)
    else:
        tokens = _tokens_per_request()
    keys = _capture_keys(tokens)
    rows = {k.rows for k in keys}
    # Eager execution can use every supported width.
    widths = tuple(
        w for w in spec.widths(isa) if w % tokens == 0 and (not keys or w in rows)
    )
    build_context = BuildContext(
        isa=isa,
        widths=widths,
        deadline_s=deadline_s(get_device().watchdog_timeout),
        copies=copies,
    )
    build_start, memory_before = time.perf_counter(), torch.cuda.memory_allocated()
    built = spec.build(model, hf_config, build_context)
    served = _served_keys(built=built, widths=widths, tokens=tokens)
    # Different routes would leave ranks waiting in incompatible collectives.
    if any(other != served for other in _gather_tp(served)):
        raise RuntimeError("monokernel: the ranks' specs serve different graph keys")
    routed = [k for k in keys if k in served] if keys else served
    if not routed:
        raise ValueError(
            f"--monokernel-backend {backend}: {spec.name} serves none of this "
            "server's steps "
            f"(widths {list(widths)}, {tokens} tokens a request)"
        )
    _install_forward(model=model, built=built, served=served)
    # Captured kernels retain weight copies that online updates cannot refresh.
    model._derived_weight_cache_error = (
        f"Online weight updates are not supported with --monokernel-backend {backend} "
        f"({spec.name})"
    )
    logger.info(
        "monokernel %s on %s runs %s%s; every other step runs the standard path. "
        "Widths %s built in %.1f s, %.1f MiB; wait deadline %.0f s (a failed wait "
        "stops every rank with a GPU fault at address 0x0); weight copies: %s",
        spec.name,
        isa,
        _format_steps(routed),
        f" ({len(routed)} of {len(keys)} decode graphs)" if keys else "",
        list(widths),
        time.perf_counter() - build_start,
        (torch.cuda.memory_allocated() - memory_before) / 2**20,
        build_context.deadline_s,
        _format_weight_copies(copies),
    )


def _served_keys(*, built, widths, tokens: int) -> frozenset[ShapeKey]:
    """Resolve supported decode and verify keys once, before any forwards."""
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


def _install_forward(*, model, built, served: frozenset[ShapeKey]) -> None:
    """Install shape-based dispatch while preserving the original forward signature."""
    original = model.forward

    # Graph runners inspect the original parameters to select forward arguments.
    @functools.wraps(original)
    def forward(*args, **kwargs):
        batch = kwargs["forward_batch"] if "forward_batch" in kwargs else args[2]
        key = shape_key(batch)
        if key not in served:
            return original(*args, **kwargs)
        return built.forward(Plan(key=key), *args, **kwargs)

    model.forward = forward


def _format_steps(keys) -> str:
    """Format supported batch sizes grouped by step mode and tokens per request."""
    sizes: dict = {}
    for k in keys:
        sizes.setdefault((k.forward_mode, k.tokens_per_request), []).append(
            k.batch_size
        )
    return ", ".join(
        f"{mode} steps of batch size {sorted(bs)}" + (f" x {t} tokens" if t > 1 else "")
        for (mode, t), bs in sorted(sizes.items())
    )


def _format_weight_copies(copies: dict) -> str:
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
    """Return configured decode or verify graph keys, or [] when graphs are disabled."""
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
