# SPDX-License-Identifier: Apache-2.0
"""Weight observation and full-forward dispatch for the selected monokernel backend."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Optional

import torch
import torch.distributed as dist

from sglang.srt.monokernels.guards import deadline_s, device_isa
from sglang.srt.monokernels.spec import (
    BuildContext,
    BuiltMonokernels,
    Feature,
    MonoSpec,
    Plan,
    ShapeKey,
    find_spec,
    spec_registered,
)
from sglang.srt.runtime_context import (
    get_context,
    get_device,
    get_exec,
    get_parallel,
    get_spec,
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
    "srt/model_executor/model_runner.py": (
        "capture_tail_hooks",
        "decode_attention_backend_str",
    ),
    "srt/model_executor/model_runner_components/weight_updater.py": (
        '"_derived_weight_cache_error"',
    ),
    "srt/model_executor/runner/decode_cuda_graph_runner.py": (
        "capture_bs",
        "model_runner",
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
    # the model loader also runs without a server's config (tools, unit tests)
    if not get_context().is_config_namespace_published("exec"):
        return None
    return get_exec().kernel.monokernel_backend


def _gather_tp(obj) -> list:
    """Gather a value from every TP rank through the CPU group."""
    group = get_parallel().tp_group
    if group.world_size == 1:
        return [obj]
    out: list[Any] = [None] * group.world_size
    dist.all_gather_object(out, obj, group=group.cpu_group)
    return out


def monokernel_weights_iterator(*, model, weights):
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


def install_monokernel_forward(*, model, hf_config, capture_tail_hooks: list) -> None:
    """Build the selected spec and install its forward; runs before KV cache sizing."""
    backend = _backend()
    if backend is None:
        return
    isa = device_isa()
    archs = hf_config.architectures
    spec, why = find_spec(backend=backend, architectures=archs, isa=isa)
    is_load_observed = "_mono_copies" in model.__dict__
    copies = model.__dict__.pop("_mono_copies", {})
    # Ranks branch together; one skipping the collectives below would hang its peers.
    lookups = _gather_tp(
        (
            spec is not None,
            spec_registered(backend=backend, architectures=archs),
            why,
            None if spec is None else (spec.name, spec.api),
        )
    )
    if not any(is_registered for _, is_registered, _, _ in lookups):
        # A draft model may have no registered spec.
        logger.info("monokernel: %s: the normal forward", why)
        return
    reasons = [
        f"rank {rank}: {reason}"
        for rank, (is_found, _, reason, _) in enumerate(lookups)
        if not is_found
    ]
    if reasons:
        raise ValueError(f"--monokernel-backend {backend}: " + "; ".join(reasons))
    if len({identity for *_, identity in lookups}) > 1:
        raise ValueError(
            f"--monokernel-backend {backend}: the ranks load different specs: "
            + "; ".join(
                f"rank {rank}: {name} (API {api})"
                for rank, (*_, (name, api)) in enumerate(lookups)
            )
        )

    reasons = list(spec.model_disable_reasons(model, hf_config, copies))
    if spec.load_tensor is not None and not is_load_observed:
        # e.g. the layered, dummy, sharded-state and remote loaders
        reasons.append(
            f"{spec.name} keeps weight copies at load, and this load path does not "
            "pass checkpoint tensors through monokernel_weights_iterator"
        )
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
    capture_keys = _capture_keys(tokens)
    capture_rows = {k.rows for k in capture_keys}
    # Eager execution can use every supported width.
    widths = tuple(
        w
        for w in spec.widths(isa)
        if w % tokens == 0 and (not capture_keys or w in capture_rows)
    )
    served = _served_keys(widths=widths, tokens=tokens)
    # Different routes would leave ranks waiting in incompatible collectives.
    if any(other != served for other in _gather_tp(served)):
        raise RuntimeError(
            f"--monokernel-backend {backend}: the ranks route different graph keys"
        )
    if not served:
        raise ValueError(
            f"--monokernel-backend {backend}: {spec.name} has no width for this "
            f"server's steps ({tokens} tokens a request)"
        )
    build_context = BuildContext(
        isa=isa,
        widths=widths,
        deadline_s=deadline_s(get_device().watchdog_timeout),
        copies=copies,
    )
    build_start, memory_before = time.perf_counter(), torch.cuda.memory_allocated()
    built = spec.build(model, hf_config, build_context)
    _install_forward(model=model, built=built, served=served)
    # Captured kernels retain weight copies that online updates cannot refresh.
    model._derived_weight_cache_error = (
        f"Online weight updates are not supported with --monokernel-backend {backend} "
        f"({spec.name})"
    )
    logger.info(
        "monokernel %s on %s: widths %s built in %.1f s, %.1f MiB; wait deadline "
        "%.0f s (a failed wait stops every rank with a GPU fault at address 0x0); "
        "weight copies: %s",
        spec.name,
        isa,
        list(widths),
        time.perf_counter() - build_start,
        (torch.cuda.memory_allocated() - memory_before) / 2**20,
        build_context.deadline_s,
        _format_weight_copies(copies),
    )
    if capture_keys:
        capture_tail_hooks.append(
            _CaptureCheck(spec=spec, backend=backend, isa=isa, served=served)
        )
    else:
        # Without CUDA graphs every step runs eagerly, routed by its own key.
        logger.info(
            "monokernel %s on %s runs %s; every other step runs the standard path",
            spec.name,
            isa,
            _format_steps(served),
        )


class _CaptureCheck:
    """Check the captured decode graphs, not the configured sizes, once all are in.

    Logs the steps that run the monokernels, captured and eager, and refuses a
    runner none of whose graphs run them or whose decode attention backend the
    spec does not accept.
    """

    def __init__(self, *, spec: MonoSpec, backend: str, isa: str, served: frozenset):
        self.spec, self.backend, self.isa, self.served = spec, backend, isa, served
        self.captured: dict[int, Optional[ShapeKey]] = {}
        self.is_done = False

    def __call__(self, runner, out, forward_batch, num_tokens) -> None:
        if self.is_done or not torch.cuda.is_current_stream_capturing():
            return  # the runner's warmup runs, and any recapture
        self.captured[int(forward_batch.batch_size)] = shape_key(forward_batch)
        if not set(runner.capture_bs) <= set(self.captured):
            return
        self.is_done = True
        attention = runner.model_runner.decode_attention_backend_str
        allowed = self.spec.requires.attention_backends
        if allowed and attention not in allowed:
            raise ValueError(
                f"--monokernel-backend {self.backend}: {self.spec.name} needs a decode "
                f"attention backend in {sorted(allowed)}, this runner's is "
                f"{attention!r}"
            )
        routed = [k for k in self.captured.values() if k in self.served]
        if not routed:
            raise ValueError(
                f"--monokernel-backend {self.backend}: none of the "
                f"{len(self.captured)} captured decode graphs (batch sizes "
                f"{sorted(self.captured)}) runs {self.spec.name}'s monokernels"
            )
        eager = _eager_keys(served=self.served, captured=self.captured.values())
        logger.info(
            "monokernel %s on %s runs %s (%d of %d captured decode graphs)%s; every "
            "other step runs the standard path",
            self.spec.name,
            self.isa,
            _format_steps(routed),
            len(routed),
            len(self.captured),
            f", and eagerly {_format_steps(eager)}" if eager else "",
        )


def _eager_keys(*, served, captured) -> list[ShapeKey]:
    """Return served keys larger than every captured graph of their mode."""
    # A step pads up to the smallest captured graph of its mode that holds it.
    largest: dict = {}
    for k in captured:
        if k is not None:
            mode = (k.forward_mode, k.tokens_per_request)
            largest[mode] = max(largest.get(mode, 0), k.batch_size)
    return [
        k
        for k in served
        if k.batch_size > largest.get((k.forward_mode, k.tokens_per_request), 0)
    ]


def _served_keys(*, widths, tokens: int) -> frozenset[ShapeKey]:
    """The graph keys of the model's step at each width that holds whole requests."""
    # A speculative target runs verify steps only; a plain server runs decode.
    mode = "decode" if tokens == 1 else "target_verify"
    return frozenset(
        ShapeKey(forward_mode=mode, batch_size=w // tokens, tokens_per_request=tokens)
        for w in widths
        if w % tokens == 0
    )


def _install_forward(
    *, model, built: BuiltMonokernels, served: frozenset[ShapeKey]
) -> None:
    """Install shape-based dispatch while preserving the original forward signature."""
    original = model.forward

    # Graph runners inspect the original parameters to select forward arguments.
    @functools.wraps(original)
    def forward(*args, **kwargs):
        # model forwards take (input_ids, positions, forward_batch, ...)
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
    totals: dict = {}  # kind -> (tensors, bytes)
    for (kind, *_), tensor in copies.items():
        count, nbytes = totals.get(kind, (0, 0))
        totals[kind] = (count + 1, nbytes + tensor.numel() * tensor.element_size())
    return ", ".join(
        f"{kind} ({count} tensor{'' if count == 1 else 's'}, {nbytes / 2**20:.1f} MiB)"
        for kind, (count, nbytes) in sorted(totals.items())
    )


def _tokens_per_request() -> int:
    spec_cfg = get_spec()
    if spec_cfg.speculative_algorithm is None:
        return 1
    return int(spec_cfg.speculative_num_draft_tokens or 1)


def _capture_keys(tokens: int) -> list[ShapeKey]:
    """Return configured decode or verify graph keys, or [] when graphs are disabled."""
    graph = get_exec().graph
    if graph.disable_cuda_graph or graph.disable_decode_cuda_graph:
        return []
    mode = "decode" if tokens == 1 else "target_verify"
    sizes = sorted(set(graph.cuda_graph_config.decode.bs or []))
    return [
        ShapeKey(forward_mode=mode, batch_size=bs, tokens_per_request=tokens)
        for bs in sizes
    ]
