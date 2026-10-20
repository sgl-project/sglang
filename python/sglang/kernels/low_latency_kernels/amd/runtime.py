# SPDX-License-Identifier: Apache-2.0
"""Weight observation and full-forward dispatch for the selected monokernel backend."""

from __future__ import annotations

import functools
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

import torch
import torch.distributed as dist

from sglang.kernels.low_latency_kernels.amd.guards import deadline_s, device_isa
from sglang.kernels.low_latency_kernels.amd.spec import (
    BACKEND,
    FLAG,
    BuildContext,
    BuiltMonokernels,
    Feature,
    MonoSpec,
    Plan,
    ShapeKey,
    StepMetadata,
    find_spec,
    spec_registered,
)
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import (
    attention_backends,
    get_context,
    get_device,
    get_exec,
    get_parallel,
    get_spec,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

logger = logging.getLogger(__name__)


def _backend() -> Optional[str]:
    """Return the selected backend, or None without published server configuration."""
    # the model loader also runs without a server's config (tools, unit tests)
    if not get_context().is_config_namespace_published("exec"):
        return None
    return BACKEND if get_exec().kernel.enable_amd_monokernel else None


def _gather_tp(obj: Any) -> list[Any]:
    """Gather a value from every TP rank through the CPU group."""
    group = get_parallel().tp_group
    if group.world_size == 1:
        return [obj]
    out: list[Any] = [None] * group.world_size
    dist.all_gather_object(out, obj, group=group.cpu_group)
    return out


def monokernel_weights_iterator(
    *, model: torch.nn.Module, weights: Iterable[tuple[str, torch.Tensor]]
) -> Iterable[tuple[str, torch.Tensor]]:
    """Let the spec observe checkpoint tensors before model-specific conversion."""
    backend = _backend()
    if backend is None:
        return weights
    spec, _ = find_spec(
        backend=backend, architectures=model.config.architectures, isa=device_isa()
    )
    if spec is None or spec.load_tensor is None:
        return weights
    # the copies ride on the model from the loader to install_monokernel_forward
    copies = model.__dict__.setdefault("_mono_copies", {})

    def watched():
        for name, tensor in weights:
            spec.load_tensor(copies, model, name, tensor)
            yield name, tensor

    return watched()


def shape_key(forward_batch: ForwardBatch) -> Optional[ShapeKey]:
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


def is_metadata_unread(
    model: torch.nn.Module, forward_batch: ForwardBatch, metadata: StepMetadata
) -> bool:
    """Whether the monokernel forward that runs this step never reads the metadata."""
    served, unread = model.__dict__.get("_monokernel_unread_metadata", ((), ()))
    return metadata in unread and shape_key(forward_batch) in served


def install_monokernel_forward(
    *,
    model: torch.nn.Module,
    hf_config: PretrainedConfig,
    capture_tail_hooks: list[Callable[..., None]],
) -> None:
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
            None if spec is None else (spec.name, spec.api_version),
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
        raise ValueError(f"{FLAG}: " + "; ".join(reasons))
    if len({identity for *_, identity in lookups}) > 1:
        raise ValueError(
            f"{FLAG}: the ranks load different specs: "
            + "; ".join(
                f"rank {rank}: {name} (API version {api_version})"
                for rank, (*_, (name, api_version)) in enumerate(lookups)
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
        raise ValueError(f"{FLAG}: " + "; ".join(reasons))

    mode, tokens = _server_step()
    if spec.tokens_per_request is not None:
        tokens = spec.tokens_per_request(model)
    capture_keys = _capture_keys(mode=mode, tokens=tokens)
    capture_rows = {k.rows for k in capture_keys}
    # Eager execution can use every supported width.
    widths = tuple(
        w
        for w in spec.widths(isa)
        if w % tokens == 0 and (not capture_keys or w in capture_rows)
    )
    served = _served_keys(mode=mode, widths=widths, tokens=tokens)
    # Different routes would leave ranks waiting in incompatible collectives.
    if any(other != served for other in _gather_tp(served)):
        raise RuntimeError(f"{FLAG}: the ranks route different graph keys")
    if not served:
        raise ValueError(
            f"{FLAG}: {spec.name} has no width for this "
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
    _install_forward(model=model, built=built, served=served, spec=spec)
    # Captured kernels retain weight copies that online updates cannot refresh.
    model._derived_weight_cache_error = (
        f"Online weight updates are not supported with {FLAG} ({spec.name})"
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
        capture_tail_hooks.append(_CaptureCheck(spec=spec, isa=isa, served=served))
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
    runner none of whose graphs run them.
    """

    def __init__(self, *, spec: MonoSpec, isa: str, served: frozenset):
        self.spec, self.isa, self.served = spec, isa, served
        self.captured: dict[int, Optional[ShapeKey]] = {}
        self.is_done = False

    def __call__(
        self,
        runner: DecodeCudaGraphRunner,
        out: Any,
        forward_batch: ForwardBatch,
        num_tokens: int,
    ) -> None:
        if self.is_done or not torch.cuda.is_current_stream_capturing():
            return  # the runner's warmup runs, and any recapture
        self.captured[int(forward_batch.batch_size)] = shape_key(forward_batch)
        if not set(runner.capture_bs) <= set(self.captured):
            return
        self.is_done = True
        routed = [k for k in self.captured.values() if k in self.served]
        if not routed:
            raise ValueError(
                f"{FLAG}: none of the "
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


def _eager_keys(
    *, served: frozenset[ShapeKey], captured: Iterable[Optional[ShapeKey]]
) -> list[ShapeKey]:
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


def _served_keys(
    *, mode: str, widths: tuple[int, ...], tokens: int
) -> frozenset[ShapeKey]:
    """The graph keys of the model's step at each width that holds whole requests."""
    return frozenset(
        ShapeKey(forward_mode=mode, batch_size=w // tokens, tokens_per_request=tokens)
        for w in widths
        if w % tokens == 0
    )


def _install_forward(
    *,
    model: torch.nn.Module,
    built: BuiltMonokernels,
    served: frozenset[ShapeKey],
    spec: MonoSpec,
) -> None:
    """Install shape-based dispatch while preserving the original forward signature."""
    original = model.forward
    allowed = spec.requires.attention_backends
    is_backend_checked = not allowed

    # Graph runners inspect the original parameters to select forward arguments.
    @functools.wraps(original)
    def forward(*args, **kwargs):
        nonlocal is_backend_checked
        # model forwards take (input_ids, positions, forward_batch, ...)
        batch = kwargs["forward_batch"] if "forward_batch" in kwargs else args[2]
        key = shape_key(batch)
        if key not in served:
            return original(*args, **kwargs)
        if not is_backend_checked:
            # a draft runner may override the backend the launcher checked: read the
            # one this step runs on, as model dispatch does, before the first forward
            _, configured = attention_backends()
            attention = get_attn_backend().decode_attention_backend_str or configured
            if attention not in allowed:
                raise ValueError(
                    f"{FLAG}: {spec.name} needs a decode "
                    f"attention backend in {sorted(allowed)}, this runner's is "
                    f"{attention!r}"
                )
            is_backend_checked = True
        return built.forward(Plan(key=key), *args, **kwargs)

    model.forward = forward
    # attention backends ask is_metadata_unread before building per-step metadata
    model._monokernel_unread_metadata = (served, spec.unread_metadata)


def _format_steps(keys: Iterable[ShapeKey]) -> str:
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


def _format_weight_copies(copies: dict[tuple, torch.Tensor]) -> str:
    totals: dict = {}  # kind -> (tensors, bytes)
    for (kind, *_), tensor in copies.items():
        count, nbytes = totals.get(kind, (0, 0))
        totals[kind] = (count + 1, nbytes + tensor.numel() * tensor.element_size())
    return ", ".join(
        f"{kind} ({count} tensor{'' if count == 1 else 's'}, {nbytes / 2**20:.1f} MiB)"
        for kind, (count, nbytes) in sorted(totals.items())
    )


def _server_step() -> tuple[str, int]:
    """The mode and tokens per request of this server's decode-phase steps."""
    spec_config = get_spec()
    if spec_config.speculative_algorithm is None:
        return "decode", 1
    # the target's and its draft's steps are verify steps at any token count
    return "target_verify", int(spec_config.speculative_num_draft_tokens or 1)


def _capture_keys(*, mode: str, tokens: int) -> list[ShapeKey]:
    """Return configured decode or verify graph keys, or [] when graphs are disabled."""
    graph = get_exec().graph.cuda_graph_config
    if graph is None or graph.decode.backend == Backend.DISABLED:
        return []
    sizes = sorted(set(graph.decode.bs or []))
    return [
        ShapeKey(forward_mode=mode, batch_size=bs, tokens_per_request=tokens)
        for bs in sizes
    ]
