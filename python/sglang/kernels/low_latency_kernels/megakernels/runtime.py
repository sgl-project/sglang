# SPDX-License-Identifier: Apache-2.0
"""Build and whole-step dispatch for --enable-nvidia-megakernel."""

from __future__ import annotations

import functools
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import msgspec
import torch
import torch.distributed as dist

from sglang.kernels.low_latency_kernels.megakernels.binding import MegakernelBinding
from sglang.kernels.low_latency_kernels.megakernels.guards import deadline_s, device_isa
from sglang.kernels.low_latency_kernels.megakernels.spec import (
    FLAG,
    BuildContext,
    BuiltMegakernel,
    Feature,
    MegakernelSpec,
    Step,
    find_spec,
    spec_registered,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import (
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


def megakernel_enabled() -> bool:
    """Whether this server runs with --enable-nvidia-megakernel."""
    # the model loader also runs without a server's config (tools, unit tests)
    if not get_context().is_config_namespace_published("exec"):
        return False
    return get_exec().kernel.enable_nvidia_megakernel


def install_megakernel(
    *,
    model: torch.nn.Module,
    hf_config: PretrainedConfig,
    capture_tail_hooks: list[Callable[..., None]],
    is_draft: bool,
) -> None:
    """Build the model's spec and install its forward; runs before KV cache sizing."""
    if not megakernel_enabled():
        return
    spec = _find_spec_on_every_rank(hf_config=hf_config, is_draft=is_draft)
    if spec is None:
        return
    # Resolution checked the target's spec on the server arguments; a draft's spec
    # is checked only here, on what the runner reads (no server-argument checks).
    reasons = _requirement_reasons(spec=spec, hf_config=hf_config)
    reasons += _model_disable_reasons(spec=spec, model=model, hf_config=hf_config)
    _refuse(_gather_reasons(reasons))

    tokens = _tokens_per_request()
    binding = MegakernelBinding(model=model)
    context = BuildContext(
        widths=tuple(range(tokens, spec.max_rows + 1, tokens)),
        tokens_per_request=tokens,
        deadline_s=deadline_s(get_device().watchdog_timeout),
        binding=binding,
    )
    start, free_before = time.perf_counter(), torch.cuda.mem_get_info()[0]
    built = spec.build(model, hf_config, context)
    # not setattr: a built object that is an nn.Module must not join the state dict
    model.__dict__["_megakernel"] = built
    # Different widths would leave ranks waiting in different collectives.
    megakernel_agree(tuple(built.widths), what=f"widths {spec.name} built")
    if spec.max_rows > 0 and not built.widths:
        _refuse([f"{spec.name} built no width of {list(context.widths)}"])
    # Captured kernels keep state derived from the weights that updates cannot refresh.
    model._derived_weight_cache_error = (
        f"Online weight updates are not supported with {FLAG} ({spec.name})"
    )
    built_s = time.perf_counter() - start
    used_mib = (free_before - torch.cuda.mem_get_info()[0]) / 2**20
    if not built.widths:
        # a spec that covers no step, e.g. a draft spec offering draft_sampler only
        logger.info(
            "NVIDIA megakernel: %s: %s",
            spec.name,
            "draft sampler"
            if built.draft_sampler is not None
            else "no step and no draft sampler",
        )
        return
    logger.info(
        "NVIDIA megakernel: active (%s: %s; widths %s; %s numerics; built in %.1f s, "
        "%.1f MiB; wait deadline %.0f s)",
        spec.name,
        built.summary(),
        list(built.widths),
        spec.numerics.kind.value,
        built_s,
        used_mib,
        context.deadline_s,
    )
    dispatch = _Dispatch(built=built, original=model.forward, binding=binding)
    model.forward = dispatch.wrapped()
    if _decode_graphs_enabled():
        capture_tail_hooks.append(_CaptureCheck(spec=spec, dispatch=dispatch))


def built_megakernel(model: torch.nn.Module) -> Optional[BuiltMegakernel]:
    """The object the spec built for this model, or None."""
    return model.__dict__.get("_megakernel")


def megakernel_agree(value: Any, *, what: str, group: Any = None) -> None:
    """Raise on every rank unless every rank of group (default: the TP group) holds
    the same value; ranks that differ would wait in different collectives."""
    values = _gather_tp(value, group=group)
    if any(v != value for v in values):
        raise RuntimeError(
            f"{FLAG}: the ranks' {what} differ: "
            + "; ".join(f"rank {rank}: {v}" for rank, v in enumerate(values))
        )


def megakernel_draft_sampler(draft_model: torch.nn.Module) -> Optional[Callable]:
    """The built draft spec's draft_sampler factory, or None for SGLang's proposal."""
    built = built_megakernel(draft_model)
    return None if built is None else built.draft_sampler


def _find_spec_on_every_rank(
    *, hf_config: PretrainedConfig, is_draft: bool
) -> Optional[MegakernelSpec]:
    """The spec every rank found; for a draft model, None when no rank has one
    registered; else raise on every rank."""
    archs = hf_config.architectures
    spec, why = find_spec(architectures=archs, isa=device_isa())
    # Ranks branch together; one skipping the collectives below would hang its peers.
    lookups = _gather_tp(
        (
            spec_registered(archs),
            why,
            None if spec is None else (spec.name, spec.api_version),
        )
    )
    if not any(is_registered for is_registered, _, _ in lookups):
        if is_draft:
            # A draft model may have no registered spec.
            logger.info("NVIDIA megakernel: %s; %s keeps its own forward", why, archs)
            return None
        # resolution found the target's spec, so the plugin failed to load here
        _refuse(
            [
                f"{why} in the model workers, though argument resolution found it; "
                "see 'Failed to load plugin' or 'Failed to execute general plugin' "
                "above"
            ]
        )
    _refuse(
        [
            f"rank {rank}: {why}"
            for rank, (_, why, identity) in enumerate(lookups)
            if identity is None
        ]
    )
    if len({identity for *_, identity in lookups}) > 1:
        raise ValueError(
            f"{FLAG}: the ranks load different specs: "
            + "; ".join(f"rank {r}: {ident}" for r, (*_, ident) in enumerate(lookups))
        )
    return spec


def _requirement_reasons(
    *, spec: MegakernelSpec, hf_config: PretrainedConfig
) -> list[str]:
    """The spec's requirements this runner can check without the server arguments."""
    req, tp_size = spec.requires, get_parallel().tp_size
    reasons = []
    # an empty set accepts any TP size, as at argument resolution
    if req.tp_sizes and tp_size not in req.tp_sizes:
        reasons.append(f"tp_size {tp_size} is not one of {sorted(req.tp_sizes)}")
    graph = get_exec().graph.cuda_graph_config
    backend = Backend.DISABLED if graph is None else graph.decode.backend
    if req.decode_graph_backends and backend not in req.decode_graph_backends:
        reasons.append(
            f"decode CUDA graphs {backend!r} are not one of "
            f"{sorted(req.decode_graph_backends)}"
        )
    exactly = [
        (f"config.{key}", getattr(hf_config, key, None), want)
        for key, want in req.hf_config.items()
    ] + [(name, getattr(envs, name).get(), want) for name, want in req.env.items()]
    reasons += [
        f"{what} {v!r} is not {want!r}" for what, v, want in exactly if v != want
    ]
    return reasons


def _model_disable_reasons(
    *, spec: MegakernelSpec, model: torch.nn.Module, hf_config: PretrainedConfig
) -> list[str]:
    """This rank's reasons the loaded model cannot run the spec."""
    reasons = []
    if spec.model_disable_reasons is not None:
        reasons += spec.model_disable_reasons(model, hf_config)
    if "forward" in model.__dict__:
        reasons.append("the model's forward is already replaced on this instance")
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    if sms < spec.requires.resident_blocks:
        reasons.append(
            f"{Feature.PARTITIONED_GPU.value}: {sms} SMs, the persistent grids need "
            f"{spec.requires.resident_blocks}"
        )
    return reasons


class _Dispatch:
    """The installed forward: covered steps run the built megakernel."""

    def __init__(
        self, *, built: BuiltMegakernel, original: Callable, binding: MegakernelBinding
    ):
        self.built, self.original, self.binding = built, original, binding
        # batch size -> the step planned while capturing its decode graphs; plan reads
        # only the mode and rows, so every graph variant of a batch size plans alike
        self.captured: dict[int, Optional[Step]] = {}

    def wrapped(self) -> Callable:
        # Graph runners and supports_pp inspect the original forward's signature.
        @functools.wraps(self.original)
        def forward(input_ids, positions, forward_batch, *args, **kwargs):
            step = self.built.plan(input_ids, forward_batch, *args, **kwargs)
            if torch.cuda.is_current_stream_capturing():
                self.captured[int(forward_batch.batch_size)] = step
            if step is None:
                return self.original(
                    input_ids, positions, forward_batch, *args, **kwargs
                )
            return self.built.forward(
                msgspec.structs.replace(step, view=self.binding.forward_view()),
                self.original,
                input_ids,
                positions,
                forward_batch,
                *args,
                **kwargs,
            )

        return forward


class _CaptureCheck:
    """Once every decode graph is captured, log the ones that run the megakernel and
    refuse a runner none of whose graphs do."""

    def __init__(self, *, spec: MegakernelSpec, dispatch: _Dispatch):
        self.spec, self.dispatch = spec, dispatch
        self.is_done = False

    def __call__(
        self,
        runner: DecodeCudaGraphRunner,
        out: Any,
        forward_batch: ForwardBatch,
        num_tokens: int,
    ) -> None:
        captured = self.dispatch.captured
        if self.is_done or not set(runner.capture_bs) <= set(captured):
            return
        self.is_done = True
        self.dispatch.binding.assert_live()
        routed = sorted(bs for bs, step in captured.items() if step is not None)
        if not routed:
            raise ValueError(
                f"{FLAG}: none of the {len(captured)} captured decode graphs "
                f"(batch sizes {sorted(captured)}) runs {self.spec.name}"
            )
        logger.info(
            "NVIDIA megakernel: %s runs the decode graphs of batch sizes %s (%d of "
            "%d); every other step runs the standard forward",
            self.spec.name,
            routed,
            len(routed),
            len(captured),
        )


def _tokens_per_request() -> int:
    """Tokens per request of this server's decode-phase steps."""
    spec_config = get_spec()
    if spec_config.speculative_algorithm is None:
        return 1
    return int(spec_config.speculative_num_draft_tokens or 1)


def _decode_graphs_enabled() -> bool:
    graph = get_exec().graph.cuda_graph_config
    return graph is not None and graph.decode.backend != Backend.DISABLED


def _gather_tp(obj: Any, group: Any = None) -> list[Any]:
    """Gather a value from every rank of group (default: TP) through its CPU group."""
    group = get_parallel().tp_group if group is None else group
    if group.world_size == 1:
        return [obj]
    out: list[Any] = [None] * group.world_size
    dist.all_gather_object(out, obj, group=group.cpu_group)
    return out


def _gather_reasons(reasons: list[str]) -> list[str]:
    return [
        f"rank {rank}: {reason}"
        for rank, rank_reasons in enumerate(_gather_tp(reasons))
        for reason in rank_reasons
    ]


def _refuse(reasons: list[str]) -> None:
    if reasons:
        raise ValueError(
            f"{FLAG} cannot run this setup:\n  - " + "\n  - ".join(reasons)
        )
