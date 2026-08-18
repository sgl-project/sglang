# SPDX-License-Identifier: Apache-2.0
"""Warmup-calibrated automatic component residency promotion.

Under ``--performance-mode auto`` with server warmup, each rank measures the
peak GPU memory of the synthetic warmup requests, extrapolates it to the
model's default workload, and promotes implicitly offloaded components
(component offload -> resident, layerwise offload -> fully loaded) when the
estimate plus the promoted weights still fits under a safety reserve.

The estimate splits the measured peak into a persistent part (weights and
buffers alive before the forward) and an activation part (everything above
it); only the activation part scales with the workload ratio. Scaling the
whole peak would multiply resident weights by the video frame/area cap ratio
(~16x for Wan-class defaults) and promotion would never trigger.

Promotion targets the model default workload only (default resolution,
default frames, batch=1). Larger shapes, batches, or multi-image inputs need
explicit ``--component-residency``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Mapping

import msgspec
import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    is_fsdp_managed_module,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
    is_layerwise_offloaded_module,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    is_dit_component_name,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = init_logger(__name__)

GIB_BYTES = 1024**3

# Activation memory rarely scales perfectly linearly with workload units;
# pad the extrapolated activation part before checking the budget.
ACTIVATION_EXTRAPOLATION_MARGIN = 1.2
# Always keep this much VRAM free after promotion for allocator slack,
# shape variance, and CUDA graph or compile pools.
VRAM_RESERVE_FRACTION = 0.10
MIN_VRAM_RESERVE_BYTES = 4 * GIB_BYTES

AUTO_RESIDENCY_FEATURE_NAME = "auto residency promotion"

PROMOTION_STATUS_SKIPPED = "skipped"
PROMOTION_STATUS_PROMOTED = "promoted"
PROMOTION_STATUS_ROLLED_BACK = "rolled_back"
PROMOTION_STATUS_ROLLBACK_FAILED = "rollback_failed"


class AutoResidencyRollbackError(RuntimeError):
    """A promotion failed AND undoing the already-applied part also failed.

    The rank is in a mixed residency state that only a restart fixes; callers
    must abort startup instead of continuing on a half-rolled-back replica.
    """


def describe_error(error: BaseException) -> str:
    """Never-empty error text (str(AssertionError()) is "" and would be
    dropped by any truthiness filter)."""
    text = str(error)
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


class WarmupMemoryRecord(msgspec.Struct, frozen=True):
    """Per-rank memory measurement of one server warmup forward."""

    width: int
    height: int
    num_frames: int
    baseline_allocated_bytes: int
    peak_reserved_bytes: int
    succeeded: bool

    def workload_units(self) -> int:
        return max(1, self.width) * max(1, self.height) * max(1, self.num_frames)


class PromotionCandidate(msgspec.Struct, frozen=True):
    """An implicitly offloaded component that could become resident."""

    component_name: str
    residency_mode: str
    promoted_weight_bytes: int
    # Estimated per-request host-to-device traffic this promotion removes.
    h2d_bytes_per_request: int


class DefaultWorkload(msgspec.Struct, frozen=True):
    """The model-default request shape the promotion is calibrated for."""

    width: int | None
    height: int | None
    num_frames: int
    num_inference_steps: int

    def workload_units(self) -> int | None:
        if self.width is None or self.height is None:
            return None
        return max(1, self.width) * max(1, self.height) * max(1, self.num_frames)

    def describe(self) -> str:
        if self.width is None or self.height is None:
            return "model-default"
        return f"{self.width}x{self.height}x{self.num_frames}f"


class RankResidencyReport(msgspec.Struct, frozen=True):
    """One rank's inputs to the replica-wide promotion decision."""

    rank: int
    budget_bytes: int
    estimated_peak_bytes: int | None
    candidates: list[PromotionCandidate] = []
    skip_reason: str | None = None


class AutoResidencyPlan(msgspec.Struct, frozen=True):
    """Deterministic promotion decision shared by every rank."""

    estimated_peak_bytes: int = 0
    reserve_bytes: int = 0
    budget_bytes: int = 0
    promotions: list[PromotionCandidate] = []
    skip_reason: str | None = None


class AppliedPromotion(msgspec.Struct, frozen=True):
    component_name: str
    residency_mode: str


def resolve_default_workload(server_args: ServerArgs) -> DefaultWorkload:
    """Resolve the default request shape promotion is optimized for."""
    from sglang.multimodal_gen.runtime.warmup_request_builder import (
        _apply_warmup_frame_contract,
        get_model_sampling_defaults,
    )

    defaults = get_model_sampling_defaults(server_args)
    # __post_init__ already applied the model's _default_width/_default_height
    width = defaults.width
    height = defaults.height
    if (width is None or height is None) and defaults.supported_resolutions:
        # worst-case the target with the largest supported shape
        width, height = max(
            defaults.supported_resolutions, key=lambda size: size[0] * size[1]
        )
    num_frames = defaults.num_frames or 1
    if num_frames > 1:
        # same contract the warmup requests and real requests get
        num_frames = _apply_warmup_frame_contract(
            server_args, defaults, num_frames=num_frames
        )
    return DefaultWorkload(
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=defaults.num_inference_steps or 1,
    )


def estimate_default_workload_peak_bytes(
    *,
    records: Iterable[WarmupMemoryRecord],
    target_units: int | None,
) -> int | None:
    """Extrapolate warmup peaks to the default workload.

    Preference order:
    1. A measurement at or above the target workload bounds the peak directly.
    2. Two distinct measured workload sizes fit ``peak = constant + slope *
       units``; only the fitted linear part is extrapolated (and padded).
       Under offload the pre-forward baseline is nearly empty, so a
       single-point split cannot separate constant costs (streamed weights,
       attention workspace, tiled VAE decode) from workload-linear
       activations -- measured on Wan2.1-14B, single-point scaling
       overestimated a ~30 GiB peak as ~183 GiB.
    3. One usable size: scale everything above the pre-forward allocated
       baseline (conservative; may block promotion but never over-promotes).

    Returns None when the estimate cannot be trusted: no records, any warmup
    forward failed (its peak does not cover the full request path), or the
    target workload is unknown (an unknown target would silently equate the
    area/frame-capped warmup peak with the real serving peak).
    """
    records = list(records)
    if not records or any(not record.succeeded for record in records):
        return None
    if target_units is None:
        return None

    peak_by_units: dict[int, int] = {}
    for record in records:
        units = record.workload_units()
        peak_by_units[units] = max(
            peak_by_units.get(units, 0), record.peak_reserved_bytes
        )

    covering_peaks = [
        peak for units, peak in peak_by_units.items() if units >= target_units
    ]
    if covering_peaks:
        return max(covering_peaks)

    if len(peak_by_units) >= 2:
        # fit on the two largest sizes: closest to the target, best local slope
        (large_units, large_peak), (small_units, small_peak) = sorted(
            peak_by_units.items(), reverse=True
        )[:2]
        slope = (large_peak - small_peak) / (large_units - small_units)
        if slope >= 0:
            constant = large_peak - slope * large_units
            return int(
                constant + slope * target_units * ACTIVATION_EXTRAPOLATION_MARGIN
            )
        # negative slope is measurement noise; fall through to the
        # conservative single-point estimate

    estimates = []
    for record in records:
        peak = record.peak_reserved_bytes
        baseline = min(record.baseline_allocated_bytes, peak)
        activation = peak - baseline
        ratio = target_units / record.workload_units()
        estimates.append(
            baseline + int(activation * ratio * ACTIVATION_EXTRAPOLATION_MARGIN)
        )
    return max(estimates)


def _module_weight_bytes(module: nn.Module) -> int:
    """Full weight+buffer footprint, reading through layerwise CPU buffers."""
    from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
        iter_materialized_weights,
    )

    param_bytes = sum(
        tensor.numel() * tensor.element_size()
        for _, tensor in iter_materialized_weights(module)
    )
    buffer_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in module.buffers()
    )
    return param_bytes + buffer_bytes


def _layerwise_offloaded_bytes(module: LayerwiseOffloadableModuleMixin) -> int:
    return sum(
        manager.offloaded_weight_bytes()
        for manager in module.layerwise_offload_managers
        if manager.enabled
    )


def component_resident_size_bytes(module: nn.Module, residency_mode: str) -> int:
    """Extra GPU bytes a promotion of this component would pin resident."""
    if residency_mode == LAYERWISE_OFFLOAD:
        if not isinstance(module, LayerwiseOffloadableModuleMixin):
            return 0
        return _layerwise_offloaded_bytes(module)
    return _module_weight_bytes(module)


def collect_promotion_candidates(
    *,
    modules: Mapping[str, object],
    residency_mode_of: Callable[[str], str],
    explicit_residency_mode_of: Callable[[str], str | None],
    custom_strategy_names: Iterable[str],
    num_inference_steps: int,
) -> list[PromotionCandidate]:
    """List implicitly offloaded native components eligible for promotion.

    Only components whose residency was chosen by the auto policy qualify:
    explicit placements, FSDP-managed modules, and components driven by a
    pipeline-custom residency strategy are never touched.
    """
    custom_names = set(custom_strategy_names)
    candidates = []
    for name in sorted(modules):
        module = modules[name]
        if not isinstance(module, nn.Module):
            continue
        if name in custom_names:
            continue
        mode = residency_mode_of(name)
        if mode not in (COMPONENT_OFFLOAD, LAYERWISE_OFFLOAD):
            continue
        if explicit_residency_mode_of(name) is not None:
            continue
        if is_fsdp_managed_module(module):
            continue
        if mode == COMPONENT_OFFLOAD and is_layerwise_offloaded_module(module):
            # The module is layerwise-managed despite its configured mode
            # (e.g. the offload_during_compile window); flipping the flag
            # would be a silent no-op while the manager keeps streaming.
            continue
        if mode == LAYERWISE_OFFLOAD and not is_layerwise_offloaded_module(module):
            continue
        promoted_bytes = component_resident_size_bytes(module, mode)
        if promoted_bytes <= 0:
            continue
        # A layerwise DiT re-streams its layers once per denoise forward;
        # every other offloaded component transfers once per request.
        uses_per_request = (
            max(1, num_inference_steps)
            if mode == LAYERWISE_OFFLOAD and is_dit_component_name(name)
            else 1
        )
        candidates.append(
            PromotionCandidate(
                component_name=name,
                residency_mode=mode,
                promoted_weight_bytes=promoted_bytes,
                h2d_bytes_per_request=promoted_bytes * uses_per_request,
            )
        )
    return candidates


def rank_candidates_by_h2d_savings(
    candidates: Iterable[PromotionCandidate],
) -> list[PromotionCandidate]:
    """Biggest per-request H2D savings first; name breaks ties deterministically.

    Shared by the promotion plan and the post-request residency hint so the
    hint always lists components in the order auto mode would promote them.
    """
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.h2d_bytes_per_request,
            candidate.component_name,
        ),
    )


def _skip_plan(reason: str) -> AutoResidencyPlan:
    return AutoResidencyPlan(skip_reason=reason)


def _consensus_candidates(
    reports: list[RankResidencyReport],
) -> list[PromotionCandidate]:
    """Merge per-rank candidates: keep components every rank agrees on.

    Sizes are worst-cased with the per-rank maximum so a promotion never
    fits on one rank but overflows another.
    """
    per_rank_maps = [
        {candidate.component_name: candidate for candidate in report.candidates}
        for report in reports
    ]
    common_names = set(per_rank_maps[0])
    for candidate_map in per_rank_maps[1:]:
        common_names &= set(candidate_map)

    merged = []
    for name in sorted(common_names):
        rank_candidates = [candidate_map[name] for candidate_map in per_rank_maps]
        modes = {candidate.residency_mode for candidate in rank_candidates}
        if len(modes) != 1:
            continue
        merged.append(
            PromotionCandidate(
                component_name=name,
                residency_mode=modes.pop(),
                promoted_weight_bytes=max(
                    candidate.promoted_weight_bytes for candidate in rank_candidates
                ),
                h2d_bytes_per_request=max(
                    candidate.h2d_bytes_per_request for candidate in rank_candidates
                ),
            )
        )
    return merged


def plan_auto_residency(*, reports: list[RankResidencyReport]) -> AutoResidencyPlan:
    """Turn the gathered rank reports into one deterministic promotion plan."""
    if not reports:
        return _skip_plan("no rank reports")
    for report in reports:
        if report.skip_reason is not None:
            return _skip_plan(f"rank {report.rank}: {report.skip_reason}")
        if report.estimated_peak_bytes is None:
            return _skip_plan(f"rank {report.rank}: no usable warmup measurement")

    estimated_peak = max(report.estimated_peak_bytes for report in reports)
    budget = min(report.budget_bytes for report in reports)
    reserve = max(int(budget * VRAM_RESERVE_FRACTION), MIN_VRAM_RESERVE_BYTES)

    candidates = _consensus_candidates(reports)
    if not candidates:
        return _skip_plan("no implicitly offloaded components to promote")

    ordered = rank_candidates_by_h2d_savings(candidates)
    promotions = []
    promoted_bytes = 0
    for candidate in ordered:
        next_bytes = promoted_bytes + candidate.promoted_weight_bytes
        # Known conservatism: estimated_peak already contains whatever part
        # of this component was on GPU at the measured peak (streamed
        # layers, keep-ready components), so those bytes are counted twice.
        # The error is in the safe direction; refining it needs per-phase
        # peaks.
        if estimated_peak + next_bytes + reserve <= budget:
            promotions.append(candidate)
            promoted_bytes = next_bytes

    return AutoResidencyPlan(
        estimated_peak_bytes=estimated_peak,
        reserve_bytes=reserve,
        budget_bytes=budget,
        promotions=promotions,
    )


def apply_promotions(
    *,
    plan: AutoResidencyPlan,
    modules: Mapping[str, object],
    server_args: ServerArgs,
) -> list[AppliedPromotion]:
    """Flip the planned components to resident on this rank.

    Component-offload modules are only re-marked resident; the physical move
    happens on their next use (the post-promotion re-warmup) through
    ``ResidentStrategy``, which also applies any per-use target dtype.
    Layerwise modules load all layers now and drop their hooks.

    Raises on failure after undoing the promotions already applied, so a
    caller observing an exception knows this rank is back on the original
    strategy -- unless the undo itself fails, which raises
    ``AutoResidencyRollbackError`` (the rank is in a mixed state and startup
    must abort).
    """
    applied: list[AppliedPromotion] = []
    try:
        for candidate in plan.promotions:
            module = modules.get(candidate.component_name)
            if not isinstance(module, nn.Module):
                raise RuntimeError(
                    f"promotion target {candidate.component_name!r} is missing"
                )
            if candidate.residency_mode == LAYERWISE_OFFLOAD and not isinstance(
                module, LayerwiseOffloadableModuleMixin
            ):
                raise RuntimeError(
                    f"promotion target {candidate.component_name!r} lost its "
                    "layerwise offload capability between planning and apply"
                )
            if candidate.residency_mode == COMPONENT_OFFLOAD and (
                is_layerwise_offloaded_module(module)
            ):
                raise RuntimeError(
                    f"promotion target {candidate.component_name!r} became "
                    "layerwise-managed between planning and apply"
                )
            # Record before acting so a mid-promotion failure rolls back the
            # partially promoted component as well.
            applied.append(
                AppliedPromotion(
                    component_name=candidate.component_name,
                    residency_mode=candidate.residency_mode,
                )
            )
            server_args.require_component_resident(
                candidate.component_name, feature_name=AUTO_RESIDENCY_FEATURE_NAME
            )
            if candidate.residency_mode == LAYERWISE_OFFLOAD:
                module.disable_offload()
    except Exception as apply_error:
        try:
            rollback_promotions(
                applied=applied, modules=modules, server_args=server_args
            )
        except Exception as rollback_error:
            raise AutoResidencyRollbackError(
                f"promotion failed ({describe_error(apply_error)}) and rollback "
                f"failed ({describe_error(rollback_error)})"
            ) from apply_error
        raise
    return applied


def rollback_promotions(
    *,
    applied: Iterable[AppliedPromotion],
    modules: Mapping[str, object],
    server_args: ServerArgs,
) -> None:
    """Restore the original residency for previously applied promotions.

    Every promotion is undone even when one of them fails; the collected
    failures are re-raised at the end so one broken component cannot leave
    the later (earlier-applied) ones promoted.
    """
    errors: list[str] = []
    for promotion in reversed(list(applied)):
        try:
            server_args.release_required_component_residency(
                promotion.component_name,
                feature_name=AUTO_RESIDENCY_FEATURE_NAME,
            )
            module = modules.get(promotion.component_name)
            if not isinstance(module, nn.Module):
                continue
            if promotion.residency_mode == LAYERWISE_OFFLOAD:
                if not isinstance(module, LayerwiseOffloadableModuleMixin):
                    raise RuntimeError("lost layerwise offload capability")
                module.enable_offload()
            elif is_layerwise_offloaded_module(module):
                # A layerwise manager owns this module's placement (e.g. the
                # offload_during_compile window); moving it to CPU behind the
                # manager would strand its bookkeeping.
                pass
            else:
                module.to("cpu")
        except Exception as e:
            errors.append(f"{promotion.component_name}: {describe_error(e)}")
    if torch.get_device_module().is_available():
        torch.get_device_module().empty_cache()
    if errors:
        raise RuntimeError("; ".join(errors))


def format_plan_summary(
    *,
    plan: AutoResidencyPlan,
    workload: DefaultWorkload,
    records: Iterable[WarmupMemoryRecord] = (),
) -> str:
    """One-line decision summary for the startup log."""
    if plan.skip_reason is not None:
        return f"Auto residency: skipped ({plan.skip_reason})"
    promoted = (
        ", ".join(
            f"{candidate.component_name}"
            f"({candidate.residency_mode}, "
            f"{candidate.promoted_weight_bytes / GIB_BYTES:.1f} GiB)"
            for candidate in plan.promotions
        )
        or "none"
    )
    measured = ", ".join(
        f"{record.width}x{record.height}x{record.num_frames}f="
        f"{record.peak_reserved_bytes / GIB_BYTES:.1f}GiB"
        for record in records
        if record.succeeded
    )
    measured_part = f"measured=[{measured}], " if measured else ""
    return (
        f"Auto residency: target={workload.describe()} "
        f"steps={workload.num_inference_steps}, "
        f"{measured_part}"
        f"estimated_peak={plan.estimated_peak_bytes / GIB_BYTES:.1f} GiB, "
        f"reserve={plan.reserve_bytes / GIB_BYTES:.1f} GiB, "
        f"budget={plan.budget_bytes / GIB_BYTES:.1f} GiB, "
        f"promoted=[{promoted}]"
    )


def format_applied_changes(*, plan: AutoResidencyPlan) -> str:
    """Describe the applied residency changes as equivalent server args.

    Users can pin the printed ``--component-residency`` flags to freeze this
    placement, or disable the adjustment entirely with the kill switch.
    """
    changes = "; ".join(
        f"{candidate.component_name}: {candidate.residency_mode} -> resident"
        for candidate in plan.promotions
    )
    equivalent = "--component-residency " + " ".join(
        f"{candidate.component_name}=resident" for candidate in plan.promotions
    )
    return (
        f"Auto residency: adjusted {changes}. "
        f"Equivalent server args: {equivalent}. "
        f"Pin these flags to make this placement explicit, or set "
        f"SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY=1 to disable auto adjustment."
    )


def plan_summary_payload(*, plan: AutoResidencyPlan, status: str) -> dict:
    """Minimal decision payload for the warmup orchestrator.

    The orchestrator only branches on ``status``; the human-readable detail
    lives in the logged ``format_plan_summary`` line.
    """
    return {
        "status": status,
        "promoted": [candidate.component_name for candidate in plan.promotions],
    }
