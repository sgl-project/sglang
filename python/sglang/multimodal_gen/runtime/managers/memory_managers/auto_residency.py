# SPDX-License-Identifier: Apache-2.0
"""Warmup-calibrated automatic component residency promotion.

Under ``--performance-mode auto`` with server warmup, each rank measures the
peak GPU memory of bounded synthetic warmup requests and a low-step probe at
the complete default serving shape, then promotes implicitly offloaded components
(component offload -> resident, layerwise offload -> fully loaded) when the
estimate plus the promoted weights still fits under a safety reserve.

When no full-shape measurement is available, the fallback estimate splits the
measured peak into persistent weights and workload-scaled activations. Scaling
the whole peak would multiply resident weights by the video frame/area cap
ratio (~16x for Wan-class defaults) and promotion would never trigger.

Promotion targets the model default workload only (default resolution,
default frames, batch=1). Larger shapes, batches, or multi-image inputs need
explicit ``--component-residency``.

Loading and serving are deliberately separate placement states. The existing
auto policy provides the initial state; when that state can complete loading
and calibration, this module optimizes the long-lived serving state and
validates the transition with a post-placement warmup. It does not force a
single placement to serve two different lifecycle objectives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Sequence

import msgspec
import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.linear import LinearBase
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    is_fsdp_managed_module,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
    HOST_COPY_RESERVE_BYTES,
    host_memory_available_bytes,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
    compute_streamed_layers,
    is_layerwise_offloaded_module,
    iter_materialized_weights,
    release_unused_pinned_memory,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    is_dit_component_name,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.placement_budget import (
    NoFeasiblePlacementError,
    PlacementOption,
    optimize_placement,
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
# The absolute floor is sized for datacenter cards, where the fraction dominates
# anyway (10% of 80 GiB is 8 GiB). Below ~40 GiB the floor takes over, and on a
# 12 GiB card a flat 4 GiB would fence off a third of the device, so cap it as a
# share of what is actually there.
MAX_VRAM_RESERVE_FRACTION = 0.20

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
    phase_peak_reserved_bytes: dict[str, int] = {}
    phase_active_components: dict[str, tuple[str, ...]] = {}

    def workload_units(self) -> int:
        return max(1, self.width) * max(1, self.height) * max(1, self.num_frames)


class PromotionCandidate(msgspec.Struct, frozen=True):
    """An implicitly offloaded component that could become resident."""

    component_name: str
    residency_mode: str
    promoted_weight_bytes: int
    # Estimated per-request host-to-device traffic this promotion removes.
    h2d_bytes_per_request: int
    # Layerwise candidates jointly choose stage-scoped GPU residency and host
    # pinning. None is used by ordinary component-offload promotion.
    target_layerwise_resident_layers: tuple[int, ...] | None = None
    target_layerwise_pinned_layers: tuple[tuple[int, ...], ...] | None = None
    pinned_host_delta_bytes: int = 0
    host_unpin_scratch_bytes: int = 0
    host_pin_scratch_bytes: int = 0
    permanent_residency: bool = False
    # Device-memory delta relative to the measured placement. A component
    # already loaded for its own phase has a different delta from phases where
    # it is absent; keeping both avoids adding the same weights twice.
    active_device_delta_bytes: int = 0
    inactive_device_delta_bytes: int = 0

    def option_key(self) -> str:
        if self.target_layerwise_resident_layers is None:
            return f"{self.component_name}:resident"
        layer_counts = ",".join(
            str(count) for count in self.target_layerwise_resident_layers
        )
        pinned = "|".join(
            ",".join(str(index) for index in indices) or "-"
            for indices in self.target_layerwise_pinned_layers or ()
        )
        permanence = "permanent" if self.permanent_residency else "stage"
        return f"{self.component_name}:{permanence}:layers={layer_counts}:pins={pinned}"


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
    estimated_peak_bytes_by_phase: dict[str, int] = {}
    active_components_by_phase: dict[str, tuple[str, ...]] = {}
    node_rank: int = 0
    pinned_host_bytes: int = 0
    host_pin_capacity_bytes: int = 0
    host_transition_headroom_bytes: int = 0
    candidates: list[PromotionCandidate] = []
    skip_reason: str | None = None


class AutoResidencyPlan(msgspec.Struct, frozen=True):
    """Deterministic promotion decision shared by every rank."""

    estimated_peak_bytes: int = 0
    reserve_bytes: int = 0
    budget_bytes: int = 0
    resource_budget_bytes: dict[str, int] = {}
    promotions: list[PromotionCandidate] = []
    skip_reason: str | None = None


class AppliedPromotion(msgspec.Struct, frozen=True):
    component_name: str
    residency_mode: str
    previous_layerwise_resident_layers: tuple[int, ...] | None = None
    previous_layerwise_pinned_layers: tuple[tuple[int, ...], ...] | None = None
    layerwise_offload_disabled: bool = False
    pinned_host_changed: bool = False


def resolve_default_workload(server_args: ServerArgs) -> DefaultWorkload:
    """Resolve the default request shape promotion is optimized for."""
    from sglang.multimodal_gen.runtime.warmup_request_builder import (
        get_model_sampling_defaults,
        resolve_default_workload_shape,
    )

    defaults = get_model_sampling_defaults(server_args)
    width, height, num_frames = resolve_default_workload_shape(server_args, defaults)
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
    constant_weight_bytes: int = 0,
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

    Returns None when the estimate cannot be trusted: no successful records,
    the target workload is unknown (an unknown target would silently equate the
    area/frame-capped warmup peak with the real serving peak), or a probe at or
    below the target ran out of memory. That last case is not missing data but
    a measurement: the card could not hold the target as it is already
    configured, so making more weights resident can only make it worse. A probe
    that failed strictly above the target says nothing about the target and is
    dropped instead.
    """
    records = list(records)
    if target_units is None:
        return None
    failed_units = [
        record.workload_units() for record in records if not record.succeeded
    ]
    if any(units <= target_units for units in failed_units):
        return None
    records = [record for record in records if record.succeeded]
    if not records:
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
            constant = max(
                large_peak - slope * large_units,
                min(constant_weight_bytes, large_peak),
            )
            return int(
                constant + slope * target_units * ACTIVATION_EXTRAPOLATION_MARGIN
            )
        # negative slope is measurement noise; fall through to the
        # conservative single-point estimate

    estimates = []
    for record in records:
        peak = record.peak_reserved_bytes
        baseline = min(
            max(record.baseline_allocated_bytes, constant_weight_bytes), peak
        )
        activation = peak - baseline
        ratio = target_units / record.workload_units()
        estimates.append(
            baseline + int(activation * ratio * ACTIVATION_EXTRAPOLATION_MARGIN)
        )
    return max(estimates)


def estimate_workload_phase_peaks(
    *,
    records: Iterable[WarmupMemoryRecord],
    target_units: int | None,
    component_weight_bytes: Mapping[str, int],
) -> tuple[dict[str, int], dict[str, tuple[str, ...]]]:
    """Estimate each measured execution phase at the target workload.

    A component already active in a phase is part of that phase's measured
    peak. Keeping a component resident therefore adds no weight bytes to its
    own component-offload phase, while it adds the full footprint to phases
    where the component was absent. Returning both the estimated peaks and the
    conservative intersection of active components preserves that distinction.
    """
    records = list(records)
    successful = [record for record in records if record.succeeded]
    phase_names = sorted(
        {
            phase_name
            for record in successful
            for phase_name in record.phase_peak_reserved_bytes
        }
    )
    estimated_peaks: dict[str, int] = {}
    active_components: dict[str, tuple[str, ...]] = {}
    for phase_name in phase_names:
        phase_records = [
            record
            for record in successful
            if phase_name in record.phase_peak_reserved_bytes
        ]
        if not phase_records:
            continue
        active_sets = [
            set(record.phase_active_components.get(phase_name, ()))
            for record in phase_records
        ]
        active = set.intersection(*active_sets) if active_sets else set()
        weight_floor = sum(component_weight_bytes.get(name, 0) for name in active)
        phase_measurements = [
            WarmupMemoryRecord(
                width=record.width,
                height=record.height,
                num_frames=record.num_frames,
                baseline_allocated_bytes=min(
                    record.baseline_allocated_bytes,
                    record.phase_peak_reserved_bytes[phase_name],
                ),
                peak_reserved_bytes=record.phase_peak_reserved_bytes[phase_name],
                succeeded=True,
            )
            for record in phase_records
        ]
        estimate = estimate_default_workload_peak_bytes(
            records=phase_measurements,
            target_units=target_units,
            constant_weight_bytes=weight_floor,
        )
        if estimate is None:
            continue
        estimated_peaks[phase_name] = estimate
        active_components[phase_name] = tuple(sorted(active))
    return estimated_peaks, active_components


def _module_weight_bytes(module: nn.Module) -> int:
    """Full weight+buffer footprint, reading through layerwise CPU buffers."""
    param_bytes = sum(
        tensor.numel() * tensor.element_size()
        for _, tensor in iter_materialized_weights(module)
    )
    buffer_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in module.buffers()
    )
    return param_bytes + buffer_bytes


def module_uses_quantized_weights(module: nn.Module) -> bool:
    """Whether a loaded native component contains quantized weight storage."""
    for submodule in module.modules():
        if isinstance(submodule, LinearBase) and submodule.quant_config is not None:
            return True
    quantized_float_dtypes = {torch.float8_e4m3fn, torch.float8_e5m2}
    return any(
        not tensor.is_floating_point() or tensor.dtype in quantized_float_dtypes
        for _, tensor in iter_materialized_weights(module)
    )


def component_runtime_weight_bytes(modules: Mapping[str, object]) -> dict[str, int]:
    """Persistent plus streamed-window weight bytes during a component use."""
    runtime_bytes: dict[str, int] = {}
    for name, module in modules.items():
        if not isinstance(module, nn.Module):
            continue
        full_bytes = _module_weight_bytes(module)
        if not isinstance(module, LayerwiseOffloadableModuleMixin):
            runtime_bytes[name] = full_bytes
            continue
        managers = [
            manager for manager in module.layerwise_offload_managers if manager.enabled
        ]
        if not managers:
            runtime_bytes[name] = full_bytes
            continue
        managed_bytes = sum(manager.offloaded_weight_bytes() for manager in managers)
        managed_peak = sum(
            manager.peak_managed_device_weight_bytes() for manager in managers
        )
        runtime_bytes[name] = max(0, full_bytes - managed_bytes) + managed_peak
    return runtime_bytes


def layerwise_pinned_host_bytes(modules: Mapping[str, object]) -> int:
    """Pinned layer-store bytes in this process, without alias double counts."""
    seen_managers: set[int] = set()
    total = 0
    for module in modules.values():
        if not isinstance(module, LayerwiseOffloadableModuleMixin):
            continue
        for manager in module.layerwise_offload_managers:
            manager_id = id(manager)
            if manager_id in seen_managers:
                continue
            seen_managers.add(manager_id)
            total += manager.pinned_host_weight_bytes()
    return total


def layerwise_host_pin_capacity_bytes(modules: Mapping[str, object]) -> int:
    """This process's non-overlapping HostPin allowance."""
    seen_budgets: set[int] = set()
    total = 0
    for module in modules.values():
        if not isinstance(module, LayerwiseOffloadableModuleMixin):
            continue
        for manager in module.layerwise_offload_managers:
            budget = manager.pin_budget
            budget_id = id(budget)
            if budget_id in seen_budgets:
                continue
            seen_budgets.add(budget_id)
            total += max(0, budget.available_bytes - budget.reserve_bytes)
    return total


def _layerwise_offloaded_bytes(module: LayerwiseOffloadableModuleMixin) -> int:
    return sum(
        sum(manager.layer_weight_bytes().values())
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


def _layerwise_transfer_work_bytes(
    *,
    managers: Sequence,
    resident_layers: tuple[int, ...],
    pinned_layers: tuple[tuple[int, ...], ...],
    uses_per_streamed_layer: int,
) -> int:
    """Relative transfer work for one request under a layerwise placement.

    Pageable copies are weighted 2x: the CUDA driver stages them through an
    internal pinned buffer and the copy cannot run ahead of compute. This is a
    conservative ordering metric rather than a latency prediction.
    """
    total = 0
    for manager, resident_count, pinned_indices in zip(
        managers, resident_layers, pinned_layers
    ):
        streamed = set(
            compute_streamed_layers(
                num_layers=manager.num_layers,
                resident_layers=resident_count,
                policy=manager.residency_policy,
            )
        )
        pinned = set(pinned_indices)
        for layer_idx, weight_bytes in manager.layer_weight_bytes().items():
            uses = uses_per_streamed_layer if layer_idx in streamed else 1
            transfer_multiplier = 1 if layer_idx in pinned else 2
            total += uses * transfer_multiplier * weight_bytes
    return total


def _layerwise_pin_targets(
    *,
    managers: Sequence,
    resident_layers: tuple[int, ...],
    current_pinned_layers: tuple[tuple[int, ...], ...],
    uses_per_streamed_layer: int,
) -> list[tuple[tuple[int, ...], ...]]:
    """Useful HostPin frontiers for one resident-layer placement.

    Streamed layers have the highest pin value because they transfer every
    denoise step. Prefixes of that deterministic value order cover every
    capacity breakpoint without enumerating all 2^N layer subsets.
    """
    ordered: list[tuple[int, int, int]] = []
    for manager_index, (manager, resident_count) in enumerate(
        zip(managers, resident_layers)
    ):
        streamed = set(
            compute_streamed_layers(
                num_layers=manager.num_layers,
                resident_layers=resident_count,
                policy=manager.residency_policy,
            )
        )
        if not manager.pin_cpu_memory:
            continue
        for layer_idx in manager.pinnable_layer_indices():
            uses = uses_per_streamed_layer if layer_idx in streamed else 1
            ordered.append((-uses, manager_index, layer_idx))
    ordered.sort()

    targets: list[tuple[tuple[int, ...], ...]] = [current_pinned_layers]
    selected = [set() for _ in managers]
    targets.append(tuple(() for _ in managers))
    for _, manager_index, layer_idx in ordered:
        selected[manager_index].add(layer_idx)
        targets.append(tuple(tuple(sorted(indices)) for indices in selected))
    return list(dict.fromkeys(targets))


def _layerwise_host_transition_bytes(
    *,
    managers: Sequence,
    current_pinned_layers: tuple[tuple[int, ...], ...],
    target_pinned_layers: tuple[tuple[int, ...], ...],
) -> tuple[int, int]:
    """Temporary host allocations needed by the release/acquire phases."""
    unpin_bytes = 0
    pin_bytes = 0
    for manager, current_indices, target_indices in zip(
        managers, current_pinned_layers, target_pinned_layers
    ):
        layer_bytes = manager.layer_host_store_bytes()
        current = set(current_indices)
        target = set(target_indices)
        unpin_bytes += sum(layer_bytes.get(index, 0) for index in current - target)
        pin_bytes += sum(layer_bytes.get(index, 0) for index in target - current)
    return unpin_bytes, pin_bytes


def collect_promotion_candidates(
    *,
    modules: Mapping[str, object],
    residency_mode_of: Callable[[str], str],
    explicit_residency_mode_of: Callable[[str], str | None],
    custom_strategy_names: Iterable[str],
    num_inference_steps: int,
    allow_host_pin_reallocation: bool = True,
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
        if mode == COMPONENT_OFFLOAD:
            candidates.append(
                PromotionCandidate(
                    component_name=name,
                    residency_mode=mode,
                    promoted_weight_bytes=promoted_bytes,
                    h2d_bytes_per_request=promoted_bytes,
                    permanent_residency=True,
                    active_device_delta_bytes=0,
                    inactive_device_delta_bytes=promoted_bytes,
                )
            )
            continue

        if not isinstance(module, LayerwiseOffloadableModuleMixin):
            continue
        managers = [
            manager for manager in module.layerwise_offload_managers if manager.enabled
        ]
        if not managers:
            continue
        current_resident_layers = tuple(manager.resident_layers for manager in managers)
        current_pinned_layers = tuple(
            manager.pinned_layer_indices() for manager in managers
        )
        current_resident_bytes = sum(
            manager.resident_weight_bytes() for manager in managers
        )
        current_peak_device_bytes = sum(
            manager.peak_managed_device_weight_bytes() for manager in managers
        )
        current_pinned_bytes = sum(
            manager.pinned_host_weight_bytes() for manager in managers
        )
        current_transfer_work = _layerwise_transfer_work_bytes(
            managers=managers,
            resident_layers=current_resident_layers,
            pinned_layers=current_pinned_layers,
            uses_per_streamed_layer=uses_per_request,
        )

        resident_targets = {current_resident_layers}
        if is_dit_component_name(name) and num_inference_steps > 1:
            for target in range(
                max(current_resident_layers, default=0),
                max(manager.num_layers for manager in managers) + 1,
            ):
                counts = tuple(min(target, manager.num_layers) for manager in managers)
                if all(
                    count >= current
                    for count, current in zip(counts, current_resident_layers)
                ):
                    resident_targets.add(counts)

        for target_resident_layers in sorted(resident_targets):
            target_resident_bytes = sum(
                manager.resident_weight_bytes(count)
                for manager, count in zip(managers, target_resident_layers)
            )
            target_peak_device_bytes = sum(
                manager.peak_managed_device_weight_bytes(count)
                for manager, count in zip(managers, target_resident_layers)
            )
            incremental_bytes = target_resident_bytes - current_resident_bytes
            pin_targets = (
                _layerwise_pin_targets(
                    managers=managers,
                    resident_layers=target_resident_layers,
                    current_pinned_layers=current_pinned_layers,
                    uses_per_streamed_layer=uses_per_request,
                )
                if allow_host_pin_reallocation
                else [current_pinned_layers]
            )
            for target_pinned_layers in pin_targets:
                if (
                    target_resident_layers == current_resident_layers
                    and target_pinned_layers == current_pinned_layers
                ):
                    continue
                target_pinned_bytes = sum(
                    sum(
                        manager.layer_host_store_bytes().get(layer_idx, 0)
                        for layer_idx in pinned_indices
                    )
                    for manager, pinned_indices in zip(managers, target_pinned_layers)
                )
                target_transfer_work = _layerwise_transfer_work_bytes(
                    managers=managers,
                    resident_layers=target_resident_layers,
                    pinned_layers=target_pinned_layers,
                    uses_per_streamed_layer=uses_per_request,
                )
                unpin_scratch, pin_scratch = _layerwise_host_transition_bytes(
                    managers=managers,
                    current_pinned_layers=current_pinned_layers,
                    target_pinned_layers=target_pinned_layers,
                )
                candidates.append(
                    PromotionCandidate(
                        component_name=name,
                        residency_mode=mode,
                        promoted_weight_bytes=incremental_bytes,
                        h2d_bytes_per_request=(
                            current_transfer_work - target_transfer_work
                        ),
                        target_layerwise_resident_layers=target_resident_layers,
                        target_layerwise_pinned_layers=target_pinned_layers,
                        pinned_host_delta_bytes=(
                            target_pinned_bytes - current_pinned_bytes
                        ),
                        host_unpin_scratch_bytes=unpin_scratch,
                        host_pin_scratch_bytes=pin_scratch,
                        active_device_delta_bytes=max(
                            0, target_peak_device_bytes - current_peak_device_bytes
                        ),
                        inactive_device_delta_bytes=0,
                    )
                )

        full_resident_layers = tuple(manager.num_layers for manager in managers)
        # Fully resident layers never read their host stores. On one worker we
        # can release those pins and reuse the allowance. Multi-worker pin
        # migration needs a node-coordinated host-RAM scratch budget because
        # repacking briefly holds both the old and new layer buffers.
        permanent_pin_targets = [current_pinned_layers]
        if allow_host_pin_reallocation:
            permanent_pin_targets.append(tuple(() for _ in managers))
        for permanent_pins in dict.fromkeys(permanent_pin_targets):
            permanent_pinned_bytes = sum(
                sum(
                    manager.layer_host_store_bytes().get(layer_idx, 0)
                    for layer_idx in pinned_indices
                )
                for manager, pinned_indices in zip(managers, permanent_pins)
            )
            unpin_scratch, pin_scratch = _layerwise_host_transition_bytes(
                managers=managers,
                current_pinned_layers=current_pinned_layers,
                target_pinned_layers=permanent_pins,
            )
            candidates.append(
                PromotionCandidate(
                    component_name=name,
                    residency_mode=mode,
                    promoted_weight_bytes=promoted_bytes - current_resident_bytes,
                    h2d_bytes_per_request=current_transfer_work,
                    target_layerwise_resident_layers=full_resident_layers,
                    target_layerwise_pinned_layers=permanent_pins,
                    pinned_host_delta_bytes=(
                        permanent_pinned_bytes - current_pinned_bytes
                    ),
                    host_unpin_scratch_bytes=unpin_scratch,
                    host_pin_scratch_bytes=pin_scratch,
                    permanent_residency=True,
                    active_device_delta_bytes=max(
                        0, promoted_bytes - current_peak_device_bytes
                    ),
                    # Stage-scoped resident layers are released after the use,
                    # so the complete managed footprint is new elsewhere.
                    inactive_device_delta_bytes=promoted_bytes,
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
            candidate.option_key(),
        ),
    )


def _skip_plan(reason: str) -> AutoResidencyPlan:
    return AutoResidencyPlan(skip_reason=reason)


def _vram_reserve_bytes(budget_bytes: int) -> int:
    return max(
        int(budget_bytes * VRAM_RESERVE_FRACTION),
        min(
            MIN_VRAM_RESERVE_BYTES,
            int(budget_bytes * MAX_VRAM_RESERVE_FRACTION),
        ),
    )


def _binding_phase_constraints(
    report: RankResidencyReport,
) -> list[tuple[str, int, tuple[str, ...]]]:
    """Keep only the highest peak for each active-component placement.

    Every candidate has the same device delta in phases with the same active
    component set. The lower peaks are therefore provably redundant; removing
    them reduces the exact optimizer's resource dimension without changing
    its feasible placements.
    """
    phase_peaks = report.estimated_peak_bytes_by_phase
    if not phase_peaks:
        if report.estimated_peak_bytes is None:
            return []
        phase_peaks = {"request": report.estimated_peak_bytes}
    binding: dict[tuple[str, ...], tuple[str, int]] = {}
    for phase_name, phase_peak in phase_peaks.items():
        active = tuple(sorted(report.active_components_by_phase.get(phase_name, ())))
        current = binding.get(active)
        if current is None or (phase_peak, phase_name) > (current[1], current[0]):
            binding[active] = (phase_name, phase_peak)
    return [
        (f"gpu:rank{report.rank}:{phase_name}", phase_peak, active)
        for active, (phase_name, phase_peak) in sorted(
            binding.items(), key=lambda item: item[1][0]
        )
    ]


def _consensus_candidates(
    reports: list[RankResidencyReport],
) -> list[PromotionCandidate]:
    """Merge per-rank candidates: keep components every rank agrees on.

    Sizes are worst-cased with the per-rank maximum so a promotion never
    fits on one rank but overflows another.
    """
    per_rank_maps = [
        {candidate.option_key(): candidate for candidate in report.candidates}
        for report in reports
    ]
    common_keys = set(per_rank_maps[0])
    for candidate_map in per_rank_maps[1:]:
        common_keys &= set(candidate_map)

    merged = []
    for option_key in sorted(common_keys):
        rank_candidates = [candidate_map[option_key] for candidate_map in per_rank_maps]
        modes = {candidate.residency_mode for candidate in rank_candidates}
        if len(modes) != 1:
            continue
        component_names = {candidate.component_name for candidate in rank_candidates}
        targets = {
            candidate.target_layerwise_resident_layers for candidate in rank_candidates
        }
        pinned_targets = {
            candidate.target_layerwise_pinned_layers for candidate in rank_candidates
        }
        permanent = {candidate.permanent_residency for candidate in rank_candidates}
        if (
            len(component_names) != 1
            or len(targets) != 1
            or len(pinned_targets) != 1
            or len(permanent) != 1
        ):
            continue
        merged.append(
            PromotionCandidate(
                component_name=component_names.pop(),
                residency_mode=modes.pop(),
                promoted_weight_bytes=max(
                    candidate.promoted_weight_bytes for candidate in rank_candidates
                ),
                h2d_bytes_per_request=min(
                    candidate.h2d_bytes_per_request for candidate in rank_candidates
                ),
                target_layerwise_resident_layers=targets.pop(),
                target_layerwise_pinned_layers=pinned_targets.pop(),
                pinned_host_delta_bytes=max(
                    candidate.pinned_host_delta_bytes for candidate in rank_candidates
                ),
                host_unpin_scratch_bytes=max(
                    candidate.host_unpin_scratch_bytes for candidate in rank_candidates
                ),
                host_pin_scratch_bytes=max(
                    candidate.host_pin_scratch_bytes for candidate in rank_candidates
                ),
                permanent_residency=permanent.pop(),
                active_device_delta_bytes=max(
                    candidate.active_device_delta_bytes for candidate in rank_candidates
                ),
                inactive_device_delta_bytes=max(
                    candidate.inactive_device_delta_bytes
                    for candidate in rank_candidates
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
    reserves_by_rank = {
        report.rank: _vram_reserve_bytes(report.budget_bytes) for report in reports
    }
    reserve = max(reserves_by_rank.values())

    candidates = _consensus_candidates(reports)
    if not candidates:
        return _skip_plan("no implicitly offloaded components to promote")

    phase_constraints = {
        report.rank: _binding_phase_constraints(report) for report in reports
    }
    resource_budgets: dict[str, int] = {}
    for report in reports:
        for resource_name, phase_peak, _ in phase_constraints[report.rank]:
            resource_budgets[resource_name] = (
                report.budget_bytes - phase_peak - reserves_by_rank[report.rank]
            )
    has_hostpin_options = any(
        candidate.pinned_host_delta_bytes != 0
        or candidate.host_unpin_scratch_bytes != 0
        or candidate.host_pin_scratch_bytes != 0
        for candidate in candidates
    )
    if has_hostpin_options:
        if len(reports) != 1:
            return _skip_plan("dynamic HostPin placement requires a single-worker node")
        report = reports[0]
        resource_budgets[f"hostpin:node{report.node_rank}"] = (
            report.host_pin_capacity_bytes - report.pinned_host_bytes
        )
        resource_budgets[f"hostram:node{report.node_rank}:unpin"] = (
            report.host_transition_headroom_bytes
        )
        resource_budgets[f"hostram:node{report.node_rank}:pin"] = (
            report.host_transition_headroom_bytes
        )

    candidate_by_key = {candidate.option_key(): candidate for candidate in candidates}
    report_candidates = [
        {candidate.option_key(): candidate for candidate in report.candidates}
        for report in reports
    ]

    options = []
    for candidate in candidates:
        resource_deltas: dict[str, int] = {}
        for report, rank_candidates in zip(reports, report_candidates):
            rank_candidate = rank_candidates[candidate.option_key()]
            for resource_name, _, active_components in phase_constraints[report.rank]:
                component_is_active = candidate.component_name in active_components
                phase_cost = (
                    rank_candidate.active_device_delta_bytes
                    if component_is_active
                    else rank_candidate.inactive_device_delta_bytes
                )
                resource_deltas[resource_name] = phase_cost
            if has_hostpin_options:
                host_resource = f"hostpin:node{report.node_rank}"
                resource_deltas[host_resource] = (
                    resource_deltas.get(host_resource, 0)
                    + rank_candidate.pinned_host_delta_bytes
                )
                resource_deltas[f"hostram:node{report.node_rank}:unpin"] = (
                    rank_candidate.host_unpin_scratch_bytes
                )
                resource_deltas[f"hostram:node{report.node_rank}:pin"] = (
                    rank_candidate.host_pin_scratch_bytes
                )
        options.append(
            PlacementOption(
                group_key=candidate.component_name,
                option_key=candidate.option_key(),
                resource_delta_bytes=resource_deltas,
                estimated_latency_savings=candidate.h2d_bytes_per_request,
            )
        )

    try:
        placement = optimize_placement(
            options,
            resource_budget_bytes=resource_budgets,
        )
    except NoFeasiblePlacementError as error:
        return _skip_plan(str(error))
    promotions = rank_candidates_by_h2d_savings(
        candidate_by_key[selection.option_key] for selection in placement.selections
    )

    return AutoResidencyPlan(
        estimated_peak_bytes=estimated_peak,
        reserve_bytes=reserve,
        budget_bytes=budget,
        resource_budget_bytes=resource_budgets,
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
    Layerwise options either retain a planned number of layers during the
    denoise stage or load the whole component and drop its hooks.

    Raises on failure after undoing the promotions already applied, so a
    caller observing an exception knows this rank is back on the original
    strategy -- unless the undo itself fails, which raises
    ``AutoResidencyRollbackError`` (the rank is in a mixed state and startup
    must abort).
    """
    applied: list[AppliedPromotion] = []
    try:
        ordered_promotions = sorted(
            plan.promotions,
            key=lambda candidate: candidate.pinned_host_delta_bytes,
        )
        promotion_modules: dict[str, nn.Module] = {}
        snapshots: dict[str, AppliedPromotion] = {}
        previous_pins_by_component: dict[str, tuple[tuple[int, ...], ...]] = {}
        for candidate in ordered_promotions:
            if candidate.component_name in promotion_modules:
                raise RuntimeError(
                    f"multiple placement options selected for "
                    f"{candidate.component_name!r}"
                )
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
            promotion_modules[candidate.component_name] = module
            if candidate.target_layerwise_resident_layers is not None:
                if not isinstance(module, LayerwiseOffloadableModuleMixin):
                    raise RuntimeError(
                        f"partial promotion target {candidate.component_name!r} "
                        "lost its layerwise offload capability"
                    )
                previous = tuple(
                    manager.resident_layers
                    for manager in module.layerwise_offload_managers
                )
                previous_pinned = module.layerwise_pinned_layers()
                previous_pins_by_component[candidate.component_name] = previous_pinned
                snapshots[candidate.component_name] = AppliedPromotion(
                    component_name=candidate.component_name,
                    residency_mode=candidate.residency_mode,
                    previous_layerwise_resident_layers=previous,
                    previous_layerwise_pinned_layers=previous_pinned,
                    layerwise_offload_disabled=candidate.permanent_residency,
                    pinned_host_changed=(
                        candidate.target_layerwise_pinned_layers != previous_pinned
                    ),
                )
                if candidate.target_layerwise_pinned_layers is None:
                    raise RuntimeError(
                        f"layerwise placement {candidate.option_key()!r} has no "
                        "host pin target"
                    )
                continue

            snapshots[candidate.component_name] = AppliedPromotion(
                component_name=candidate.component_name,
                residency_mode=candidate.residency_mode,
            )
        pinning_changes = [
            candidate
            for candidate in ordered_promotions
            if candidate.target_layerwise_pinned_layers is not None
            and candidate.target_layerwise_pinned_layers
            != previous_pins_by_component[candidate.component_name]
        ]
        if pinning_changes:
            host_headroom = max(
                0, host_memory_available_bytes() - HOST_COPY_RESERVE_BYTES
            )
            required_unpin = sum(
                candidate.host_unpin_scratch_bytes for candidate in pinning_changes
            )
            required_pin = sum(
                candidate.host_pin_scratch_bytes for candidate in pinning_changes
            )
            if max(required_unpin, required_pin) > host_headroom:
                raise MemoryError(
                    "host memory changed after planning: HostPin repack needs "
                    f"{max(required_unpin, required_pin) / GIB_BYTES:.2f} GiB "
                    f"but {host_headroom / GIB_BYTES:.2f} GiB is available"
                )
        applied_names: set[str] = set()
        # 1. release every component's old allowance
        # 2. return cached pinned blocks to the driver
        # 3. acquire the final allowances
        # This avoids a transient HostPin oversubscription when one component
        # hands its budget to another.
        for candidate in pinning_changes:
            applied.append(snapshots[candidate.component_name])
            applied_names.add(candidate.component_name)
            module = promotion_modules[candidate.component_name]
            assert isinstance(module, LayerwiseOffloadableModuleMixin)
            module.release_layerwise_pins_outside(
                candidate.target_layerwise_pinned_layers or ()
            )
        if pinning_changes:
            release_unused_pinned_memory()
        for candidate in ordered_promotions:
            if candidate.component_name not in applied_names:
                applied.append(snapshots[candidate.component_name])
                applied_names.add(candidate.component_name)
            module = promotion_modules[candidate.component_name]
            if candidate.target_layerwise_resident_layers is not None:
                assert isinstance(module, LayerwiseOffloadableModuleMixin)
                module.set_layerwise_pinned_layers(
                    candidate.target_layerwise_pinned_layers or ()
                )
                if candidate.permanent_residency:
                    server_args.require_component_resident(
                        candidate.component_name,
                        feature_name=AUTO_RESIDENCY_FEATURE_NAME,
                    )
                    module.disable_offload()
                else:
                    module.set_layerwise_resident_layer_counts(
                        candidate.target_layerwise_resident_layers
                    )
                continue
            server_args.require_component_resident(
                candidate.component_name, feature_name=AUTO_RESIDENCY_FEATURE_NAME
            )
            if candidate.residency_mode == LAYERWISE_OFFLOAD:
                assert isinstance(module, LayerwiseOffloadableModuleMixin)
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
    applied = list(applied)
    errors: list[str] = []
    pinning_changes = [
        promotion
        for promotion in applied
        if promotion.pinned_host_changed
        and promotion.previous_layerwise_pinned_layers is not None
    ]
    for promotion in pinning_changes:
        try:
            module = modules.get(promotion.component_name)
            if not isinstance(module, LayerwiseOffloadableModuleMixin):
                raise RuntimeError("lost layerwise offload capability")
            module.release_layerwise_pins_outside(
                promotion.previous_layerwise_pinned_layers or ()
            )
        except Exception as e:
            errors.append(f"{promotion.component_name}: {describe_error(e)}")
    if pinning_changes:
        release_unused_pinned_memory()
    for promotion in reversed(applied):
        try:
            module = modules.get(promotion.component_name)
            if promotion.previous_layerwise_resident_layers is not None:
                if not isinstance(module, LayerwiseOffloadableModuleMixin):
                    raise RuntimeError("lost layerwise offload capability")
                if promotion.layerwise_offload_disabled:
                    module.enable_offload()
                    server_args.release_required_component_residency(
                        promotion.component_name,
                        feature_name=AUTO_RESIDENCY_FEATURE_NAME,
                    )
                module.restore_layerwise_resident_layers(
                    promotion.previous_layerwise_resident_layers
                )
                if promotion.previous_layerwise_pinned_layers is None:
                    raise RuntimeError("lost previous layerwise host placement")
                module.restore_layerwise_pinned_layers(
                    promotion.previous_layerwise_pinned_layers
                )
                continue
            server_args.release_required_component_residency(
                promotion.component_name,
                feature_name=AUTO_RESIDENCY_FEATURE_NAME,
            )
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
        ", ".join(_format_candidate_summary(candidate) for candidate in plan.promotions)
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
        _format_promotion_change(candidate) for candidate in plan.promotions
    )
    component_args = [
        f"{candidate.component_name}=resident"
        for candidate in plan.promotions
        if candidate.permanent_residency
    ]
    equivalent = (
        "--component-residency " + " ".join(component_args)
        if component_args
        else "none (the selected partial layer/HostPin placement is auto-only)"
    )
    return (
        f"Auto residency: adjusted {changes}. "
        f"Equivalent server args: {equivalent}. "
        f"Pin these flags to make this placement explicit, or set "
        f"SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY=1 to disable auto adjustment."
    )


def _format_promotion_change(candidate: PromotionCandidate) -> str:
    if candidate.target_layerwise_resident_layers is not None:
        pin_counts = tuple(
            len(indices) for indices in candidate.target_layerwise_pinned_layers or ()
        )
        scope = "permanent" if candidate.permanent_residency else "stage-scoped"
        return (
            f"{candidate.component_name}: {scope} resident layers="
            f"{candidate.target_layerwise_resident_layers}, pinned layers="
            f"{pin_counts}"
        )
    return f"{candidate.component_name}: {candidate.residency_mode} -> resident"


def _format_candidate_summary(candidate: PromotionCandidate) -> str:
    target = ""
    if candidate.target_layerwise_resident_layers is not None:
        target = (
            f", layers={candidate.target_layerwise_resident_layers}, "
            f"pins={tuple(len(indices) for indices in candidate.target_layerwise_pinned_layers or ())}"
        )
    return (
        f"{candidate.component_name}({candidate.residency_mode}{target}, "
        f"{candidate.promoted_weight_bytes / GIB_BYTES:.1f} GiB)"
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
