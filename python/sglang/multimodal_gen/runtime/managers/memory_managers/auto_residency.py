# SPDX-License-Identifier: Apache-2.0
"""Warmup-calibrated automatic component residency placement.

Under ``--performance-mode auto`` with server warmup, each rank measures the
peak GPU memory of bounded synthetic warmup requests and a low-step probe at
the complete default serving shape, then selects a complete serving placement
for every eligible component under the measured memory constraints.

When no full-shape measurement is available, the fallback estimate splits the
measured peak into persistent weights and workload-scaled activations. Scaling
the whole peak would multiply resident weights by the video frame/area cap
ratio (~16x for Wan-class defaults) and residency adjustment would never trigger.

The planner targets the model default workload only (default resolution,
default frames, batch=1). Larger shapes, batches, or multi-image inputs need
explicit ``--component-residency``.

Loading and serving are deliberately separate placement states. The existing
auto policy provides the initial state; when that state can complete loading
and calibration, this module optimizes the long-lived serving state and
validates the transition with a post-placement warmup. It does not force a
single placement to serve two different lifecycle objectives.
"""

from __future__ import annotations

import statistics
import time
from itertools import chain, product
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Sequence

import msgspec
import torch.nn as nn

from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
    RESIDENT,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    is_fsdp_managed_module,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
    tensor_storage_bytes,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
    compute_streamed_layers,
    estimate_layer_weight_bytes,
    is_layerwise_offloaded_module,
    iter_materialized_weights,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    is_dit_component_name,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.placement_budget import (
    NoFeasiblePlacementError,
    PlacementOption,
    optimize_placement,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = init_logger(__name__)

GIB_BYTES = 1024**3

# Activation memory rarely scales perfectly linearly with workload units;
# pad the extrapolated activation part before checking the budget.
ACTIVATION_EXTRAPOLATION_MARGIN = 1.2
# A target-shape measurement plus the mandatory post-placement warmup justifies
# a tighter reserve than an extrapolated estimate. Both retain an absolute
# floor for allocator slack, shape variance, and CUDA graph or compile pools.
MEASURED_VRAM_RESERVE_FRACTION = 0.05
EXTRAPOLATED_VRAM_RESERVE_FRACTION = 0.10
MIN_VRAM_RESERVE_BYTES = 4 * GIB_BYTES
# The absolute floor is sized for datacenter cards, where either fraction can
# dominate. On a 12 GiB card a flat 4 GiB would fence off a third of the device,
# so cap the floor as a share of what is actually there.
MAX_VRAM_RESERVE_FRACTION = 0.20

# A feasible placement is not automatically useful. Predictions inside this
# interval are treated as latency-equivalent. The joint optimizer then avoids
# changing strategy, minimizes device memory, preserves the faster estimate,
# and finally minimizes HostPin. The raw estimate is already an upper bound:
# transfer time is capped by the measured request.
ESTIMATED_PINNED_H2D_BYTES_PER_SECOND = 24 * GIB_BYTES
MIN_LATENCY_EQUIVALENCE_NS = 50_000_000
MAX_LATENCY_EQUIVALENCE_NS = 100_000_000
LATENCY_EQUIVALENCE_FRACTION = 0.01
# The transfer model ranks feasible placements; the mandatory warmup is the
# authority on whether a selected placement actually helped. Allow normal
# measurement noise, but undo a round whose calibrated request is materially
# slower than the original layout.
POST_ADJUSTMENT_REGRESSION_FRACTION = 0.05

PLACEMENT_STATUS_SKIPPED = "skipped"
PLACEMENT_STATUS_ADJUSTED = "adjusted"
PLACEMENT_STATUS_VALIDATED = "validated"
PLACEMENT_STATUS_ROLLED_BACK = "rolled_back"
PLACEMENT_STATUS_ROLLBACK_FAILED = "rollback_failed"


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
    peak_allocated_bytes: int
    succeeded: bool
    peak_reserved_bytes: int = 0
    phase_peak_allocated_bytes: dict[str, int] = {}
    phase_active_components: dict[str, tuple[str, ...]] = {}
    phase_used_components: dict[str, tuple[str, ...]] = {}
    phase_full_weight_transition_components: dict[str, tuple[str, ...]] = {}
    layerwise_layer_uses: dict[str, dict[str, tuple[int, ...]]] = {}
    layerwise_layer_uses_by_stage: dict[str, dict[str, dict[str, tuple[int, ...]]]] = {}
    num_inference_steps: int = 1
    total_duration_ms: float = 0.0
    stage_duration_ms: dict[str, float] = {}
    step_duration_ms: tuple[float, ...] = ()
    step_duration_ms_by_stage: dict[str, tuple[float, ...]] = {}
    stage_iterations: dict[str, tuple[int, int]] = {}

    def workload_units(self) -> int:
        return max(1, self.width) * max(1, self.height) * max(1, self.num_frames)


class ResidencyTarget(msgspec.Struct, frozen=True):
    """One complete target state for an auto-managed component."""

    component_name: str
    residency_mode: str
    target_resident_weight_bytes: int
    # Estimated per-request host-to-device traffic this target removes.
    h2d_bytes_per_request: int
    # Layerwise candidates jointly choose stage-scoped GPU residency and host
    # pinning. None is used by ordinary component placement.
    target_layerwise_resident_layers: tuple[int, ...] | None = None
    target_layerwise_pinned_layers: tuple[tuple[int, ...], ...] | None = None
    pinned_host_delta_bytes: int = 0
    host_unpin_scratch_bytes: int = 0
    host_pin_scratch_bytes: int = 0
    host_materialize_scratch_bytes: int = 0
    # Signed device-memory delta while applying the placement before the
    # validation warmup. Layerwise -> resident materializes every managed
    # layer immediately; a demotion can release those bytes first and fund a
    # later materialization in the same transaction.
    device_transition_delta_bytes: int = 0
    permanent_residency: bool = False
    # Device-memory delta relative to the measured placement. A component
    # already loaded for its own phase has a different delta from phases where
    # it is absent; keeping both avoids adding the same weights twice.
    active_device_delta_bytes: int = 0
    # Delta when the component is already present because of async prefetch,
    # but is not the semantic owner of this phase.
    present_device_delta_bytes: int = 0
    inactive_device_delta_bytes: int = 0
    # None preserves the historical derived target for hand-built callers:
    # partial layerwise targets remain layerwise, every other option is
    # resident. Generated complete-state frontiers set this explicitly.
    target_residency_mode: str | None = None
    current_placement: bool = False
    target_device_weight_bytes: int = 0
    target_pinned_host_bytes: int = 0

    def target_mode(self) -> str:
        if self.target_residency_mode is not None:
            return self.target_residency_mode
        if (
            self.target_layerwise_resident_layers is not None
            and not self.permanent_residency
        ):
            return LAYERWISE_OFFLOAD
        return RESIDENT

    def option_key(self) -> str:
        target_mode = self.target_mode()
        if target_mode == COMPONENT_OFFLOAD:
            return f"{self.component_name}:component-offload"
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
    """The model-default request shape the planner is calibrated for."""

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
    """One rank's inputs to the replica-wide placement decision."""

    rank: int
    budget_bytes: int
    estimated_peak_bytes: int | None
    target_workload_measured: bool = False
    observed_reserved_bytes: int = 0
    estimated_peak_bytes_by_phase: dict[str, int] = {}
    active_components_by_phase: dict[str, tuple[str, ...]] = {}
    used_components_by_phase: dict[str, tuple[str, ...]] = {}
    full_weight_transition_components_by_phase: dict[str, tuple[str, ...]] = {}
    current_device_weight_bytes_by_component: dict[str, int] = {}
    node_rank: int = 0
    pinned_host_bytes: int = 0
    host_pin_capacity_bytes: int = 0
    host_transition_headroom_bytes: int = 0
    device_transition_allocated_bytes: int = 0
    estimated_request_duration_ns: int = 0
    measured_request_duration_ns: int = 0
    candidate_latency_savings_ns: dict[str, int] = {}
    candidates: list[ResidencyTarget] = []
    skip_reason: str | None = None


class AutoResidencyPlan(msgspec.Struct, frozen=True):
    """Deterministic placement changes shared by every rank."""

    estimated_peak_bytes: int = 0
    reserve_bytes: int = 0
    budget_bytes: int = 0
    resource_budget_bytes: dict[str, int] = {}
    changes: list[ResidencyTarget] = []
    skip_reason: str | None = None
    current_placement_reserve_shortfall_bytes: int = 0


def residency_device_growth_bytes(candidate: ResidencyTarget) -> int:
    """Largest modeled accelerator increase while realizing one target."""
    return max(
        candidate.device_transition_delta_bytes,
        candidate.active_device_delta_bytes,
        candidate.present_device_delta_bytes,
        candidate.inactive_device_delta_bytes,
    )


def resolve_default_workload(server_args: ServerArgs) -> DefaultWorkload:
    """Resolve the default request shape the planner is optimized for."""
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


def resolve_measured_default_workload(
    workload: DefaultWorkload, records: Iterable[WarmupMemoryRecord]
) -> DefaultWorkload:
    """Fill an implicit default resolution from the executed warmup.

    Image-edit pipelines can derive their output size from the input image, so
    the sampling defaults legitimately omit width and height. The warmup record
    is captured after input validation and therefore contains the effective
    serving shape. Keep the model-default frame count because video warmup may
    intentionally cap frames before measurement.
    """
    if workload.workload_units() is not None:
        return workload
    measured = [
        record
        for record in records
        if record.succeeded and record.width > 0 and record.height > 0
    ]
    if not measured:
        return workload
    representative = max(measured, key=lambda record: record.width * record.height)
    return DefaultWorkload(
        width=representative.width,
        height=representative.height,
        num_frames=workload.num_frames,
        num_inference_steps=workload.num_inference_steps,
    )


def estimate_layerwise_layer_uses(
    *,
    records: Iterable[WarmupMemoryRecord],
    target_units: int | None,
    target_num_inference_steps: int,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Estimate per-request layer calls from the same calibration forward.

    A full-shape memory probe deliberately runs only a few denoise steps.
    Stage-attributed calls use that stage's measured and target iteration
    counts, so independent shape, paint, refiner, and chunk loops are not all
    multiplied by one request-wide ratio. Legacy records retain the repeated
    DiT-layer heuristic.
    """
    successful = [record for record in records if record.succeeded]
    if target_units is not None:
        covering = [
            record for record in successful if record.workload_units() >= target_units
        ]
        if covering:
            successful = covering

    estimated: dict[str, dict[str, list[int]]] = {}
    for record in successful:
        source_steps = max(1, record.num_inference_steps)
        component_stages = _component_stages(record)
        repeated_stages = _repeated_stages(record, component_stages)
        for component_name, groups in record.layerwise_layer_uses.items():
            component = estimated.setdefault(component_name, {})
            for layer_name, counts in groups.items():
                target = component.setdefault(layer_name, [0] * len(counts))
                if len(target) != len(counts):
                    continue
                stage_counts = [
                    (
                        stage_name,
                        stage_components[component_name][layer_name],
                    )
                    for stage_name, stage_components in (
                        record.layerwise_layer_uses_by_stage.items()
                    )
                    if component_name in stage_components
                    and layer_name in stage_components[component_name]
                    and len(stage_components[component_name][layer_name]) == len(counts)
                ]
                for layer_index, count in enumerate(counts):
                    if stage_counts:
                        measured_by_stage = sum(
                            per_layer_counts[layer_index]
                            for _, per_layer_counts in stage_counts
                        )
                        scaled = max(0, count - measured_by_stage)
                        for stage_name, per_layer_counts in stage_counts:
                            measured_iterations, target_iterations = _stage_iterations(
                                record,
                                stage_name,
                                repeated_stages=repeated_stages,
                                target_num_inference_steps=(target_num_inference_steps),
                            )
                            stage_count = per_layer_counts[layer_index]
                            if stage_count <= 1:
                                scaled += stage_count
                            else:
                                scaled += (
                                    stage_count * target_iterations
                                    + measured_iterations
                                    - 1
                                ) // measured_iterations
                    else:
                        scaled = count
                        component_is_repeated = is_dit_component_name(
                            component_name
                        ) or any(
                            stage_name in repeated_stages
                            for stage_name in component_stages.get(component_name, ())
                        )
                        if (
                            component_is_repeated
                            and count > 1
                            and target_num_inference_steps > source_steps
                        ):
                            scaled = (
                                count * target_num_inference_steps + source_steps - 1
                            ) // source_steps
                    target[layer_index] = max(target[layer_index], scaled)
    return {
        component_name: {
            layer_name: tuple(counts) for layer_name, counts in groups.items()
        }
        for component_name, groups in estimated.items()
    }


def _component_stages(
    record: WarmupMemoryRecord,
    *,
    timed_stage_names: set[str] | None = None,
) -> dict[str, set[str]]:
    component_stages: dict[str, set[str]] = {}
    phase_components = record.phase_used_components or record.phase_active_components
    for phase_name, components in phase_components.items():
        fields = phase_name.split(":", 2)
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        stage_name = fields[1]
        if timed_stage_names is not None and stage_name not in timed_stage_names:
            continue
        for component_name in components:
            component_stages.setdefault(component_name, set()).add(stage_name)
    return component_stages


def _repeated_stages(
    record: WarmupMemoryRecord,
    component_stages: Mapping[str, set[str]],
) -> set[str]:
    stages = {
        stage_name
        for component_name, stage_names in component_stages.items()
        if is_dit_component_name(component_name)
        for stage_name in stage_names
    }
    stages.update(record.stage_iterations)
    stages.update(
        stage_name
        for stage_name in set(record.stage_duration_ms).union(
            *(stage_names for stage_names in component_stages.values())
        )
        if stage_name.endswith("DenoisingStage")
        and not stage_name.endswith("BeforeDenoisingStage")
    )
    return stages


def _stage_iterations(
    record: WarmupMemoryRecord,
    stage_name: str,
    *,
    repeated_stages: set[str],
    target_num_inference_steps: int,
) -> tuple[int, int]:
    explicit = record.stage_iterations.get(stage_name)
    if explicit is not None:
        return max(1, explicit[0]), max(1, explicit[1])
    measured = max(1, record.num_inference_steps)
    target = (
        max(1, target_num_inference_steps)
        if stage_name in repeated_stages
        else measured
    )
    return measured, target


def estimate_default_workload_timing(
    *,
    records: Iterable[WarmupMemoryRecord],
    target_units: int | None,
    target_num_inference_steps: int,
) -> tuple[int, dict[str, int], dict[str, tuple[str, ...]]]:
    """Estimate full-request and stage durations from the warmup workload.

    Repeated stages scale by their own measured/default iteration counts. The
    full-shape probe intentionally executes only a few steps, so using its raw
    total would make every one-shot encoder transfer look important relative
    to a long video request.
    """
    successful = [record for record in records if record.succeeded]
    if not successful:
        return 0, {}, {}
    if target_units is not None:
        at_target = [
            record for record in successful if record.workload_units() >= target_units
        ]
        if at_target:
            successful = at_target
    representative = max(
        successful,
        key=lambda record: (
            record.workload_units(),
            record.total_duration_ms,
        ),
    )
    if representative.total_duration_ms <= 0 or not representative.stage_duration_ms:
        return 0, {}, {}

    component_stages = _component_stages(
        representative,
        timed_stage_names=set(representative.stage_duration_ms),
    )
    repeated_stages = _repeated_stages(representative, component_stages)

    stage_duration_ns: dict[str, int] = {}
    for stage_name, duration_ms in representative.stage_duration_ms.items():
        measured_iterations, target_iterations = _stage_iterations(
            representative,
            stage_name,
            repeated_stages=repeated_stages,
            target_num_inference_steps=target_num_inference_steps,
        )
        step_durations = representative.step_duration_ms_by_stage.get(stage_name, ())
        if not step_durations and len(repeated_stages) == 1:
            step_durations = representative.step_duration_ms
        if stage_name in repeated_stages and step_durations:
            steady_steps = (
                step_durations[1:] if len(step_durations) > 1 else step_durations
            )
            non_step_ms = max(0.0, duration_ms - sum(step_durations))
            target_duration_ms = non_step_ms + (
                statistics.median(steady_steps) * target_iterations
            )
        else:
            target_duration_ms = duration_ms * target_iterations / measured_iterations
        stage_duration_ns[stage_name] = max(0, int(target_duration_ms * 1_000_000))

    measured_stage_ms = sum(representative.stage_duration_ms.values())
    untracked_ms = max(0.0, representative.total_duration_ms - measured_stage_ms)
    request_duration_ns = sum(stage_duration_ns.values()) + int(
        untracked_ms * 1_000_000
    )

    return (
        request_duration_ns,
        stage_duration_ns,
        {
            component_name: tuple(sorted(stage_names))
            for component_name, stage_names in component_stages.items()
        },
    )


def estimate_candidate_latency_savings_ns(
    *,
    candidates: Iterable[ResidencyTarget],
    request_duration_ns: int,
) -> dict[str, int]:
    """Estimate each complete placement relative to the measured one.

    An async prefetch can consume copy-engine and memory bandwidth outside the
    component's own stage, so that stage is not a sound cap. The complete
    request duration bounds removable latency. Added transfer time is not
    capped: an unmeasured mechanism may be slower than the whole request.
    """
    candidates = list(candidates)
    candidates_by_component: dict[str, list[ResidencyTarget]] = {}
    for candidate in candidates:
        candidates_by_component.setdefault(candidate.component_name, []).append(
            candidate
        )

    estimates: dict[str, int] = {}
    for component_candidates in candidates_by_component.values():
        current = next(
            (
                candidate
                for candidate in component_candidates
                if candidate.current_placement
            ),
            None,
        )
        if current is None:
            continue

        for candidate in component_candidates:
            relative_savings_ns = int(
                (candidate.h2d_bytes_per_request - current.h2d_bytes_per_request)
                / ESTIMATED_PINNED_H2D_BYTES_PER_SECOND
                * 1_000_000_000
            )
            # The request duration bounds removable latency, but not latency an
            # unmeasured placement can add. Capping both directions can make a
            # repeatedly streamed DiT appear neutral and let an unrelated small
            # gain select a several-times-slower combined placement.
            if relative_savings_ns > 0 and request_duration_ns > 0:
                relative_savings_ns = min(relative_savings_ns, request_duration_ns)
            estimates[candidate.option_key()] = relative_savings_ns

    for candidate in candidates:
        if candidate.option_key() in estimates:
            continue
        transfer_ns = int(
            candidate.h2d_bytes_per_request
            / ESTIMATED_PINNED_H2D_BYTES_PER_SECOND
            * 1_000_000_000
        )
        estimates[candidate.option_key()] = (
            min(transfer_ns, request_duration_ns)
            if transfer_ns > 0 and request_duration_ns > 0
            else transfer_ns
        )
    return estimates


def estimate_default_workload_peak_bytes(
    *,
    records: Iterable[WarmupMemoryRecord],
    target_units: int | None,
    constant_weight_bytes: int = 0,
) -> int | None:
    """Extrapolate live warmup memory to the default workload.

    Real measurements use allocated bytes. Cached allocator blocks are
    reclaimable and are covered by the explicit VRAM reserve; treating them as
    live memory would charge the same storage again when adding resident
    weights.

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
       baseline (conservative; may block adjustment but never over-allocates).

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
        peak = record.peak_allocated_bytes
        peak_by_units[units] = max(peak_by_units.get(units, 0), peak)

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
        peak = record.peak_allocated_bytes
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
) -> tuple[
    dict[str, int],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    """Estimate each measured execution phase at the target workload.

    A component already active in a phase is part of that phase's measured
    peak. Keeping a component resident therefore adds no weight bytes to its
    own component-offload phase, while it adds the full footprint to phases
    where the component was absent. Measurements with different active layouts
    remain separate constraints; combining one layout's peak with another
    layout's component set would describe a state that never occurred.
    """
    successful = [record for record in records if record.succeeded]
    if target_units is not None:
        covering = [
            record for record in successful if record.workload_units() >= target_units
        ]
        if covering:
            successful = covering

    grouped: dict[
        tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]],
        list[WarmupMemoryRecord],
    ] = {}
    for record in successful:
        used_by_phase = record.phase_used_components or record.phase_active_components
        for phase_name in record.phase_peak_allocated_bytes:
            active = tuple(sorted(record.phase_active_components.get(phase_name, ())))
            used = tuple(sorted(used_by_phase.get(phase_name, ())))
            full_weight_transitions = tuple(
                sorted(
                    record.phase_full_weight_transition_components.get(phase_name, ())
                )
            )
            grouped.setdefault(
                (phase_name, active, used, full_weight_transitions), []
            ).append(record)

    layouts_per_phase: dict[str, int] = {}
    for phase_name, _, _, _ in grouped:
        layouts_per_phase[phase_name] = layouts_per_phase.get(phase_name, 0) + 1

    estimated_peaks: dict[str, int] = {}
    active_components: dict[str, tuple[str, ...]] = {}
    used_components: dict[str, tuple[str, ...]] = {}
    full_weight_transition_components: dict[str, tuple[str, ...]] = {}
    layout_indices: dict[str, int] = {}
    for (
        phase_name,
        active,
        used,
        full_weight_transitions,
    ), phase_records in sorted(grouped.items()):
        output_name = phase_name
        if layouts_per_phase[phase_name] > 1:
            index = layout_indices.get(phase_name, 0)
            layout_indices[phase_name] = index + 1
            output_name = f"{phase_name}:layout:{index}"
        weight_floor = sum(component_weight_bytes.get(name, 0) for name in active)
        phase_measurements = [
            WarmupMemoryRecord(
                width=record.width,
                height=record.height,
                num_frames=record.num_frames,
                baseline_allocated_bytes=min(
                    record.baseline_allocated_bytes,
                    record.phase_peak_allocated_bytes[phase_name],
                ),
                peak_allocated_bytes=record.phase_peak_allocated_bytes[phase_name],
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
        estimated_peaks[output_name] = estimate
        active_components[output_name] = active
        used_components[output_name] = used
        full_weight_transition_components[output_name] = full_weight_transitions
    return (
        estimated_peaks,
        active_components,
        used_components,
        full_weight_transition_components,
    )


def _module_weight_bytes(module: nn.Module) -> int:
    """Full weight+buffer footprint, reading through layerwise CPU buffers."""
    return tensor_storage_bytes(
        chain(
            (tensor for _, tensor in iter_materialized_weights(module)),
            module.buffers(),
        )
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


def component_current_device_weight_bytes(
    modules: Mapping[str, object],
) -> dict[str, int]:
    """Physical component storage currently resident on the local accelerator."""
    result: dict[str, int] = {}
    for name, module in modules.items():
        if not isinstance(module, nn.Module):
            continue
        result[name] = tensor_storage_bytes(
            tensor
            for tensor in chain(module.parameters(), module.buffers())
            if current_platform.is_device_type(tensor.device.type)
        )
    return result


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
    """GPU weight bytes used by a resident version of this component."""
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
    layer_uses: tuple[tuple[int, ...], ...] | None = None,
) -> int:
    """Relative transfer work for one request under a layerwise placement.

    Pageable copies are weighted 2x: the CUDA driver stages them through an
    internal pinned buffer and the copy cannot run ahead of compute. This is a
    conservative ordering metric rather than a latency prediction.
    """
    resolved_uses = _resolve_layerwise_uses(
        managers=managers,
        layer_uses=layer_uses,
        fallback_uses=uses_per_streamed_layer,
    )
    total = 0
    for manager, resident_count, pinned_indices, manager_uses in zip(
        managers, resident_layers, pinned_layers, resolved_uses
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
            observed_uses = manager_uses[layer_idx]
            uses = observed_uses if layer_idx in streamed else min(1, observed_uses)
            transfer_multiplier = 1 if layer_idx in pinned else 2
            total += uses * transfer_multiplier * weight_bytes
    return total


def _resolve_layerwise_uses(
    *,
    managers: Sequence,
    layer_uses: tuple[tuple[int, ...], ...] | None,
    fallback_uses: int,
) -> tuple[tuple[int, ...], ...]:
    if layer_uses is None:
        return tuple(
            tuple(max(0, fallback_uses) for _ in range(manager.num_layers))
            for manager in managers
        )
    if len(layer_uses) != len(managers):
        raise ValueError("layerwise usage group count changed")
    resolved = []
    for manager, counts in zip(managers, layer_uses):
        if len(counts) != manager.num_layers:
            raise ValueError("layerwise usage layer count changed")
        resolved.append(tuple(max(0, int(count)) for count in counts))
    return tuple(resolved)


def _layer_uses_for_managers(
    *,
    managers: Sequence,
    uses_by_layer_name: Mapping[str, tuple[int, ...]] | None,
    fallback_uses: int,
) -> tuple[tuple[int, ...], ...] | None:
    if uses_by_layer_name is None:
        return None
    return tuple(
        uses_by_layer_name.get(
            manager.layers_attr_str,
            tuple(max(0, fallback_uses) for _ in range(manager.num_layers)),
        )
        for manager in managers
    )


def _layerwise_active_peak_device_bytes(
    *,
    managers: Sequence,
    resident_layers: tuple[int, ...],
    layer_uses: tuple[tuple[int, ...], ...] | None,
) -> int:
    if layer_uses is None:
        return sum(
            manager.peak_managed_device_weight_bytes(count)
            for manager, count in zip(managers, resident_layers)
        )
    return sum(
        manager.peak_managed_device_weight_bytes(count)
        for manager, count, uses in zip(managers, resident_layers, layer_uses)
        if any(uses)
    )


def _layerwise_pin_targets(
    *,
    managers: Sequence,
    resident_layers: tuple[int, ...],
    current_pinned_layers: tuple[tuple[int, ...], ...],
    uses_per_streamed_layer: int,
    layer_uses: tuple[tuple[int, ...], ...] | None = None,
    constrain_host_transitions: bool = True,
    maximum_utility_only: bool = False,
) -> list[tuple[tuple[int, ...], ...]]:
    """Pareto-optimal HostPin targets for one resident-layer placement.

    Layers with identical transfer value, host cost and current pin state are
    interchangeable, so they are grouped before subset construction. This
    keeps repeated transformer blocks linear while retaining non-prefix
    packings needed when layer sizes differ.
    """
    resolved_uses = _resolve_layerwise_uses(
        managers=managers,
        layer_uses=layer_uses,
        fallback_uses=uses_per_streamed_layer,
    )
    if maximum_utility_only:
        target = []
        for manager, resident_count, manager_uses in zip(
            managers, resident_layers, resolved_uses
        ):
            if not manager.pin_cpu_memory:
                target.append(())
                continue
            streamed = set(
                compute_streamed_layers(
                    num_layers=manager.num_layers,
                    resident_layers=resident_count,
                    policy=manager.residency_policy,
                )
            )
            transfer_bytes = manager.layer_weight_bytes()
            target.append(
                tuple(
                    layer_idx
                    for layer_idx in manager.pinnable_layer_indices()
                    if (
                        manager_uses[layer_idx]
                        if layer_idx in streamed
                        else min(1, manager_uses[layer_idx])
                    )
                    * transfer_bytes.get(layer_idx, 0)
                    > 0
                )
            )
        return [tuple(target)]

    current = [set(indices) for indices in current_pinned_layers]
    current_bytes = 0
    grouped: dict[tuple[int, int, int, bool], list[tuple[int, int]]] = {}
    for manager_index, (manager, resident_count, manager_uses) in enumerate(
        zip(managers, resident_layers, resolved_uses)
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
        transfer_bytes = manager.layer_weight_bytes()
        host_bytes = manager.layer_host_store_bytes()
        for layer_idx in manager.pinnable_layer_indices():
            observed_uses = manager_uses[layer_idx]
            uses = observed_uses if layer_idx in streamed else min(1, observed_uses)
            layer_host_bytes = host_bytes.get(layer_idx, 0)
            is_current = layer_idx in current[manager_index]
            if is_current:
                current_bytes += layer_host_bytes
            grouped.setdefault(
                (
                    uses,
                    transfer_bytes.get(layer_idx, 0),
                    layer_host_bytes,
                    is_current,
                ),
                [],
            ).append((manager_index, layer_idx))

    # target HostPin bytes, avoided pageable-transfer work, unpin scratch,
    # pin scratch, selected layer indices.
    states = [(0, 0, current_bytes, 0, tuple(() for _ in managers))]

    def dominates(left, right) -> bool:
        return (
            left[0] <= right[0]
            and left[1] >= right[1]
            and left[2] <= right[2]
            and left[3] <= right[3]
        )

    def prune(candidates):
        if not constrain_host_transitions:
            # host transition scratch cannot constrain this solve, so the local
            # frontier is exactly pinned bytes versus avoided pageable transfer
            best_by_pinned_bytes = {}
            for state in candidates:
                incumbent = best_by_pinned_bytes.get(state[0])
                if incumbent is None or (
                    -state[1],
                    state[2] + state[3],
                    state[2],
                    state[3],
                    state[4],
                ) < (
                    -incumbent[1],
                    incumbent[2] + incumbent[3],
                    incumbent[2],
                    incumbent[3],
                    incumbent[4],
                ):
                    best_by_pinned_bytes[state[0]] = state
            frontier = []
            best_utility = -1
            for state in sorted(
                best_by_pinned_bytes.values(),
                key=lambda item: (item[0], -item[1], item[4]),
            ):
                if state[1] <= best_utility:
                    continue
                frontier.append(state)
                best_utility = state[1]
            return frontier

        best_by_cost = {}
        for state in candidates:
            key = state[:4]
            incumbent = best_by_cost.get(key)
            if incumbent is None or state[4] < incumbent[4]:
                best_by_cost[key] = state
        frontier = []
        for state in sorted(
            best_by_cost.values(),
            key=lambda item: (item[0], item[2], item[3], -item[1], item[4]),
        ):
            if any(dominates(existing, state) for existing in frontier):
                continue
            frontier = [
                existing for existing in frontier if not dominates(state, existing)
            ]
            frontier.append(state)
        return frontier

    for (uses, transfer_bytes, host_bytes, is_current), layers in sorted(
        grouped.items(),
        key=lambda item: (-item[0][0], -item[0][1], item[0][2], item[0][3]),
    ):
        layers.sort()
        choices = []
        selected = [set() for _ in managers]
        for count in range(len(layers) + 1):
            if count:
                manager_index, layer_idx = layers[count - 1]
                selected[manager_index].add(layer_idx)
            choices.append(
                (
                    count * host_bytes,
                    count * uses * transfer_bytes,
                    -count * host_bytes if is_current else 0,
                    0 if is_current else count * host_bytes,
                    tuple(tuple(sorted(indices)) for indices in selected),
                )
            )

        expanded = []
        for state in states:
            for choice in choices:
                expanded.append(
                    (
                        state[0] + choice[0],
                        state[1] + choice[1],
                        state[2] + choice[2],
                        state[3] + choice[3],
                        tuple(
                            tuple(sorted(set(left) | set(right)))
                            for left, right in zip(state[4], choice[4])
                        ),
                    )
                )
        states = prune(expanded)

    return [
        state[4]
        for state in sorted(
            states,
            key=lambda item: (item[0], item[2], item[3], -item[1], item[4]),
        )
    ]


def _layerwise_resident_targets(
    managers: Sequence,
    *,
    layer_uses: tuple[tuple[int, ...], ...] | None = None,
) -> list[tuple[int, ...]]:
    """Useful resident-count tuples for the measured request.

    With manager-level measurements, each repeatedly executed layer group is
    independent. This admits states such as ``(encoder=0, decoder=k)`` or
    separate double/single-block DiT allocations. A group used at most once
    cannot avoid any per-request H2D bytes through stage residency: its layers
    are still loaded once when the component activates. Keep it streamed and
    let HostPin or whole-component residency represent its useful choices.

    Without measurements, retain every state expressible by the public
    component-level CLI knob. This fallback is used by static/unit callers and
    avoids inventing per-group workload semantics before calibration.

    ``--layerwise-resident-layers`` accepts either one absolute count or one
    ratio for a component. Those are different paths when a component owns
    layer stacks of unequal length: for stacks of 4 and 8 layers, an absolute
    value of 2 gives ``(2, 2)``, while a ratio of 0.25 gives ``(1, 2)``.
    Enumerate every interval where the rounded ratio result is constant, plus
    every absolute count, so auto placement does not silently omit a state the
    explicit interface can represent.
    """
    layer_counts = tuple(manager.num_layers for manager in managers)
    if layer_uses is not None:
        resolved_uses = _resolve_layerwise_uses(
            managers=managers,
            layer_uses=layer_uses,
            fallback_uses=0,
        )
        choices = [
            range(num_layers + 1) if any(count > 1 for count in uses) else (0,)
            for num_layers, uses in zip(layer_counts, resolved_uses)
        ]
        return list(product(*choices))

    targets = {tuple(0 for _ in managers)}

    for count in range(1, max(layer_counts) + 1):
        targets.add(tuple(min(count, num_layers) for num_layers in layer_counts))

    boundaries = {0.0, 1.0}
    for num_layers in layer_counts:
        boundaries.update(
            (count + 0.5) / num_layers
            for count in range(num_layers)
            if count + 0.5 < num_layers
        )
    ordered_boundaries = sorted(boundaries)
    for lower, upper in zip(ordered_boundaries, ordered_boundaries[1:]):
        ratio = (lower + upper) / 2
        targets.add(
            tuple(max(1, int(round(ratio * num_layers))) for num_layers in layer_counts)
        )

    return sorted(targets)


class _EstimatedLayerwiseManager:
    """Size-only layer group used before a real offload manager exists."""

    def __init__(
        self,
        *,
        layers_attr_str: str,
        layer_bytes: dict[int, int],
        prefetch_size: int,
        residency_policy: str,
        pin_cpu_memory: bool,
    ) -> None:
        self.layers_attr_str = layers_attr_str
        self._layer_bytes = layer_bytes
        self.num_layers = len(layer_bytes)
        self.prefetch_size = min(max(1, prefetch_size), self.num_layers)
        self.residency_policy = residency_policy
        self.pin_cpu_memory = pin_cpu_memory
        self.resident_layers = 0

    def layer_weight_bytes(self) -> dict[int, int]:
        return self._layer_bytes

    def layer_host_store_bytes(self) -> dict[int, int]:
        return self._layer_bytes

    def pinned_host_weight_bytes(self) -> int:
        return 0

    def pinned_layer_indices(self) -> tuple[int, ...]:
        return ()

    def pinnable_layer_indices(self) -> tuple[int, ...]:
        if not self.pin_cpu_memory:
            return ()
        return tuple(self._layer_bytes)

    def resident_weight_bytes(self, resident_layers: int | None = None) -> int:
        count = self.resident_layers if resident_layers is None else resident_layers
        streamed = set(
            compute_streamed_layers(
                num_layers=self.num_layers,
                resident_layers=count,
                policy=self.residency_policy,
            )
        )
        return sum(
            weight_bytes
            for layer_index, weight_bytes in self._layer_bytes.items()
            if layer_index not in streamed
        )

    def peak_managed_device_weight_bytes(
        self, resident_layers: int | None = None
    ) -> int:
        count = self.resident_layers if resident_layers is None else resident_layers
        streamed = compute_streamed_layers(
            num_layers=self.num_layers,
            resident_layers=count,
            policy=self.residency_policy,
        )
        resident_bytes = self.resident_weight_bytes(count)
        copy_window = min(self.prefetch_size, len(streamed))
        streamed_window_bytes = sum(
            sorted(
                (self._layer_bytes.get(index, 0) for index in streamed),
                reverse=True,
            )[:copy_window]
        )
        return resident_bytes + streamed_window_bytes


def _estimate_unconfigured_layerwise_managers(
    *,
    module: LayerwiseOffloadableModuleMixin,
    prefetch_value: float,
    residency_policy: str,
    pin_cpu_memory: bool,
) -> list[_EstimatedLayerwiseManager]:
    """Read an unloaded module's block sizes without allocating CPU stores."""
    named_modules = dict(module.named_modules())
    estimates = []
    for layer_name in module.layer_names:
        layers = named_modules.get(layer_name)
        if not isinstance(layers, (nn.ModuleList, nn.Sequential)) or not layers:
            continue
        num_layers = len(layers)
        if prefetch_value < 1.0:
            prefetch_size = 1 + int(round(prefetch_value * (num_layers - 1)))
        else:
            prefetch_size = int(prefetch_value)
        estimates.append(
            _EstimatedLayerwiseManager(
                layers_attr_str=layer_name,
                layer_bytes=estimate_layer_weight_bytes(layers),
                prefetch_size=prefetch_size,
                residency_policy=residency_policy,
                pin_cpu_memory=pin_cpu_memory,
            )
        )
    return estimates


def _unconfigured_layerwise_targets(
    *,
    component_name: str,
    module: LayerwiseOffloadableModuleMixin,
    current_mode: str,
    full_weight_bytes: int,
    num_inference_steps: int,
    prefetch_value: float,
    residency_policy: str,
    pin_cpu_memory: bool,
    component_used: bool,
    layer_uses_by_name: Mapping[str, tuple[int, ...]] | None,
) -> tuple[list[ResidencyTarget], int]:
    """Virtual layerwise frontier for a component still using coarse offload.

    The first selected virtual state configures real managers lazily. It starts
    pageable and is remeasured immediately; the next fixed-point round then
    exposes the exact HostPin frontier from the realized stores.
    """
    managers = _estimate_unconfigured_layerwise_managers(
        module=module,
        prefetch_value=prefetch_value,
        residency_policy=residency_policy,
        pin_cpu_memory=pin_cpu_memory,
    )
    if not managers:
        return [], 0
    managed_weight_bytes = sum(
        sum(manager.layer_weight_bytes().values()) for manager in managers
    )
    if managed_weight_bytes <= 0:
        return [], 0
    unmanaged_weight_bytes = max(0, full_weight_bytes - managed_weight_bytes)
    uses_per_request = (
        max(1, num_inference_steps)
        if component_used and is_dit_component_name(component_name)
        else int(component_used)
    )
    layer_uses = _layer_uses_for_managers(
        managers=managers,
        uses_by_layer_name=layer_uses_by_name,
        fallback_uses=uses_per_request,
    )
    empty_pins = tuple(() for _ in managers)
    maximum_transfer_work = _layerwise_transfer_work_bytes(
        managers=managers,
        resident_layers=tuple(0 for _ in managers),
        pinned_layers=empty_pins,
        uses_per_streamed_layer=uses_per_request,
        layer_uses=layer_uses,
    )
    coarse_transfer_work = 2 * full_weight_bytes if component_used else 0
    coarse_savings = max(0, maximum_transfer_work - coarse_transfer_work)
    current_resident = current_mode == RESIDENT
    current_inactive_bytes = full_weight_bytes if current_resident else 0
    host_materialize_scratch = max(
        (
            weight_bytes
            for manager in managers
            for weight_bytes in manager.layer_host_store_bytes().values()
        ),
        default=0,
    )
    targets = []
    for resident_layers in _layerwise_resident_targets(managers, layer_uses=layer_uses):
        resident_bytes = sum(
            manager.resident_weight_bytes(count)
            for manager, count in zip(managers, resident_layers)
        )
        active_managed_bytes = _layerwise_active_peak_device_bytes(
            managers=managers,
            resident_layers=resident_layers,
            layer_uses=layer_uses,
        )
        transfer_work = _layerwise_transfer_work_bytes(
            managers=managers,
            resident_layers=resident_layers,
            pinned_layers=empty_pins,
            uses_per_streamed_layer=uses_per_request,
            layer_uses=layer_uses,
        )
        targets.append(
            ResidencyTarget(
                component_name=component_name,
                residency_mode=COMPONENT_OFFLOAD,
                target_residency_mode=LAYERWISE_OFFLOAD,
                target_resident_weight_bytes=resident_bytes,
                # This strategy has not run yet, so transfer bytes cannot
                # establish that it beats the calibrated coarse path. It may
                # tie that path as a memory alternative; only a subsequent
                # calibration may justify further layerwise tuning.
                h2d_bytes_per_request=min(
                    coarse_savings,
                    max(0, maximum_transfer_work - transfer_work),
                ),
                target_layerwise_resident_layers=resident_layers,
                target_layerwise_pinned_layers=empty_pins,
                host_materialize_scratch_bytes=host_materialize_scratch,
                device_transition_delta_bytes=(
                    unmanaged_weight_bytes - current_inactive_bytes
                ),
                active_device_delta_bytes=(
                    unmanaged_weight_bytes + active_managed_bytes - full_weight_bytes
                ),
                present_device_delta_bytes=(
                    unmanaged_weight_bytes + active_managed_bytes - full_weight_bytes
                ),
                inactive_device_delta_bytes=(
                    unmanaged_weight_bytes - current_inactive_bytes
                ),
                target_device_weight_bytes=unmanaged_weight_bytes + resident_bytes,
            )
        )
    return targets, maximum_transfer_work


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


def collect_residency_targets(
    *,
    modules: Mapping[str, object],
    residency_mode_of: Callable[[str], str],
    baseline_residency_mode_of: Callable[[str], str] | None = None,
    explicit_residency_mode_of: Callable[[str], str | None],
    custom_strategy_names: Iterable[str],
    num_inference_steps: int,
    allow_host_pin_reallocation: bool = True,
    mixed_dtype_components: Iterable[str] = (),
    auto_resident_components: Iterable[str] = (),
    layerwise_tuning_of: Callable[[str, bool], tuple[float, float, str]] | None = None,
    pin_cpu_memory: bool = True,
    used_components: Iterable[str] | None = None,
    layerwise_layer_uses: Mapping[str, Mapping[str, tuple[int, ...]]] | None = None,
    host_transition_headroom_bytes: int | None = None,
    host_pin_headroom_bytes: int | None = None,
    request_duration_ns: int = 0,
) -> list[ResidencyTarget]:
    """Build complete target-state frontiers for auto-managed components.

    Every option is expressed relative to the currently measured placement,
    but its transfer utility is absolute within the component's frontier.
    Including the current and lower-memory states lets later calibration rounds
    replace an earlier choice instead of being limited to monotonic upgrades.

    When both host constraints provably cannot bind and the request-duration
    cap cannot flatten transfer utility, every pin subset except the unique
    maximum-utility one is dominated under the solver's latency-first ordering.
    Only that exact condition permits collapsing the HostPin frontier; otherwise
    every Pareto-relevant subset remains available to the joint optimizer.
    """
    custom_names = set(custom_strategy_names)
    mixed_dtype_names = set(mixed_dtype_components)
    auto_resident_names = set(auto_resident_components)
    measured_used_names = set(used_components) if used_components is not None else None
    if baseline_residency_mode_of is None:
        baseline_residency_mode_of = residency_mode_of
    constrain_host_transitions = True
    constrain_host_pin = True
    if (
        host_transition_headroom_bytes is not None
        or host_pin_headroom_bytes is not None
    ):
        seen_managers: set[int] = set()
        maximum_unpin_bytes = 0
        maximum_pin_bytes = 0
        for module in modules.values():
            if not isinstance(module, LayerwiseOffloadableModuleMixin):
                continue
            for manager in module.layerwise_offload_managers:
                manager_id = id(manager)
                if manager_id in seen_managers:
                    continue
                seen_managers.add(manager_id)
                pinned = set(manager.pinned_layer_indices())
                host_bytes = manager.layer_host_store_bytes()
                maximum_unpin_bytes += sum(
                    host_bytes.get(layer_idx, 0) for layer_idx in pinned
                )
                maximum_pin_bytes += sum(
                    host_bytes.get(layer_idx, 0)
                    for layer_idx in manager.pinnable_layer_indices()
                    if layer_idx not in pinned
                )
        if host_transition_headroom_bytes is not None:
            constrain_host_transitions = max(
                maximum_unpin_bytes, maximum_pin_bytes
            ) > max(0, host_transition_headroom_bytes)
        if host_pin_headroom_bytes is not None:
            constrain_host_pin = maximum_pin_bytes > max(0, host_pin_headroom_bytes)
    candidates = []
    for name in sorted(modules):
        module = modules[name]
        if not isinstance(module, nn.Module):
            continue
        if name in custom_names:
            continue
        baseline_mode = baseline_residency_mode_of(name)
        if baseline_mode not in (COMPONENT_OFFLOAD, LAYERWISE_OFFLOAD):
            continue
        if explicit_residency_mode_of(name) is not None:
            continue
        if is_fsdp_managed_module(module):
            continue
        component_layer_uses = (
            layerwise_layer_uses.get(name) if layerwise_layer_uses is not None else None
        )
        measured_layer_use = bool(
            component_layer_uses
            and any(
                count for counts in component_layer_uses.values() for count in counts
            )
        )
        current_mode = residency_mode_of(name)
        component_used = (
            measured_used_names is None
            or name in measured_used_names
            or measured_layer_use
        )
        if current_mode == RESIDENT and name not in auto_resident_names:
            # A loader or another runtime feature owns this hard requirement.
            continue

        has_layerwise_managers = isinstance(
            module, LayerwiseOffloadableModuleMixin
        ) and bool(module.layerwise_offload_managers)
        if has_layerwise_managers and current_mode == COMPONENT_OFFLOAD:
            # Temporary managers used by offload-during-compile do not own the
            # serving placement. Only an effective layerwise mode proves that
            # these managers form the current serving frontier.
            continue
        frontier_mode = LAYERWISE_OFFLOAD if has_layerwise_managers else baseline_mode

        if frontier_mode == COMPONENT_OFFLOAD:
            if current_mode not in (COMPONENT_OFFLOAD, RESIDENT):
                continue
            if is_layerwise_offloaded_module(module):
                # The module is layerwise-managed despite its configured mode
                # (e.g. the offload_during_compile window). Moving it behind
                # the manager would strand its bookkeeping.
                continue
            weight_bytes = _module_weight_bytes(module)
            if weight_bytes <= 0:
                continue
            current_resident = current_mode == RESIDENT
            virtual_targets: list[ResidencyTarget] = []
            maximum_transfer_work = 0
            if (
                isinstance(module, LayerwiseOffloadableModuleMixin)
                and layerwise_tuning_of is not None
            ):
                prefetch_value, _, residency_policy = layerwise_tuning_of(
                    name, module.layerwise_offload_dit_group_enabled
                )
                virtual_targets, maximum_transfer_work = (
                    _unconfigured_layerwise_targets(
                        component_name=name,
                        module=module,
                        current_mode=current_mode,
                        full_weight_bytes=weight_bytes,
                        num_inference_steps=num_inference_steps,
                        prefetch_value=prefetch_value,
                        residency_policy=residency_policy,
                        pin_cpu_memory=pin_cpu_memory,
                        component_used=component_used,
                        layer_uses_by_name=component_layer_uses,
                    )
                )
            uses_per_request = (
                max(1, num_inference_steps)
                if component_used and is_dit_component_name(name)
                else int(component_used)
            )
            component_transfer_work = 2 * weight_bytes if component_used else 0
            candidates.extend(
                [
                    ResidencyTarget(
                        component_name=name,
                        residency_mode=baseline_mode,
                        target_residency_mode=COMPONENT_OFFLOAD,
                        target_resident_weight_bytes=0,
                        h2d_bytes_per_request=max(
                            0, maximum_transfer_work - component_transfer_work
                        ),
                        host_materialize_scratch_bytes=(
                            weight_bytes if current_resident else 0
                        ),
                        device_transition_delta_bytes=(
                            -weight_bytes if current_resident else 0
                        ),
                        active_device_delta_bytes=0,
                        inactive_device_delta_bytes=(
                            -weight_bytes if current_resident else 0
                        ),
                        present_device_delta_bytes=(
                            -weight_bytes if current_resident else 0
                        ),
                        current_placement=not current_resident,
                    ),
                    ResidencyTarget(
                        component_name=name,
                        residency_mode=baseline_mode,
                        target_residency_mode=RESIDENT,
                        target_resident_weight_bytes=weight_bytes,
                        h2d_bytes_per_request=(maximum_transfer_work or weight_bytes),
                        permanent_residency=True,
                        device_transition_delta_bytes=0,
                        active_device_delta_bytes=0,
                        inactive_device_delta_bytes=(
                            0 if current_resident else weight_bytes
                        ),
                        present_device_delta_bytes=0,
                        current_placement=current_resident,
                        target_device_weight_bytes=weight_bytes,
                    ),
                ]
            )
            candidates.extend(virtual_targets)
            continue

        if not isinstance(module, LayerwiseOffloadableModuleMixin):
            continue
        managers = list(module.layerwise_offload_managers)
        if not managers:
            continue
        enabled = {manager.enabled for manager in managers}
        if len(enabled) != 1:
            logger.warning(
                "Skipping auto residency for %s: layerwise groups disagree on "
                "whether offload is enabled",
                name,
            )
            continue
        current_permanent = not enabled.pop()
        if current_permanent != (current_mode == RESIDENT):
            # The module is layerwise-managed despite its configured mode
            # or is hard-required resident by a different owner.
            continue
        managed_weight_bytes = sum(
            sum(manager.layer_weight_bytes().values()) for manager in managers
        )
        if managed_weight_bytes <= 0:
            continue
        # A layerwise DiT re-streams its layers once per denoise forward;
        # every other offloaded component transfers once per request.
        uses_per_request = (
            max(1, num_inference_steps)
            if component_used and is_dit_component_name(name)
            else int(component_used)
        )
        layer_uses = _layer_uses_for_managers(
            managers=managers,
            uses_by_layer_name=component_layer_uses,
            fallback_uses=uses_per_request,
        )
        current_resident_layers = tuple(manager.resident_layers for manager in managers)
        current_pinned_layers = tuple(
            manager.pinned_layer_indices() for manager in managers
        )
        current_peak_device_bytes = (
            managed_weight_bytes
            if current_permanent
            else _layerwise_active_peak_device_bytes(
                managers=managers,
                resident_layers=current_resident_layers,
                layer_uses=layer_uses,
            )
        )
        current_inactive_device_bytes = managed_weight_bytes if current_permanent else 0
        current_pinned_bytes = sum(
            manager.pinned_host_weight_bytes() for manager in managers
        )
        layerwise_maximum_transfer_work = _layerwise_transfer_work_bytes(
            managers=managers,
            resident_layers=tuple(0 for _ in managers),
            pinned_layers=tuple(() for _ in managers),
            uses_per_streamed_layer=uses_per_request,
            layer_uses=layer_uses,
        )
        full_weight_bytes = _module_weight_bytes(module)
        unmanaged_weight_bytes = max(0, full_weight_bytes - managed_weight_bytes)
        component_transfer_work = 2 * full_weight_bytes if component_used else 0
        maximum_transfer_work = max(
            layerwise_maximum_transfer_work, component_transfer_work
        )
        pin_utility_cannot_saturate = (
            request_duration_ns > 0
            and int(
                maximum_transfer_work
                / ESTIMATED_PINNED_H2D_BYTES_PER_SECOND
                * 1_000_000_000
            )
            <= request_duration_ns
        )

        empty_resident_layers = tuple(0 for _ in managers)
        empty_pinned_layers = tuple(() for _ in managers)
        unpin_scratch, _ = _layerwise_host_transition_bytes(
            managers=managers,
            current_pinned_layers=current_pinned_layers,
            target_pinned_layers=empty_pinned_layers,
        )
        candidates.append(
            ResidencyTarget(
                component_name=name,
                residency_mode=frontier_mode,
                target_residency_mode=COMPONENT_OFFLOAD,
                target_resident_weight_bytes=0,
                h2d_bytes_per_request=max(
                    0, maximum_transfer_work - component_transfer_work
                ),
                target_layerwise_resident_layers=empty_resident_layers,
                target_layerwise_pinned_layers=empty_pinned_layers,
                pinned_host_delta_bytes=-current_pinned_bytes,
                host_unpin_scratch_bytes=unpin_scratch,
                device_transition_delta_bytes=(
                    0 if current_permanent else managed_weight_bytes
                ),
                active_device_delta_bytes=(
                    full_weight_bytes - current_peak_device_bytes
                ),
                present_device_delta_bytes=(
                    full_weight_bytes - current_peak_device_bytes
                ),
                inactive_device_delta_bytes=-current_inactive_device_bytes,
                target_device_weight_bytes=0,
                target_pinned_host_bytes=0,
            )
        )

        for target_resident_layers in _layerwise_resident_targets(
            managers, layer_uses=layer_uses
        ):
            target_resident_bytes = sum(
                manager.resident_weight_bytes(count)
                for manager, count in zip(managers, target_resident_layers)
            )
            target_peak_device_bytes = _layerwise_active_peak_device_bytes(
                managers=managers,
                resident_layers=target_resident_layers,
                layer_uses=layer_uses,
            )
            pin_targets = (
                _layerwise_pin_targets(
                    managers=managers,
                    resident_layers=target_resident_layers,
                    current_pinned_layers=current_pinned_layers,
                    uses_per_streamed_layer=uses_per_request,
                    layer_uses=layer_uses,
                    constrain_host_transitions=constrain_host_transitions,
                    maximum_utility_only=(
                        not constrain_host_transitions
                        and not constrain_host_pin
                        and pin_utility_cannot_saturate
                    ),
                )
                if allow_host_pin_reallocation
                else [current_pinned_layers]
            )
            if (
                target_resident_layers == current_resident_layers
                and current_pinned_layers not in pin_targets
            ):
                # relative utility is anchored to the measured placement; keep
                # that exact state when non-binding host resources allow
                # every other pin subset to collapse to its maximum-utility
                # representative
                pin_targets.append(current_pinned_layers)
            for target_pinned_layers in pin_targets:
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
                    layer_uses=layer_uses,
                )
                unpin_scratch, pin_scratch = _layerwise_host_transition_bytes(
                    managers=managers,
                    current_pinned_layers=current_pinned_layers,
                    target_pinned_layers=target_pinned_layers,
                )
                candidates.append(
                    ResidencyTarget(
                        component_name=name,
                        residency_mode=frontier_mode,
                        target_residency_mode=LAYERWISE_OFFLOAD,
                        target_resident_weight_bytes=target_resident_bytes,
                        h2d_bytes_per_request=(
                            maximum_transfer_work - target_transfer_work
                        ),
                        target_layerwise_resident_layers=target_resident_layers,
                        target_layerwise_pinned_layers=target_pinned_layers,
                        pinned_host_delta_bytes=(
                            target_pinned_bytes - current_pinned_bytes
                        ),
                        host_unpin_scratch_bytes=unpin_scratch,
                        host_pin_scratch_bytes=pin_scratch,
                        device_transition_delta_bytes=(
                            -managed_weight_bytes if current_permanent else 0
                        ),
                        active_device_delta_bytes=(
                            target_peak_device_bytes - current_peak_device_bytes
                        ),
                        inactive_device_delta_bytes=-current_inactive_device_bytes,
                        present_device_delta_bytes=-current_inactive_device_bytes,
                        current_placement=(
                            not current_permanent
                            and target_resident_layers == current_resident_layers
                            and target_pinned_layers == current_pinned_layers
                        ),
                        target_device_weight_bytes=(
                            unmanaged_weight_bytes + target_resident_bytes
                        ),
                        target_pinned_host_bytes=target_pinned_bytes,
                    )
                )

        # Layerwise stores have one fixed dtype. ResidentStrategy instead casts
        # at every declared use, so a component with mixed use dtypes cannot be
        # switched permanently without changing its numerical path. HostPin
        # repacking and stage-scoped layer residency keep the layerwise strategy
        # active and remain valid candidates.
        if name in mixed_dtype_names:
            continue

        full_resident_layers = tuple(manager.num_layers for manager in managers)
        # Fully resident layers never read their host stores, so their pins can
        # be released and reused. Every worker plans against its non-overlapping
        # HostPin and transition-headroom share before repacking concurrently.
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
                ResidencyTarget(
                    component_name=name,
                    residency_mode=frontier_mode,
                    target_residency_mode=RESIDENT,
                    target_resident_weight_bytes=managed_weight_bytes,
                    h2d_bytes_per_request=maximum_transfer_work,
                    target_layerwise_resident_layers=full_resident_layers,
                    target_layerwise_pinned_layers=permanent_pins,
                    pinned_host_delta_bytes=(
                        permanent_pinned_bytes - current_pinned_bytes
                    ),
                    host_unpin_scratch_bytes=unpin_scratch,
                    host_pin_scratch_bytes=pin_scratch,
                    device_transition_delta_bytes=(
                        0 if current_permanent else managed_weight_bytes
                    ),
                    permanent_residency=True,
                    active_device_delta_bytes=(
                        managed_weight_bytes - current_peak_device_bytes
                    ),
                    # Stage-scoped resident layers are released after the use,
                    # so the complete managed footprint is new elsewhere.
                    inactive_device_delta_bytes=(
                        managed_weight_bytes - current_inactive_device_bytes
                    ),
                    present_device_delta_bytes=(
                        managed_weight_bytes - current_inactive_device_bytes
                    ),
                    current_placement=(
                        current_permanent and permanent_pins == current_pinned_layers
                    ),
                    target_device_weight_bytes=full_weight_bytes,
                    target_pinned_host_bytes=permanent_pinned_bytes,
                )
            )
    return candidates


def rank_candidates_by_h2d_savings(
    candidates: Iterable[ResidencyTarget],
) -> list[ResidencyTarget]:
    """Biggest per-request H2D savings first; name breaks ties deterministically.

    Shared by the placement plan and the post-request residency hint so both
    use the same benefit ordering.
    """
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.h2d_bytes_per_request,
            candidate.component_name,
            candidate.option_key(),
        ),
    )


def _skip_plan(
    reason: str, *, current_placement_reserve_shortfall_bytes: int = 0
) -> AutoResidencyPlan:
    return AutoResidencyPlan(
        skip_reason=reason,
        current_placement_reserve_shortfall_bytes=(
            current_placement_reserve_shortfall_bytes
        ),
    )


def _vram_reserve_bytes(budget_bytes: int, *, target_workload_measured: bool) -> int:
    reserve_fraction = (
        MEASURED_VRAM_RESERVE_FRACTION
        if target_workload_measured
        else EXTRAPOLATED_VRAM_RESERVE_FRACTION
    )
    return max(
        int(budget_bytes * reserve_fraction),
        min(
            MIN_VRAM_RESERVE_BYTES,
            int(budget_bytes * MAX_VRAM_RESERVE_FRACTION),
        ),
    )


def current_placement_reserve_shortfall_bytes(
    reports: Iterable[RankResidencyReport],
) -> int:
    """Return the largest reserve deficit of the measured placement.

    This checks the placement that just executed without regenerating a
    candidate frontier or running the optimizer. It is therefore suitable for
    the single post-adjustment validation pass.
    """
    reports = list(reports)
    if not reports:
        return 0
    target_workload_measured = any(
        report.target_workload_measured for report in reports
    )
    shortfalls = []
    for report in reports:
        if report.estimated_peak_bytes is None:
            continue
        reserve = _vram_reserve_bytes(
            report.budget_bytes,
            target_workload_measured=target_workload_measured,
        )
        measured_peak = max(
            [
                report.estimated_peak_bytes,
                *report.estimated_peak_bytes_by_phase.values(),
            ]
        )
        # Candidate deltas are intentionally solved against live allocated
        # bytes, so reclaimable allocator cache is not charged twice. Once a
        # placement has actually run, however, its mapped allocator footprint
        # must still leave the same reserve for the next request.
        realized_footprint = max(measured_peak, report.observed_reserved_bytes)
        shortfalls.append(realized_footprint + reserve - report.budget_bytes)
    return max(0, max(shortfalls, default=0))


def _binding_phase_constraints(
    report: RankResidencyReport,
    candidate_component_names: set[str],
) -> list[
    tuple[
        str,
        int,
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]
]:
    """Keep the binding measured and conservative unobserved phases.

    Every candidate has the same device delta in phases with the same active
    component set. The lower peaks are therefore provably redundant; removing
    them reduces the exact optimizer's resource dimension without changing
    its feasible placements. A short calibration may not reach a later
    timestep-routed component (for example the second DiT in a two-stage
    denoiser), so give each unobserved candidate a synthetic phase at the
    measured request peak. This prevents permanent residents from being
    placed in memory that the later component may need. Components left on the
    device at request end also carry into the next request, so add their weight
    to cold-start phases that did not observe them yet.
    """
    phase_peaks = report.estimated_peak_bytes_by_phase
    if not phase_peaks:
        if report.estimated_peak_bytes is None:
            return []
        phase_peaks = {"request": report.estimated_peak_bytes}
    request_peak = (
        report.estimated_peak_bytes
        if report.estimated_peak_bytes is not None
        else max(phase_peaks.values())
    )
    current_device_bytes = dict(report.current_device_weight_bytes_by_component)
    for candidate in report.candidates:
        if (
            candidate.current_placement
            and candidate.component_name not in current_device_bytes
        ):
            current_device_bytes[candidate.component_name] = (
                candidate.target_device_weight_bytes
            )
    steady_state_components = {
        component_name
        for phase_name, components in report.active_components_by_phase.items()
        if phase_name == "idle" or phase_name.startswith("idle:layout:")
        for component_name in components
    }

    binding: dict[
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
        ],
        tuple[str, int],
    ] = {}
    for phase_name, phase_peak in phase_peaks.items():
        measured_components = set(report.active_components_by_phase.get(phase_name, ()))
        carried_components = steady_state_components - measured_components
        phase_peak += sum(
            current_device_bytes.get(component_name, 0)
            for component_name in carried_components
        )
        present = tuple(
            sorted(
                (measured_components | steady_state_components)
                & candidate_component_names
            )
        )
        measured = tuple(sorted(measured_components & candidate_component_names))
        used = tuple(sorted(report.used_components_by_phase.get(phase_name, ())))
        full_weight_transitions = tuple(
            sorted(
                set(
                    report.full_weight_transition_components_by_phase.get(
                        phase_name, ()
                    )
                )
                & candidate_component_names
            )
        )
        layout = (present, measured, used, full_weight_transitions)
        current = binding.get(layout)
        if current is None or (phase_peak, phase_name) > (current[1], current[0]):
            binding[layout] = (phase_name, phase_peak)
    observed_components = {
        component_name
        for (_, _, used, full_weight_transitions) in binding
        for component_name in set(used) | set(full_weight_transitions)
    }
    steady_request_peak = max(
        request_peak,
        *(phase_peak for _, phase_peak in binding.values()),
    )
    for component_name in sorted(candidate_component_names - observed_components):
        present = tuple(
            sorted(
                (steady_state_components | {component_name}) & candidate_component_names
            )
        )
        binding[(present, (), (component_name,), ())] = (
            f"unobserved:{component_name}",
            steady_request_peak,
        )
    return [
        (
            f"gpu:rank{report.rank}:{phase_name}",
            phase_peak,
            present,
            measured,
            used,
            full_weight_transitions,
        )
        for (present, measured, used, full_weight_transitions), (
            phase_name,
            phase_peak,
        ) in sorted(binding.items(), key=lambda item: item[1][0])
    ]


def _consensus_candidates(
    reports: list[RankResidencyReport],
) -> list[ResidencyTarget]:
    """Merge per-rank candidates: keep components every rank agrees on.

    Sizes are worst-cased with the per-rank maximum so an adjustment never
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
        target_modes = {
            candidate.target_residency_mode for candidate in rank_candidates
        }
        current = {candidate.current_placement for candidate in rank_candidates}
        if (
            len(component_names) != 1
            or len(targets) != 1
            or len(pinned_targets) != 1
            or len(permanent) != 1
            or len(target_modes) != 1
            or len(current) != 1
        ):
            continue
        merged.append(
            ResidencyTarget(
                component_name=component_names.pop(),
                residency_mode=modes.pop(),
                target_resident_weight_bytes=max(
                    candidate.target_resident_weight_bytes
                    for candidate in rank_candidates
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
                host_materialize_scratch_bytes=max(
                    candidate.host_materialize_scratch_bytes
                    for candidate in rank_candidates
                ),
                device_transition_delta_bytes=max(
                    candidate.device_transition_delta_bytes
                    for candidate in rank_candidates
                ),
                permanent_residency=permanent.pop(),
                active_device_delta_bytes=max(
                    candidate.active_device_delta_bytes for candidate in rank_candidates
                ),
                present_device_delta_bytes=max(
                    candidate.present_device_delta_bytes
                    for candidate in rank_candidates
                ),
                inactive_device_delta_bytes=max(
                    candidate.inactive_device_delta_bytes
                    for candidate in rank_candidates
                ),
                target_residency_mode=target_modes.pop(),
                current_placement=current.pop(),
                target_device_weight_bytes=max(
                    candidate.target_device_weight_bytes
                    for candidate in rank_candidates
                ),
                target_pinned_host_bytes=max(
                    candidate.target_pinned_host_bytes for candidate in rank_candidates
                ),
            )
        )
    return merged


def plan_auto_residency(*, reports: list[RankResidencyReport]) -> AutoResidencyPlan:
    """Turn gathered rank reports into one deterministic placement plan."""
    if not reports:
        return _skip_plan("no rank reports")
    for report in reports:
        if report.skip_reason is not None:
            return _skip_plan(f"rank {report.rank}: {report.skip_reason}")
        if report.estimated_peak_bytes is None:
            return _skip_plan(f"rank {report.rank}: no usable warmup measurement")

    estimated_peak = max(report.estimated_peak_bytes for report in reports)
    budget = min(report.budget_bytes for report in reports)
    # The warmup request is executed collectively. Some non-output ranks may
    # not retain the effective request shape in their local Req, but one rank's
    # covering measurement proves that the replica executed the target shape.
    target_workload_measured = any(
        report.target_workload_measured for report in reports
    )
    reserves_by_rank = {
        report.rank: _vram_reserve_bytes(
            report.budget_bytes,
            target_workload_measured=target_workload_measured,
        )
        for report in reports
    }
    reserve = max(reserves_by_rank.values())
    current_placement_reserve_shortfall = current_placement_reserve_shortfall_bytes(
        reports
    )

    candidates = _consensus_candidates(reports)
    if not candidates:
        return _skip_plan(
            "no eligible residency alternatives",
            current_placement_reserve_shortfall_bytes=(
                current_placement_reserve_shortfall
            ),
        )

    candidate_component_names = {candidate.component_name for candidate in candidates}
    phase_constraints = {
        report.rank: _binding_phase_constraints(report, candidate_component_names)
        for report in reports
    }
    current_placement_reserve_shortfall = max(
        [
            current_placement_reserve_shortfall,
            *(
                max(
                    (
                        phase_peak + reserves_by_rank[report.rank] - report.budget_bytes
                        for _, phase_peak, _, _, _, _ in phase_constraints[report.rank]
                    ),
                    default=0,
                )
                for report in reports
            ),
        ]
    )
    current_placement_reserve_shortfall = max(0, current_placement_reserve_shortfall)
    resource_budgets: dict[str, int] = {}
    for report in reports:
        for resource_name, phase_peak, _, _, _, _ in phase_constraints[report.rank]:
            # The measured placement has already completed warmup. A negative
            # reserve headroom therefore means "do not grow this phase", not
            # that the current zero-delta placement is infeasible.
            resource_budgets[resource_name] = max(
                0,
                report.budget_bytes - phase_peak - reserves_by_rank[report.rank],
            )
    has_device_transition_options = any(
        candidate.device_transition_delta_bytes != 0 for candidate in candidates
    )
    if has_device_transition_options:
        for report in reports:
            resource_budgets[f"gpu:rank{report.rank}:placement-transition"] = max(
                0,
                report.budget_bytes
                - report.device_transition_allocated_bytes
                - reserves_by_rank[report.rank],
            )
    has_host_resources = any(
        candidate.pinned_host_delta_bytes != 0
        or candidate.host_unpin_scratch_bytes != 0
        or candidate.host_pin_scratch_bytes != 0
        or candidate.host_materialize_scratch_bytes != 0
        for candidate in candidates
    )
    if has_host_resources:
        for report in reports:
            prefix = f"node{report.node_rank}:rank{report.rank}"
            resource_budgets[f"hostpin:{prefix}"] = max(
                0,
                report.host_pin_capacity_bytes - report.pinned_host_bytes,
            )
            resource_budgets[f"hostram:{prefix}:unpin"] = (
                report.host_transition_headroom_bytes
            )
            resource_budgets[f"hostram:{prefix}:pin"] = (
                report.host_transition_headroom_bytes
            )
            resource_budgets[f"hostram:{prefix}:materialize"] = (
                report.host_transition_headroom_bytes
            )

    candidate_by_key = {candidate.option_key(): candidate for candidate in candidates}
    report_candidates = [
        {candidate.option_key(): candidate for candidate in report.candidates}
        for report in reports
    ]
    current_report_candidates = [
        {
            candidate.component_name: candidate
            for candidate in report.candidates
            if candidate.current_placement
        }
        for report in reports
    ]
    full_device_weight_bytes_by_component = [
        {
            component_name: max(
                candidate.target_device_weight_bytes
                for candidate in report.candidates
                if candidate.component_name == component_name
            )
            for component_name in {
                candidate.component_name for candidate in report.candidates
            }
        }
        for report in reports
    ]
    # Request timing is produced by the output rank. Other SPMD ranks still
    # contribute their VRAM and HostPin constraints, but their empty metrics
    # must not discard the replica's measured latency utility and fall back to
    # the much more aggressive transfer-byte ordering.
    timed_reports = [
        report
        for report in reports
        if report.estimated_request_duration_ns > 0
        and report.candidate_latency_savings_ns
    ]
    use_latency_utility = bool(timed_reports)
    latency_equivalence_ns = (
        max(
            MIN_LATENCY_EQUIVALENCE_NS,
            min(
                MAX_LATENCY_EQUIVALENCE_NS,
                int(
                    max(
                        report.estimated_request_duration_ns for report in timed_reports
                    )
                    * LATENCY_EQUIVALENCE_FRACTION
                ),
            ),
        )
        if use_latency_utility
        else 0
    )
    option_build_started = time.perf_counter()
    options = []
    for candidate in candidates:
        resource_deltas: dict[str, int] = {}
        for (
            report,
            rank_candidates,
            current_rank_candidates,
            rank_full_device_weight_bytes,
        ) in zip(
            reports,
            report_candidates,
            current_report_candidates,
            full_device_weight_bytes_by_component,
        ):
            rank_candidate = rank_candidates[candidate.option_key()]
            current_rank_candidate = current_rank_candidates.get(
                candidate.component_name
            )
            for (
                resource_name,
                _,
                present_components,
                measured_components,
                used_components,
                full_weight_transition_components,
            ) in phase_constraints[report.rank]:
                # ``present_components`` also includes request-end placement
                # carried into this phase. Only the measured set proves that
                # the phase peak already captured a complete materialization.
                transition_measured_full_weights = (
                    candidate.component_name in full_weight_transition_components
                    and (
                        candidate.component_name in measured_components
                        or (
                            current_rank_candidate is not None
                            and current_rank_candidate.target_mode()
                            in (LAYERWISE_OFFLOAD, RESIDENT)
                        )
                    )
                )
                if transition_measured_full_weights:
                    # The measured phase already contains the complete weights.
                    phase_cost = 0
                elif (
                    candidate.component_name in full_weight_transition_components
                    and rank_candidate.target_mode() == LAYERWISE_OFFLOAD
                ):
                    # Coarse offload can update CPU weights without putting the
                    # component on the device. Layerwise offload must materialize
                    # every layer for the same update, including non-resident ones.
                    phase_cost = rank_full_device_weight_bytes[candidate.component_name]
                elif candidate.component_name in used_components:
                    phase_cost = rank_candidate.active_device_delta_bytes
                elif candidate.component_name in present_components:
                    phase_cost = rank_candidate.present_device_delta_bytes
                elif rank_candidate.permanent_residency:
                    # This phase observed none of the component. A permanent
                    # target introduces its complete footprint, including the
                    # unmanaged weights of a formerly layerwise component.
                    phase_cost = max(
                        rank_candidate.target_device_weight_bytes,
                        rank_candidate.target_resident_weight_bytes,
                        0,
                    )
                else:
                    phase_cost = rank_candidate.inactive_device_delta_bytes
                resource_deltas[resource_name] = phase_cost
            if has_device_transition_options:
                resource_deltas[f"gpu:rank{report.rank}:placement-transition"] = (
                    rank_candidate.device_transition_delta_bytes
                )
            if has_host_resources:
                prefix = f"node{report.node_rank}:rank{report.rank}"
                host_resource = f"hostpin:{prefix}"
                resource_deltas[host_resource] = (
                    resource_deltas.get(host_resource, 0)
                    + rank_candidate.pinned_host_delta_bytes
                )
                resource_deltas[f"hostram:{prefix}:unpin"] = (
                    rank_candidate.host_unpin_scratch_bytes
                )
                resource_deltas[f"hostram:{prefix}:pin"] = (
                    rank_candidate.host_pin_scratch_bytes
                )
                resource_deltas[f"hostram:{prefix}:materialize"] = (
                    rank_candidate.host_materialize_scratch_bytes
                )
        estimated_latency_savings = (
            min(
                report.candidate_latency_savings_ns.get(candidate.option_key(), 0)
                for report in timed_reports
            )
            if use_latency_utility
            else candidate.h2d_bytes_per_request
        )
        options.append(
            PlacementOption(
                group_key=candidate.component_name,
                option_key=candidate.option_key(),
                resource_delta_bytes=resource_deltas,
                estimated_latency_savings=estimated_latency_savings,
                placement_cost_bytes=(
                    (
                        0
                        if candidate.current_placement
                        or candidate.target_mode() == candidate.residency_mode
                        else 1
                    ),
                    candidate.target_device_weight_bytes
                    or max(0, candidate.target_resident_weight_bytes),
                    -estimated_latency_savings,
                    candidate.target_pinned_host_bytes
                    or max(0, candidate.pinned_host_delta_bytes),
                ),
            )
        )

    candidates_by_component: dict[str, list[ResidencyTarget]] = {}
    for candidate in candidates:
        candidates_by_component.setdefault(candidate.component_name, []).append(
            candidate
        )
    complete_state_frontier = all(
        any(candidate.current_placement for candidate in component_candidates)
        for component_candidates in candidates_by_component.values()
    )
    logger.debug(
        "Auto residency option vectors built in %.3fs: options=%d, resources=%d",
        time.perf_counter() - option_build_started,
        len(options),
        len(resource_budgets),
    )
    solve_started = time.perf_counter()
    try:
        placement = optimize_placement(
            options,
            resource_budget_bytes=resource_budgets,
            estimated_latency_tolerance=(
                latency_equivalence_ns if use_latency_utility else 0
            ),
            require_selection_from_every_group=complete_state_frontier,
        )
    except NoFeasiblePlacementError as error:
        return _skip_plan(
            str(error),
            current_placement_reserve_shortfall_bytes=(
                current_placement_reserve_shortfall
            ),
        )
    logger.debug(
        "Auto residency joint solve completed in %.3fs",
        time.perf_counter() - solve_started,
    )
    changed_candidates = []
    for selection in placement.selections:
        candidate = candidate_by_key[selection.option_key]
        if not candidate.current_placement:
            changed_candidates.append(candidate)
    changes = rank_candidates_by_h2d_savings(changed_candidates)

    return AutoResidencyPlan(
        estimated_peak_bytes=estimated_peak,
        reserve_bytes=reserve,
        budget_bytes=budget,
        resource_budget_bytes=resource_budgets,
        changes=changes,
        current_placement_reserve_shortfall_bytes=(current_placement_reserve_shortfall),
    )


def format_plan_summary(
    *,
    plan: AutoResidencyPlan,
    workload: DefaultWorkload,
    records: Iterable[WarmupMemoryRecord] = (),
) -> str:
    """One-line decision summary for the startup log."""
    if plan.skip_reason is not None:
        return f"Auto residency: skipped ({plan.skip_reason})"
    changes = (
        ", ".join(_format_candidate_summary(candidate) for candidate in plan.changes)
        or "none"
    )
    measured = ", ".join(
        f"{record.width}x{record.height}x{record.num_frames}f="
        f"{record.peak_allocated_bytes / GIB_BYTES:.1f}GiB"
        for record in records
        if record.succeeded
    )
    measured_part = f"measured_allocated=[{measured}], " if measured else ""
    return (
        f"Auto residency: target={workload.describe()} "
        f"steps={workload.num_inference_steps}, "
        f"{measured_part}"
        f"estimated_peak={plan.estimated_peak_bytes / GIB_BYTES:.1f} GiB, "
        f"reserve={plan.reserve_bytes / GIB_BYTES:.1f} GiB, "
        f"budget={plan.budget_bytes / GIB_BYTES:.1f} GiB, "
        f"changes=[{changes}]"
    )


def _format_candidate_summary(candidate: ResidencyTarget) -> str:
    target_mode = candidate.target_mode()
    details = ""
    if target_mode == LAYERWISE_OFFLOAD:
        details = (
            f", layers={candidate.target_layerwise_resident_layers}, "
            f"pins={tuple(len(indices) for indices in candidate.target_layerwise_pinned_layers or ())}"
        )
    return f"{candidate.component_name}({target_mode}{details})"


def plan_summary_payload(
    *,
    plan: AutoResidencyPlan,
    status: str,
    short_validation: bool = False,
) -> dict:
    """Minimal decision payload for the warmup orchestrator.

    The orchestrator only branches on ``status``; the human-readable detail
    lives in the logged ``format_plan_summary`` line.
    """
    return {
        "status": status,
        "changed": [candidate.component_name for candidate in plan.changes],
        "short_validation": short_validation,
    }
