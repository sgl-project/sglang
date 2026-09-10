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
from typing import TYPE_CHECKING, Iterable, Mapping

import msgspec

from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
    RESIDENT,
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
