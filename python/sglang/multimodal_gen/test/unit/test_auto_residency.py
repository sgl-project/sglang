# SPDX-License-Identifier: Apache-2.0
"""Unit tests for warmup-calibrated auto residency adjustment."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.longlive2 import LongLive2T2VConfig
from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
    ACTIVATION_EXTRAPOLATION_MARGIN,
    GIB_BYTES,
    MAX_LAYERWISE_POLICY_TARGETS,
    MAX_LAYERWISE_RESIDENT_TARGETS,
    MIN_VRAM_RESERVE_BYTES,
    PAGEABLE_H2D_COST_MULTIPLIER,
    AppliedResidencyChange,
    AutoResidencyPlan,
    AutoResidencyRollbackError,
    DefaultWorkload,
    RankResidencyReport,
    ResidencyTarget,
    WarmupMemoryRecord,
    _layerwise_pin_targets,
    _layerwise_policy_targets,
    _layerwise_resident_targets,
    apply_residency_changes,
    collect_residency_targets,
    commit_residency_changes,
    component_resident_size_bytes,
    component_runtime_weight_bytes,
    current_placement_reserve_shortfall_bytes,
    estimate_allocator_headroom_bytes,
    estimate_candidate_latency_savings_ns,
    estimate_default_workload_peak_bytes,
    estimate_default_workload_timing,
    estimate_layerwise_layer_uses,
    estimate_workload_phase_peaks,
    format_applied_changes,
    measured_failed_workload_phase_peaks,
    plan_auto_residency,
    rank_candidates_by_h2d_savings,
    resolve_measured_default_workload,
    rollback_residency_changes,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
    RESIDENT,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
    HostPinBudget,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    RESIDENCY_POLICY_LEADING,
    RESIDENCY_POLICY_STRIDED,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.warmup_request_builder import (
    SERVER_WARMUP_MAX_VIDEO_FRAMES,
    _resolve_auto_residency_warmup_shape,
    _resolve_warmup_num_frames,
)


def _record(
    *,
    width=832,
    height=480,
    num_frames=17,
    baseline_gib=10,
    peak_gib=12,
    peak_reserved_gib=None,
    succeeded=True,
    num_inference_steps=1,
    total_duration_ms=0.0,
    stage_duration_ms=None,
    step_duration_ms=(),
    step_duration_ms_by_stage=None,
    stage_iterations=None,
    phase_active_components=None,
    phase_used_components=None,
    phase_full_weight_transition_components=None,
    layerwise_layer_uses=None,
    layerwise_layer_uses_by_stage=None,
) -> WarmupMemoryRecord:
    return WarmupMemoryRecord(
        width=width,
        height=height,
        num_frames=num_frames,
        baseline_allocated_bytes=baseline_gib * GIB_BYTES,
        peak_allocated_bytes=peak_gib * GIB_BYTES,
        succeeded=succeeded,
        peak_reserved_bytes=int((peak_reserved_gib or peak_gib) * GIB_BYTES),
        num_inference_steps=num_inference_steps,
        total_duration_ms=total_duration_ms,
        stage_duration_ms=stage_duration_ms or {},
        step_duration_ms=step_duration_ms,
        step_duration_ms_by_stage=step_duration_ms_by_stage or {},
        stage_iterations=stage_iterations or {},
        phase_active_components=phase_active_components or {},
        phase_used_components=phase_used_components or {},
        phase_full_weight_transition_components=(
            phase_full_weight_transition_components or {}
        ),
        layerwise_layer_uses=layerwise_layer_uses or {},
        layerwise_layer_uses_by_stage=layerwise_layer_uses_by_stage or {},
    )


class TestEstimateDefaultWorkloadTiming:
    def test_scales_only_denoising_with_target_steps(self):
        record = _record(
            num_inference_steps=2,
            total_duration_ms=1_200,
            stage_duration_ms={
                "TextEncodingStage": 100,
                "DenoisingStage": 1_000,
            },
            phase_active_components={
                "0:TextEncodingStage:use:text_encoder": ("text_encoder",),
                "1:DenoisingStage:use:transformer": ("transformer",),
            },
        )

        request_ns, stage_ns, component_stages = estimate_default_workload_timing(
            records=[record],
            target_units=record.workload_units(),
            target_num_inference_steps=40,
        )

        assert stage_ns == {
            "TextEncodingStage": 100_000_000,
            "DenoisingStage": 20_000_000_000,
        }
        assert request_ns == 20_200_000_000
        assert component_stages == {
            "text_encoder": ("TextEncodingStage",),
            "transformer": ("DenoisingStage",),
        }

    def test_uses_steady_step_instead_of_scaling_first_step_setup(self):
        record = _record(
            num_inference_steps=2,
            total_duration_ms=720,
            stage_duration_ms={
                "TextEncodingStage": 100,
                "DenoisingStage": 620,
            },
            step_duration_ms=(500, 100),
            phase_active_components={
                "0:TextEncodingStage:use:text_encoder": ("text_encoder",),
                "1:DenoisingStage:use:transformer": ("transformer",),
            },
        )

        request_ns, stage_ns, _ = estimate_default_workload_timing(
            records=[record],
            target_units=record.workload_units(),
            target_num_inference_steps=10,
        )

        assert stage_ns["DenoisingStage"] == 1_020_000_000
        assert request_ns == 1_120_000_000

    def test_scales_standard_and_nonstandard_denoising_stages_together(self):
        record = _record(
            num_inference_steps=2,
            total_duration_ms=1_200,
            stage_duration_ms={
                "TextEncodingStage": 100,
                "DenoisingStage": 500,
                "CustomDenoisingStage": 600,
            },
            phase_active_components={
                "0:TextEncodingStage:use:text_encoder": ("text_encoder",),
                "1:DenoisingStage:use:transformer": ("transformer",),
                "2:CustomDenoisingStage:use:custom_refiner": ("custom_refiner",),
            },
        )

        request_ns, stage_ns, _ = estimate_default_workload_timing(
            records=[record],
            target_units=record.workload_units(),
            target_num_inference_steps=10,
        )

        assert stage_ns == {
            "TextEncodingStage": 100_000_000,
            "DenoisingStage": 2_500_000_000,
            "CustomDenoisingStage": 3_000_000_000,
        }
        assert request_ns == 5_600_000_000

    def test_uses_each_stage_iteration_target(self):
        record = _record(
            num_inference_steps=4,
            total_duration_ms=1_100,
            stage_duration_ms={
                "ShapeDenoisingStage": 500,
                "PaintStage": 600,
            },
            step_duration_ms_by_stage={
                "ShapeDenoisingStage": (200, 100, 100, 100),
            },
            stage_iterations={
                "ShapeDenoisingStage": (4, 50),
                "PaintStage": (4, 30),
            },
            phase_active_components={
                "0:ShapeDenoisingStage:use:hy3dshape_model": ("hy3dshape_model",),
                "1:PaintStage:use:paint_transformer": ("paint_transformer",),
            },
        )

        request_ns, stage_ns, _ = estimate_default_workload_timing(
            records=[record],
            target_units=record.workload_units(),
            target_num_inference_steps=50,
        )

        assert stage_ns == {
            "ShapeDenoisingStage": 5_000_000_000,
            "PaintStage": 4_500_000_000,
        }
        assert request_ns == 9_500_000_000

    def test_async_stage_time_does_not_cap_repeated_dit_transfers(self):
        partial = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=10 * GIB_BYTES,
            h2d_bytes_per_request=5 * GIB_BYTES,
            target_layerwise_resident_layers=(5,),
            target_layerwise_pinned_layers=((),),
        )
        resident = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=20 * GIB_BYTES,
            h2d_bytes_per_request=10 * GIB_BYTES,
            target_layerwise_resident_layers=(10,),
            target_layerwise_pinned_layers=((),),
        )

        savings = estimate_candidate_latency_savings_ns(
            candidates=[partial, resident],
            request_duration_ns=10_000_000_000,
            stage_duration_ns={"denoise": 100_000_000},
            component_stages={"transformer": ("denoise",)},
        )

        assert savings[partial.option_key()] > 100_000_000
        assert savings[resident.option_key()] > savings[partial.option_key()]

    def test_dit_request_cap_does_not_compress_marginal_transfer_savings(self):
        partial = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=95 * GIB_BYTES,
            h2d_bytes_per_request=95 * GIB_BYTES,
            target_layerwise_resident_layers=(95,),
            target_layerwise_pinned_layers=((),),
        )
        resident = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=100 * GIB_BYTES,
            h2d_bytes_per_request=100 * GIB_BYTES,
            target_layerwise_resident_layers=(100,),
            target_layerwise_pinned_layers=((),),
        )

        savings = estimate_candidate_latency_savings_ns(
            candidates=[partial, resident],
            request_duration_ns=1_000_000_000,
            stage_duration_ns={"denoise": 100_000_000},
            component_stages={"transformer": ("denoise",)},
        )

        assert savings[resident.option_key()] == 1_000_000_000
        assert (
            savings[resident.option_key()] - savings[partial.option_key()] > 200_000_000
        )

    def test_measured_placement_keeps_the_full_cost_of_a_slower_dit_target(self):
        coarse = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=140 * GIB_BYTES,
            current_placement=True,
        )
        virtual_layerwise = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=10 * GIB_BYTES,
            h2d_bytes_per_request=20 * GIB_BYTES,
            target_layerwise_resident_layers=(10,),
            target_layerwise_pinned_layers=((),),
        )
        resident = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=60 * GIB_BYTES,
            h2d_bytes_per_request=150 * GIB_BYTES,
            permanent_residency=True,
        )

        savings = estimate_candidate_latency_savings_ns(
            candidates=[coarse, virtual_layerwise, resident],
            request_duration_ns=1_000_000_000,
            stage_duration_ns={"denoise": 1_000_000_000},
            component_stages={"transformer": ("denoise",)},
        )

        assert savings[coarse.option_key()] == 0
        assert savings[resident.option_key()] == 416_666_666
        assert savings[virtual_layerwise.option_key()] == -5_000_000_000

    def test_prefetch_phase_does_not_claim_stage_latency(self):
        record = _record(
            num_inference_steps=2,
            total_duration_ms=1_000,
            stage_duration_ms={"encode": 100, "denoise": 900},
            phase_active_components={
                "0:encode:prefetch:transformer": ("transformer",),
                "1:denoise:use:transformer": ("transformer",),
            },
        )

        _, _, component_stages = estimate_default_workload_timing(
            records=[record],
            target_units=record.workload_units(),
            target_num_inference_steps=2,
        )

        assert component_stages == {"transformer": ("denoise",)}

    def test_repeated_nonstandard_component_uses_request_latency_cap(self):
        partial = ResidencyTarget(
            component_name="custom_refiner",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=10 * GIB_BYTES,
            h2d_bytes_per_request=5 * GIB_BYTES,
            target_layerwise_resident_layers=(5,),
            target_layerwise_pinned_layers=((),),
        )
        resident = ResidencyTarget(
            component_name="custom_refiner",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=20 * GIB_BYTES,
            h2d_bytes_per_request=10 * GIB_BYTES,
            target_layerwise_resident_layers=(10,),
            target_layerwise_pinned_layers=((),),
        )

        savings = estimate_candidate_latency_savings_ns(
            candidates=[partial, resident],
            request_duration_ns=10_000_000_000,
            stage_duration_ns={"denoise": 100_000_000},
            component_stages={"custom_refiner": ("denoise",)},
            repeated_components={"custom_refiner"},
        )

        assert savings[partial.option_key()] > 100_000_000
        assert savings[resident.option_key()] > savings[partial.option_key()]

    def test_stage_cap_preserves_order_inside_one_shot_component_frontier(self):
        partial = ResidencyTarget(
            component_name="text_encoder",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=10 * GIB_BYTES,
            h2d_bytes_per_request=5 * GIB_BYTES,
            target_layerwise_resident_layers=(5,),
            target_layerwise_pinned_layers=((),),
        )
        resident = ResidencyTarget(
            component_name="text_encoder",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=20 * GIB_BYTES,
            h2d_bytes_per_request=10 * GIB_BYTES,
            target_layerwise_resident_layers=(10,),
            target_layerwise_pinned_layers=((),),
        )

        savings = estimate_candidate_latency_savings_ns(
            candidates=[partial, resident],
            request_duration_ns=10_000_000_000,
            stage_duration_ns={"encode": 100_000_000},
            component_stages={"text_encoder": ("encode",)},
        )

        assert savings[partial.option_key()] == 50_000_000
        assert savings[resident.option_key()] == 100_000_000

    def test_transfer_savings_uses_component_stage_as_cross_component_cap(self):
        candidates = [
            _candidate("text_encoder", weight_gib=10, h2d_gib=10),
            _candidate("transformer", weight_gib=10, h2d_gib=10),
            _candidate("cold_encoder", weight_gib=10, h2d_gib=-1),
        ]

        savings = estimate_candidate_latency_savings_ns(
            candidates=candidates,
            request_duration_ns=10_000_000_000,
            stage_duration_ns={"encode": 100_000_000, "denoise": 4_000_000_000},
            component_stages={
                "text_encoder": ("encode",),
                "transformer": ("denoise",),
            },
        )

        assert savings[candidates[0].option_key()] == 100_000_000
        assert savings[candidates[1].option_key()] > 400_000_000
        assert savings[candidates[2].option_key()] < 0


class TestEstimateLayerwiseLayerUses:
    def test_scales_repeated_groups_but_not_one_shot_groups(self):
        record = _record(
            num_inference_steps=2,
            layerwise_layer_uses={
                "transformer": {
                    "token_refiner.blocks": (1, 1),
                    "blocks": (2, 2, 2),
                },
                "vae": {
                    "encoder.down_blocks": (0, 0),
                    "decoder.up_blocks": (1, 1),
                },
            },
        )

        uses = estimate_layerwise_layer_uses(
            records=[record],
            target_units=record.workload_units(),
            target_num_inference_steps=10,
        )

        assert uses["transformer"] == {
            "token_refiner.blocks": (1, 1),
            "blocks": (10, 10, 10),
        }
        assert uses["vae"] == {
            "encoder.down_blocks": (0, 0),
            "decoder.up_blocks": (1, 1),
        }

    def test_stage_role_scales_nonstandard_denoiser_but_not_encoder_repeats(self):
        record = _record(
            num_inference_steps=2,
            phase_used_components={
                "0:TextEncodingStage:use:custom_encoder": ("custom_encoder",),
                "1:CustomDenoisingStage:use:custom_refiner": ("custom_refiner",),
            },
            layerwise_layer_uses={
                "custom_encoder": {"layers": (2, 2)},
                "custom_refiner": {"blocks": (2, 2)},
            },
        )

        uses = estimate_layerwise_layer_uses(
            records=[record],
            target_units=record.workload_units(),
            target_num_inference_steps=10,
        )

        assert uses["custom_encoder"]["layers"] == (2, 2)
        assert uses["custom_refiner"]["blocks"] == (10, 10)

    def test_scales_each_stage_calls_with_its_own_iteration_target(self):
        record = _record(
            num_inference_steps=4,
            phase_used_components={
                "0:ShapeStage:use:transformer": ("transformer",),
                "1:PaintStage:use:transformer": ("transformer",),
            },
            stage_iterations={
                "ShapeStage": (4, 50),
                "PaintStage": (4, 30),
            },
            layerwise_layer_uses={
                "transformer": {"blocks": (9, 9), "one_shot": (2, 2)}
            },
            layerwise_layer_uses_by_stage={
                "ShapeStage": {"transformer": {"blocks": (4, 4), "one_shot": (1, 1)}},
                "PaintStage": {"transformer": {"blocks": (4, 4), "one_shot": (1, 1)}},
            },
        )

        uses = estimate_layerwise_layer_uses(
            records=[record],
            target_units=record.workload_units(),
            target_num_inference_steps=50,
        )

        # one untracked one-shot call + 50 shape calls + 30 paint calls
        assert uses["transformer"]["blocks"] == (81, 81)
        assert uses["transformer"]["one_shot"] == (2, 2)


class TestEstimateDefaultWorkloadPeak:
    def test_same_shape_uses_measured_peak(self):
        record = _record()
        estimate = estimate_default_workload_peak_bytes(
            records=[record], target_units=record.workload_units()
        )
        assert estimate == record.peak_allocated_bytes

    def test_unknown_target_disables_estimation(self):
        # An unknown target would silently equate the capped warmup peak with
        # the serving peak and promote with no margin at all.
        record = _record()
        estimate = estimate_default_workload_peak_bytes(
            records=[record], target_units=None
        )
        assert estimate is None

    def test_single_capped_record_scales_only_the_activation_part(self):
        # Warmup capped to 832x480x17; Wan-class default is 704x1280x121.
        record = _record()
        target_units = 704 * 1280 * 121
        ratio = target_units / record.workload_units()
        assert ratio > 10  # the cap ratio this formula exists for

        estimate = estimate_default_workload_peak_bytes(
            records=[record], target_units=target_units
        )
        activation = record.peak_allocated_bytes - record.baseline_allocated_bytes
        expected = record.baseline_allocated_bytes + int(
            activation * ratio * ACTIVATION_EXTRAPOLATION_MARGIN
        )
        assert estimate == expected
        # Scaling the whole peak would inflate the estimate by the resident
        # weights times the cap ratio and adjustment would never trigger.
        naive = int(record.peak_allocated_bytes * ratio)
        assert estimate < naive

    def test_two_point_fit_separates_constant_from_linear(self):
        # Two calibration sizes let the estimator measure the slope instead
        # of assuming everything above the baseline scales. Under offload the
        # baseline is nearly empty, so the single-point formula would scale
        # the whole peak (~x13 here); the fit extrapolates only the measured
        # linear part.
        small = _record(num_frames=9, baseline_gib=1, peak_gib=14)
        large = _record(num_frames=17, baseline_gib=1, peak_gib=16)
        target_units = 1280 * 720 * 81

        estimate = estimate_default_workload_peak_bytes(
            records=[small, large], target_units=target_units
        )
        slope = (large.peak_allocated_bytes - small.peak_allocated_bytes) / (
            large.workload_units() - small.workload_units()
        )
        constant = large.peak_allocated_bytes - slope * large.workload_units()
        expected = int(
            constant + slope * target_units * ACTIVATION_EXTRAPOLATION_MARGIN
        )
        assert estimate == expected
        single_point = estimate_default_workload_peak_bytes(
            records=[large], target_units=target_units
        )
        assert estimate < single_point

    def test_negative_slope_falls_back_to_single_point_formula(self):
        small = _record(num_frames=9, peak_gib=16)
        large = _record(num_frames=17, peak_gib=14)
        target_units = 1280 * 720 * 81
        estimate = estimate_default_workload_peak_bytes(
            records=[small, large], target_units=target_units
        )
        fallback = max(
            estimate_default_workload_peak_bytes(
                records=[record], target_units=target_units
            )
            for record in (small, large)
        )
        assert estimate == fallback

    def test_weight_floor_prevents_scaling_constant_component_memory(self):
        small = WarmupMemoryRecord(
            width=832,
            height=480,
            num_frames=9,
            baseline_allocated_bytes=GIB_BYTES,
            peak_allocated_bytes=int(29.5 * GIB_BYTES),
            succeeded=True,
        )
        large = WarmupMemoryRecord(
            width=832,
            height=480,
            num_frames=17,
            baseline_allocated_bytes=GIB_BYTES,
            peak_allocated_bytes=int(29.4 * GIB_BYTES),
            succeeded=True,
        )
        target_units = 1024 * 1024 * 81

        estimate = estimate_default_workload_peak_bytes(
            records=[small, large],
            target_units=target_units,
            constant_weight_bytes=int(29.4 * GIB_BYTES),
        )
        fallback_without_weights = estimate_default_workload_peak_bytes(
            records=[small, large], target_units=target_units
        )

        assert estimate < 50 * GIB_BYTES
        assert estimate < fallback_without_weights

    def test_covering_measurement_bounds_the_target(self):
        capped = _record(num_frames=17, peak_gib=12)
        full = _record(width=1280, height=720, num_frames=81, peak_gib=30)
        estimate = estimate_default_workload_peak_bytes(
            records=[capped, full], target_units=1280 * 720 * 81
        )
        assert estimate == full.peak_allocated_bytes

    def test_multiple_records_take_the_max(self):
        low = _record(peak_gib=12)
        high = _record(peak_gib=20)
        estimate = estimate_default_workload_peak_bytes(
            records=[low, high], target_units=low.workload_units()
        )
        assert estimate == high.peak_allocated_bytes

    def test_failure_at_the_target_size_disables_estimation(self):
        good = _record(num_frames=9, peak_gib=8)
        failed = _record(num_frames=17, succeeded=False)
        assert (
            estimate_default_workload_peak_bytes(
                records=[good, failed], target_units=failed.workload_units()
            )
            is None
        )

    def test_failure_below_the_target_size_disables_estimation(self):
        good = _record(num_frames=9, peak_gib=8)
        failed = _record(num_frames=17, succeeded=False)
        assert (
            estimate_default_workload_peak_bytes(
                records=[good, failed], target_units=failed.workload_units() * 2
            )
            is None
        )

    def test_failure_above_the_target_size_is_dropped(self):
        good = _record(num_frames=9, peak_gib=8)
        failed = _record(num_frames=81, succeeded=False)
        estimate = estimate_default_workload_peak_bytes(
            records=[good, failed], target_units=good.workload_units()
        )
        assert estimate == good.peak_allocated_bytes

    def test_no_records_disables_estimation(self):
        assert (
            estimate_default_workload_peak_bytes(records=[], target_units=None) is None
        )

    def test_phase_estimation_preserves_active_component_membership(self):
        small = _record(num_frames=9, peak_gib=30)
        large = _record(num_frames=17, peak_gib=32)
        small = WarmupMemoryRecord(
            width=small.width,
            height=small.height,
            num_frames=small.num_frames,
            baseline_allocated_bytes=small.baseline_allocated_bytes,
            peak_allocated_bytes=small.peak_allocated_bytes,
            succeeded=small.succeeded,
            phase_peak_allocated_bytes={"denoise": 30 * GIB_BYTES},
            phase_active_components={"denoise": ("transformer",)},
        )
        large = WarmupMemoryRecord(
            width=large.width,
            height=large.height,
            num_frames=large.num_frames,
            baseline_allocated_bytes=large.baseline_allocated_bytes,
            peak_allocated_bytes=large.peak_allocated_bytes,
            succeeded=large.succeeded,
            phase_peak_allocated_bytes={"denoise": 32 * GIB_BYTES},
            phase_active_components={"denoise": ("transformer",)},
        )

        peaks, active, used, _, _ = estimate_workload_phase_peaks(
            records=[small, large],
            target_units=832 * 480 * 81,
            component_weight_bytes={"transformer": 28 * GIB_BYTES},
        )

        assert peaks["denoise"] >= 32 * GIB_BYTES
        assert active == {"denoise": ("transformer",)}
        assert used == active

    def test_phase_estimation_uses_allocated_peak(self):
        record = WarmupMemoryRecord(
            width=1024,
            height=1024,
            num_frames=1,
            baseline_allocated_bytes=5 * GIB_BYTES,
            peak_allocated_bytes=12 * GIB_BYTES,
            succeeded=True,
            phase_peak_allocated_bytes={"denoise": 11 * GIB_BYTES},
            phase_active_components={"denoise": ("transformer",)},
        )

        peaks, _, _, _, _ = estimate_workload_phase_peaks(
            records=[record],
            target_units=record.workload_units(),
            component_weight_bytes={"transformer": 10 * GIB_BYTES},
        )

        assert peaks["denoise"] == 11 * GIB_BYTES

    def test_phase_estimation_preserves_full_weight_transition_components(self):
        record = WarmupMemoryRecord(
            width=1024,
            height=1024,
            num_frames=1,
            baseline_allocated_bytes=2 * GIB_BYTES,
            peak_allocated_bytes=4 * GIB_BYTES,
            succeeded=True,
            phase_peak_allocated_bytes={"lora_switch": 4 * GIB_BYTES},
            phase_full_weight_transition_components={"lora_switch": ("transformer",)},
        )

        peaks, _, _, _, transitions = estimate_workload_phase_peaks(
            records=[record],
            target_units=record.workload_units(),
            component_weight_bytes={"transformer": 2 * GIB_BYTES},
        )

        assert peaks == {"lora_switch": 4 * GIB_BYTES}
        assert transitions == {"lora_switch": ("transformer",)}

    def test_phase_estimation_prefers_target_layout_over_smaller_warmup(self):
        small = WarmupMemoryRecord(
            width=256,
            height=256,
            num_frames=9,
            baseline_allocated_bytes=2 * GIB_BYTES,
            peak_allocated_bytes=30 * GIB_BYTES,
            succeeded=True,
            phase_peak_allocated_bytes={"denoise": 30 * GIB_BYTES},
            phase_active_components={"denoise": ()},
        )
        target = WarmupMemoryRecord(
            width=768,
            height=512,
            num_frames=25,
            baseline_allocated_bytes=2 * GIB_BYTES,
            peak_allocated_bytes=48 * GIB_BYTES,
            succeeded=True,
            phase_peak_allocated_bytes={"denoise": 48 * GIB_BYTES},
            phase_active_components={"denoise": ("transformer",)},
        )

        peaks, active, used, _, _ = estimate_workload_phase_peaks(
            records=[small, target],
            target_units=target.workload_units(),
            component_weight_bytes={"transformer": 40 * GIB_BYTES},
        )

        assert peaks == {"denoise": 48 * GIB_BYTES}
        assert active == {"denoise": ("transformer",)}
        assert used == active

    def test_phase_estimation_keeps_distinct_active_layouts_separate(self):
        transformer_phase = WarmupMemoryRecord(
            width=768,
            height=512,
            num_frames=25,
            baseline_allocated_bytes=2 * GIB_BYTES,
            peak_allocated_bytes=48 * GIB_BYTES,
            succeeded=True,
            phase_peak_allocated_bytes={"denoise": 48 * GIB_BYTES},
            phase_active_components={"denoise": ("transformer",)},
        )
        encoder_phase = WarmupMemoryRecord(
            width=768,
            height=512,
            num_frames=25,
            baseline_allocated_bytes=2 * GIB_BYTES,
            peak_allocated_bytes=20 * GIB_BYTES,
            succeeded=True,
            phase_peak_allocated_bytes={"denoise": 20 * GIB_BYTES},
            phase_active_components={"denoise": ("text_encoder",)},
        )

        peaks, active, used, _, _ = estimate_workload_phase_peaks(
            records=[transformer_phase, encoder_phase],
            target_units=transformer_phase.workload_units(),
            component_weight_bytes={
                "transformer": 40 * GIB_BYTES,
                "text_encoder": 10 * GIB_BYTES,
            },
        )

        assert peaks == {
            "denoise:layout:0": 20 * GIB_BYTES,
            "denoise:layout:1": 48 * GIB_BYTES,
        }
        assert active == {
            "denoise:layout:0": ("text_encoder",),
            "denoise:layout:1": ("transformer",),
        }
        assert used == active

    def test_failed_probe_preserves_phase_layouts_for_recovery(self):
        first = WarmupMemoryRecord(
            width=512,
            height=512,
            num_frames=1,
            baseline_allocated_bytes=20 * GIB_BYTES,
            peak_allocated_bytes=30 * GIB_BYTES,
            succeeded=False,
            phase_peak_allocated_bytes={"denoise": 29 * GIB_BYTES},
            phase_active_components={"denoise": ("transformer",)},
            phase_used_components={"denoise": ("transformer",)},
        )
        second = WarmupMemoryRecord(
            width=512,
            height=512,
            num_frames=1,
            baseline_allocated_bytes=20 * GIB_BYTES,
            peak_allocated_bytes=31 * GIB_BYTES,
            succeeded=False,
            phase_peak_allocated_bytes={"denoise": 31 * GIB_BYTES},
            phase_active_components={"denoise": ("transformer_2",)},
            phase_used_components={"denoise": ("transformer_2",)},
        )

        peak, phases, active, used, _, _ = measured_failed_workload_phase_peaks(
            records=[first, second],
            target_units=first.workload_units(),
        )

        assert peak == 31 * GIB_BYTES
        assert phases == {
            "denoise:layout:0": 29 * GIB_BYTES,
            "denoise:layout:1": 31 * GIB_BYTES,
        }
        assert set(active.values()) == {("transformer",), ("transformer_2",)}
        assert used == active

    def test_phase_estimation_keeps_distinct_prefetch_layouts_separate(self):
        without_prefetch = WarmupMemoryRecord(
            width=768,
            height=512,
            num_frames=25,
            baseline_allocated_bytes=2 * GIB_BYTES,
            peak_allocated_bytes=20 * GIB_BYTES,
            succeeded=True,
            phase_peak_allocated_bytes={"encode": 20 * GIB_BYTES},
            phase_active_components={"encode": ("text_encoder",)},
            phase_used_components={"encode": ("text_encoder",)},
        )
        with_prefetch = WarmupMemoryRecord(
            width=768,
            height=512,
            num_frames=25,
            baseline_allocated_bytes=2 * GIB_BYTES,
            peak_allocated_bytes=30 * GIB_BYTES,
            succeeded=True,
            phase_peak_allocated_bytes={"encode": 30 * GIB_BYTES},
            phase_active_components={
                "encode": ("text_encoder", "transformer"),
            },
            phase_used_components={"encode": ("text_encoder",)},
            phase_prefetched_components={"encode": ("transformer",)},
        )

        peaks, active, used, prefetched, _ = estimate_workload_phase_peaks(
            records=[without_prefetch, with_prefetch],
            target_units=without_prefetch.workload_units(),
            component_weight_bytes={
                "transformer": 10 * GIB_BYTES,
                "text_encoder": 8 * GIB_BYTES,
            },
        )

        assert peaks == {
            "encode:layout:0": 20 * GIB_BYTES,
            "encode:layout:1": 30 * GIB_BYTES,
        }
        assert set(active.values()) == {
            ("text_encoder",),
            ("text_encoder", "transformer"),
        }
        assert set(used.values()) == {("text_encoder",)}
        assert set(prefetched.values()) == {(), ("transformer",)}


def _candidate(
    name: str,
    *,
    mode: str = COMPONENT_OFFLOAD,
    weight_gib: int = 10,
    h2d_gib: int | None = None,
) -> ResidencyTarget:
    layerwise = mode == LAYERWISE_OFFLOAD
    return ResidencyTarget(
        component_name=name,
        residency_mode=mode,
        target_resident_weight_bytes=weight_gib * GIB_BYTES,
        h2d_bytes_per_request=(h2d_gib if h2d_gib is not None else weight_gib)
        * GIB_BYTES,
        target_layerwise_resident_layers=(1,) if layerwise else None,
        target_layerwise_pinned_layers=((),) if layerwise else None,
        permanent_residency=True,
        active_device_delta_bytes=0,
        inactive_device_delta_bytes=weight_gib * GIB_BYTES,
    )


def _report(
    *,
    rank: int = 0,
    budget_gib: float = 100,
    estimated_gib: float | None = 50,
    planning_headroom_correction_gib: int = 0,
    observed_reserved_gib: float = 0,
    candidates: list[ResidencyTarget] | None = None,
    skip_reason: str | None = None,
    phase_peaks_gib: dict[str, float] | None = None,
    phase_components: dict[str, tuple[str, ...]] | None = None,
    phase_present_components: dict[str, tuple[str, ...]] | None = None,
    phase_prefetched_components: dict[str, tuple[str, ...]] | None = None,
    phase_full_weight_transition_components: dict[str, tuple[str, ...]] | None = None,
    component_weight_gib: dict[str, int] | None = None,
    component_active_weight_gib: dict[str, int] | None = None,
    node_rank: int = 0,
    pinned_host_gib: int = 0,
    host_pin_capacity_gib: int = 0,
    host_transition_headroom_gib: int = 0,
    device_transition_allocated_gib: int = 0,
    target_workload_measured: bool = False,
    estimated_request_duration_ns: int = 0,
    measured_request_duration_ns: int = 0,
    candidate_latency_savings_ns: dict[str, int] | None = None,
    warmup_oom: bool = False,
    require_feasible_placement: bool = False,
) -> RankResidencyReport:
    return RankResidencyReport(
        rank=rank,
        budget_bytes=int(budget_gib * GIB_BYTES),
        estimated_peak_bytes=(
            None if estimated_gib is None else int(estimated_gib * GIB_BYTES)
        ),
        planning_headroom_correction_bytes=(
            planning_headroom_correction_gib * GIB_BYTES
        ),
        target_workload_measured=target_workload_measured,
        observed_reserved_bytes=observed_reserved_gib * GIB_BYTES,
        estimated_peak_bytes_by_phase={
            name: int(value * GIB_BYTES)
            for name, value in (phase_peaks_gib or {}).items()
        },
        active_components_by_phase=(
            phase_present_components
            if phase_present_components is not None
            else phase_components or {}
        ),
        used_components_by_phase=phase_components or {},
        prefetched_components_by_phase=phase_prefetched_components or {},
        full_weight_transition_components_by_phase=(
            phase_full_weight_transition_components or {}
        ),
        current_device_weight_bytes_by_component={
            name: value * GIB_BYTES
            for name, value in (component_weight_gib or {}).items()
        },
        current_active_weight_bytes_by_component={
            name: value * GIB_BYTES
            for name, value in (component_active_weight_gib or {}).items()
        },
        node_rank=node_rank,
        pinned_host_bytes=pinned_host_gib * GIB_BYTES,
        host_pin_capacity_bytes=host_pin_capacity_gib * GIB_BYTES,
        host_transition_headroom_bytes=(host_transition_headroom_gib * GIB_BYTES),
        device_transition_allocated_bytes=(device_transition_allocated_gib * GIB_BYTES),
        estimated_request_duration_ns=estimated_request_duration_ns,
        measured_request_duration_ns=measured_request_duration_ns,
        candidate_latency_savings_ns=candidate_latency_savings_ns or {},
        candidates=candidates if candidates is not None else [_candidate("vae")],
        warmup_oom=warmup_oom,
        require_feasible_placement=require_feasible_placement,
        skip_reason=skip_reason,
    )


class TestResolveMeasuredDefaultWorkload:
    def test_uses_effective_warmup_resolution_for_implicit_image_size(self):
        workload = DefaultWorkload(
            width=None,
            height=None,
            num_frames=1,
            num_inference_steps=40,
        )

        resolved = resolve_measured_default_workload(
            workload,
            [
                _record(width=512, height=512),
                _record(width=1024, height=1024),
            ],
        )

        assert resolved == DefaultWorkload(
            width=1024,
            height=1024,
            num_frames=1,
            num_inference_steps=40,
        )

    def test_keeps_default_frames_when_warmup_caps_video(self):
        workload = DefaultWorkload(
            width=None,
            height=None,
            num_frames=81,
            num_inference_steps=30,
        )

        resolved = resolve_measured_default_workload(
            workload, [_record(width=832, height=480, num_frames=17)]
        )

        assert resolved.num_frames == 81

    def test_does_not_replace_explicit_default_shape(self):
        workload = DefaultWorkload(
            width=1280,
            height=720,
            num_frames=81,
            num_inference_steps=30,
        )

        assert (
            resolve_measured_default_workload(
                workload, [_record(width=512, height=512)]
            )
            is workload
        )


class TestPlanAutoResidency:
    def test_rank_skip_reason_propagates(self):
        plan = plan_auto_residency(
            reports=[_report(), _report(rank=1, skip_reason="no measurements")]
        )
        assert plan.skip_reason == "rank 1: no measurements"

    def test_missing_estimate_skips(self):
        plan = plan_auto_residency(reports=[_report(estimated_gib=None)])
        assert plan.skip_reason is not None

    def test_worst_case_aggregation_across_ranks(self):
        plan = plan_auto_residency(
            reports=[
                _report(rank=0, budget_gib=100, estimated_gib=40),
                _report(rank=1, budget_gib=80, estimated_gib=60),
            ]
        )
        assert plan.budget_bytes == 80 * GIB_BYTES
        assert plan.estimated_peak_bytes == 60 * GIB_BYTES

    def test_reserve_has_absolute_floor(self):
        # 10% of a 20 GiB budget is 2 GiB; the floor must lift it to 4 GiB.
        plan = plan_auto_residency(reports=[_report(budget_gib=20, estimated_gib=10)])
        assert plan.reserve_bytes == MIN_VRAM_RESERVE_BYTES

    def test_reserve_floor_is_capped_on_a_small_card(self):
        # Keep enough room for unmeasured activations without applying the
        # datacenter-card 4 GiB floor verbatim.
        plan = plan_auto_residency(reports=[_report(budget_gib=10, estimated_gib=4)])
        assert plan.reserve_bytes == 3 * GIB_BYTES

    def test_target_workload_measurement_uses_tighter_reserve(self):
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=100,
                    estimated_gib=50,
                    target_workload_measured=True,
                )
            ]
        )
        assert plan.reserve_bytes == 5 * GIB_BYTES

    def test_target_workload_measurement_is_replica_wide(self):
        candidate = _candidate("vae", weight_gib=1)
        plan = plan_auto_residency(
            reports=[
                _report(
                    rank=0,
                    budget_gib=100,
                    estimated_gib=50,
                    candidates=[candidate],
                    target_workload_measured=True,
                ),
                _report(
                    rank=1,
                    budget_gib=140,
                    estimated_gib=70,
                    candidates=[candidate],
                ),
            ]
        )

        assert plan.reserve_bytes == 7 * GIB_BYTES
        assert plan.resource_budget_bytes["gpu:rank0:request"] == 45 * GIB_BYTES
        assert plan.resource_budget_bytes["gpu:rank1:request"] == 63 * GIB_BYTES

    def test_optimizes_promotion_by_h2d_savings(self):
        # dit saves 25 GiB x 50 steps per request; text_encoder saves 30 GiB
        # once. dit must be promoted first, after which the text encoder no
        # longer fits (50 + 25 + 30 + 10 > 100).
        candidates = [
            _candidate("text_encoder", weight_gib=30),
            _candidate("dit", mode=LAYERWISE_OFFLOAD, weight_gib=25, h2d_gib=25 * 50),
        ]
        plan = plan_auto_residency(
            reports=[_report(budget_gib=100, estimated_gib=50, candidates=candidates)]
        )
        assert [c.component_name for c in plan.changes] == ["dit"]

    def test_output_rank_timing_applies_to_untimed_peer_ranks(self):
        transformer = _candidate(
            "transformer",
            mode=LAYERWISE_OFFLOAD,
            weight_gib=40,
            h2d_gib=40 * 40,
        )
        text_encoder = _candidate("text_encoder", weight_gib=8, h2d_gib=8)
        candidates = [transformer, text_encoder]
        timing = {
            transformer.option_key(): 10_000_000_000,
            text_encoder.option_key(): 83_000_000,
        }

        plan = plan_auto_residency(
            reports=[
                _report(
                    rank=0,
                    budget_gib=80,
                    estimated_gib=20,
                    candidates=candidates,
                    estimated_request_duration_ns=66_000_000_000,
                    candidate_latency_savings_ns=timing,
                ),
                _report(
                    rank=1,
                    budget_gib=80,
                    estimated_gib=20,
                    candidates=candidates,
                ),
            ]
        )

        assert [candidate.component_name for candidate in plan.changes] == [
            "transformer",
            "text_encoder",
        ]

    def test_smaller_component_still_fits_after_skipping_a_big_one(self):
        candidates = [
            _candidate("text_encoder", weight_gib=45, h2d_gib=100),
            _candidate("vae", weight_gib=5, h2d_gib=5),
        ]
        plan = plan_auto_residency(
            reports=[_report(budget_gib=100, estimated_gib=50, candidates=candidates)]
        )
        assert [c.component_name for c in plan.changes] == ["vae"]

    def test_consensus_requires_component_on_every_rank(self):
        plan = plan_auto_residency(
            reports=[
                _report(
                    rank=0,
                    candidates=[_candidate("vae"), _candidate("text_encoder")],
                ),
                _report(rank=1, candidates=[_candidate("vae", weight_gib=12)]),
            ]
        )
        names = [c.component_name for c in plan.changes]
        assert names == ["vae"]
        # worst-case size across ranks
        assert plan.changes[0].target_resident_weight_bytes == 12 * GIB_BYTES

    def test_no_candidates_skips(self):
        plan = plan_auto_residency(reports=[_report(candidates=[])])
        assert plan.skip_reason is not None

    def test_component_active_phase_is_not_double_counted(self):
        # Cosmos3-Super TP2: the 66 GiB denoise peak already includes the
        # roughly 60 GiB/rank DiT. Keeping it resident adds those bytes only
        # to the 4 GiB idle phase, matching the measured 73 GiB resident peak
        # instead of inventing a 126 GiB denoise peak.
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=139,
                    estimated_gib=66,
                    candidates=[_candidate("transformer", weight_gib=60)],
                    phase_peaks_gib={"denoise": 66, "idle": 4},
                    phase_components={"denoise": ("transformer",), "idle": ()},
                )
            ]
        )

        assert [candidate.component_name for candidate in plan.changes] == [
            "transformer"
        ]

    def test_async_prefetch_presence_is_not_charged_twice(self):
        offload = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=0,
            current_placement=True,
        )
        resident = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=42 * GIB_BYTES,
            h2d_bytes_per_request=42 * GIB_BYTES,
            inactive_device_delta_bytes=42 * GIB_BYTES,
            present_device_delta_bytes=0,
            target_device_weight_bytes=42 * GIB_BYTES,
        )
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=95,
                    estimated_gib=50,
                    target_workload_measured=True,
                    phase_peaks_gib={
                        "text_and_prefetch": 50,
                        "denoise": 50,
                    },
                    phase_components={
                        "text_and_prefetch": ("text_encoder",),
                        "denoise": ("transformer",),
                    },
                    phase_present_components={
                        "text_and_prefetch": ("text_encoder", "transformer"),
                        "denoise": ("transformer",),
                    },
                    phase_prefetched_components={
                        "text_and_prefetch": ("transformer",),
                    },
                    candidates=[offload, resident],
                )
            ]
        )

        assert [candidate.option_key() for candidate in plan.changes] == [
            "transformer:resident"
        ]

    def test_request_end_residency_is_carried_into_the_next_request(self):
        text_offload = ResidencyTarget(
            component_name="text_encoder",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=0,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((),),
            current_placement=True,
        )
        text_resident = ResidencyTarget(
            component_name="text_encoder",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=20 * GIB_BYTES,
            h2d_bytes_per_request=20 * GIB_BYTES,
            target_layerwise_resident_layers=(48,),
            target_layerwise_pinned_layers=((),),
            active_device_delta_bytes=20 * GIB_BYTES,
            inactive_device_delta_bytes=20 * GIB_BYTES,
            target_device_weight_bytes=20 * GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=80,
                    estimated_gib=40,
                    target_workload_measured=True,
                    phase_peaks_gib={"text_encode": 30, "idle": 40},
                    phase_components={
                        "text_encode": ("text_encoder",),
                        "idle": (),
                    },
                    phase_present_components={
                        "text_encode": ("text_encoder",),
                        "idle": ("transformer",),
                    },
                    component_weight_gib={"transformer": 40},
                    candidates=[text_offload, text_resident],
                )
            ]
        )

        # The first warmup encoded text before the retained DiT existed. The
        # next request starts with that 40 GiB DiT, so a resident encoder would
        # need 90 GiB before the explicit reserve and is not feasible.
        assert plan.changes == []

    def test_other_phase_can_block_component_residency(self):
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=139,
                    estimated_gib=90,
                    candidates=[_candidate("transformer", weight_gib=60)],
                    phase_peaks_gib={"denoise": 66, "decode": 90},
                    phase_components={"denoise": ("transformer",), "decode": ()},
                )
            ]
        )

        assert not plan.changes

    def test_inactive_phase_charges_full_permanent_target(self):
        layerwise = ResidencyTarget(
            component_name="text_encoder",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=0,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((),),
            current_placement=True,
        )
        resident = ResidencyTarget(
            component_name="text_encoder",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=20 * GIB_BYTES,
            h2d_bytes_per_request=20 * GIB_BYTES,
            target_layerwise_resident_layers=(48,),
            target_layerwise_pinned_layers=((),),
            permanent_residency=True,
            inactive_device_delta_bytes=20 * GIB_BYTES,
            target_device_weight_bytes=25 * GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=80,
                    estimated_gib=55,
                    target_workload_measured=True,
                    phase_peaks_gib={"denoise": 55},
                    candidates=[layerwise, resident],
                )
            ]
        )

        # The managed-layer delta would fit in the 21 GiB post-reserve
        # headroom, but the complete permanent 25 GiB footprint does not.
        assert plan.changes == []

    def test_unmaterialized_weight_update_charges_full_layerwise_target(self):
        component_offload = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=0,
            current_placement=True,
        )
        layerwise = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=GIB_BYTES,
            h2d_bytes_per_request=100 * GIB_BYTES,
            target_layerwise_resident_layers=(1,),
            target_layerwise_pinned_layers=((),),
            active_device_delta_bytes=GIB_BYTES,
            inactive_device_delta_bytes=GIB_BYTES,
            target_device_weight_bytes=GIB_BYTES,
        )
        resident = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=40 * GIB_BYTES,
            h2d_bytes_per_request=0,
            permanent_residency=True,
            inactive_device_delta_bytes=40 * GIB_BYTES,
            target_device_weight_bytes=40 * GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=90,
                    estimated_gib=50,
                    target_workload_measured=True,
                    phase_peaks_gib={"lora_switch": 50},
                    phase_present_components={
                        "lora_switch": (),
                        "idle": ("transformer",),
                    },
                    phase_full_weight_transition_components={
                        "lora_switch": ("transformer",)
                    },
                    candidates=[component_offload, layerwise, resident],
                )
            ]
        )

        assert plan.changes == []

    def test_measured_component_transition_does_not_double_count_weights(self):
        component_offload = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=0,
            current_placement=True,
        )
        layerwise = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=GIB_BYTES,
            h2d_bytes_per_request=100 * GIB_BYTES,
            target_layerwise_resident_layers=(1,),
            target_layerwise_pinned_layers=((),),
            active_device_delta_bytes=GIB_BYTES,
            inactive_device_delta_bytes=GIB_BYTES,
            target_device_weight_bytes=GIB_BYTES,
        )
        resident = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=40 * GIB_BYTES,
            h2d_bytes_per_request=0,
            permanent_residency=True,
            inactive_device_delta_bytes=40 * GIB_BYTES,
            target_device_weight_bytes=40 * GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=90,
                    estimated_gib=50,
                    target_workload_measured=True,
                    phase_peaks_gib={"lora_switch": 50},
                    phase_present_components={
                        "lora_switch": ("transformer",),
                    },
                    phase_full_weight_transition_components={
                        "lora_switch": ("transformer",)
                    },
                    candidates=[component_offload, layerwise, resident],
                )
            ]
        )

        assert [candidate.target_mode() for candidate in plan.changes] == [
            LAYERWISE_OFFLOAD
        ]

    def test_materialized_layerwise_transition_does_not_double_count_weights(self):
        current_layerwise = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=0,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((),),
            current_placement=True,
        )
        layerwise = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=GIB_BYTES,
            h2d_bytes_per_request=100 * GIB_BYTES,
            target_layerwise_resident_layers=(1,),
            target_layerwise_pinned_layers=((),),
            active_device_delta_bytes=GIB_BYTES,
            inactive_device_delta_bytes=GIB_BYTES,
            target_device_weight_bytes=GIB_BYTES,
        )
        resident = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=40 * GIB_BYTES,
            h2d_bytes_per_request=0,
            permanent_residency=True,
            inactive_device_delta_bytes=40 * GIB_BYTES,
            target_device_weight_bytes=40 * GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=90,
                    estimated_gib=50,
                    target_workload_measured=True,
                    phase_peaks_gib={"lora_switch": 50},
                    phase_full_weight_transition_components={
                        "lora_switch": ("transformer",)
                    },
                    candidates=[current_layerwise, layerwise, resident],
                )
            ]
        )

        assert [candidate.target_mode() for candidate in plan.changes] == [
            LAYERWISE_OFFLOAD
        ]

    def test_reports_reserve_shortfall_for_a_newly_calibrated_placement(self):
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=30,
                    estimated_gib=28,
                    target_workload_measured=True,
                    phase_peaks_gib={"decode": 28},
                )
            ]
        )

        assert plan.current_placement_reserve_shortfall_bytes == GIB_BYTES

    def test_validation_checks_measured_placement_without_candidate_frontier(self):
        report = _report(
            budget_gib=30,
            estimated_gib=20,
            target_workload_measured=True,
            phase_peaks_gib={"decode": 28},
        )

        assert current_placement_reserve_shortfall_bytes([report]) == GIB_BYTES

    def test_validation_preserves_reserve_beyond_allocator_mapped_footprint(self):
        report = _report(
            budget_gib=80,
            estimated_gib=60,
            observed_reserved_gib=78,
            target_workload_measured=True,
        )

        assert current_placement_reserve_shortfall_bytes([report]) == 2 * GIB_BYTES

    def test_unobserved_later_component_gets_a_conservative_phase(self):
        first_transformer = _candidate("first_transformer", weight_gib=26, h2d_gib=100)
        second_transformer = ResidencyTarget(
            component_name="second_transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=26 * GIB_BYTES,
            h2d_bytes_per_request=90 * GIB_BYTES,
            target_layerwise_resident_layers=(40,),
            target_layerwise_pinned_layers=((),),
            active_device_delta_bytes=26 * GIB_BYTES,
            inactive_device_delta_bytes=0,
        )
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=80,
                    estimated_gib=30,
                    candidates=[first_transformer, second_transformer],
                    phase_peaks_gib={"early_denoise": 30},
                    phase_components={
                        "early_denoise": ("first_transformer",),
                    },
                )
            ]
        )

        assert [candidate.component_name for candidate in plan.changes] == [
            "first_transformer"
        ]
        assert "gpu:rank0:unobserved:second_transformer" in plan.resource_budget_bytes

    def test_redundant_phase_constraints_collapse_to_the_highest_peak(self):
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=100,
                    estimated_gib=70,
                    candidates=[_candidate("transformer", weight_gib=10)],
                    phase_peaks_gib={
                        "denoise:first": 60,
                        "denoise:later": 70,
                        "decode": 50,
                    },
                    phase_components={
                        "denoise:first": ("transformer",),
                        "denoise:later": ("transformer",),
                        "decode": ("vae",),
                    },
                )
            ]
        )

        assert "gpu:rank0:denoise:first" not in plan.resource_budget_bytes
        assert plan.resource_budget_bytes["gpu:rank0:denoise:later"] == (20 * GIB_BYTES)
        assert "gpu:rank0:decode" in plan.resource_budget_bytes

    def test_validated_baseline_with_no_reserve_headroom_accepts_zero_delta(self):
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=80,
                    estimated_gib=76,
                    candidates=[_candidate("transformer", weight_gib=10)],
                    phase_peaks_gib={"denoise": 76},
                    phase_components={"denoise": ("transformer",)},
                )
            ]
        )

        assert [candidate.component_name for candidate in plan.changes] == [
            "transformer"
        ]
        assert plan.resource_budget_bytes["gpu:rank0:denoise"] == 0

    def test_each_rank_keeps_a_reserve_scaled_to_its_own_gpu(self):
        candidate = _candidate("vae", weight_gib=1)
        plan = plan_auto_residency(
            reports=[
                _report(
                    rank=0,
                    budget_gib=40,
                    estimated_gib=20,
                    candidates=[candidate],
                    phase_peaks_gib={"decode": 20},
                    phase_components={"decode": ("vae",)},
                ),
                _report(
                    rank=1,
                    budget_gib=140,
                    estimated_gib=100,
                    candidates=[candidate],
                    phase_peaks_gib={"decode": 100},
                    phase_components={"decode": ("vae",)},
                ),
            ]
        )

        assert plan.resource_budget_bytes["gpu:rank0:decode"] == 16 * GIB_BYTES
        assert plan.resource_budget_bytes["gpu:rank1:decode"] == 26 * GIB_BYTES
        assert plan.reserve_bytes == 14 * GIB_BYTES

    def test_partial_layer_residency_is_selected_when_full_dit_does_not_fit(self):
        full = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=30 * GIB_BYTES,
            h2d_bytes_per_request=300 * GIB_BYTES,
            target_layerwise_resident_layers=(30,),
            target_layerwise_pinned_layers=((),),
            permanent_residency=True,
            active_device_delta_bytes=30 * GIB_BYTES,
            inactive_device_delta_bytes=30 * GIB_BYTES,
        )
        partial = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=10 * GIB_BYTES,
            h2d_bytes_per_request=90 * GIB_BYTES,
            target_layerwise_resident_layers=(12,),
            target_layerwise_pinned_layers=((),),
            active_device_delta_bytes=10 * GIB_BYTES,
        )
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=100,
                    estimated_gib=70,
                    candidates=[full, partial],
                    phase_peaks_gib={"denoise": 70},
                    phase_components={"denoise": ("transformer",)},
                )
            ]
        )

        assert len(plan.changes) == 1
        assert plan.changes[0].target_layerwise_resident_layers == (12,)

    def test_layerwise_release_can_unlock_a_more_valuable_component(self):
        release_cold_layers = ResidencyTarget(
            component_name="transformer_2",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=-20 * GIB_BYTES,
            h2d_bytes_per_request=-20 * GIB_BYTES,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((),),
            active_device_delta_bytes=-20 * GIB_BYTES,
        )
        keep_hot_encoder = ResidencyTarget(
            component_name="text_encoder",
            residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=30 * GIB_BYTES,
            h2d_bytes_per_request=100 * GIB_BYTES,
            permanent_residency=True,
            active_device_delta_bytes=0,
            inactive_device_delta_bytes=30 * GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=100,
                    estimated_gib=80,
                    candidates=[release_cold_layers, keep_hot_encoder],
                    phase_peaks_gib={"denoise": 80, "encode": 80},
                    phase_components={
                        "denoise": ("transformer_2",),
                        "encode": ("text_encoder",),
                    },
                )
            ]
        )

        assert {candidate.component_name for candidate in plan.changes} == {
            "transformer_2",
            "text_encoder",
        }

    def test_node_hostpin_is_optimized_with_the_same_selection_vector(self):
        cold_pageable = ResidencyTarget(
            component_name="cold_encoder",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=-GIB_BYTES,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((),),
            pinned_host_delta_bytes=-10 * GIB_BYTES,
        )
        hot_pinned = ResidencyTarget(
            component_name="hot_dit",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=100 * GIB_BYTES,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((0,),),
            pinned_host_delta_bytes=10 * GIB_BYTES,
        )
        plan = plan_auto_residency(
            reports=[
                _report(
                    pinned_host_gib=10,
                    host_pin_capacity_gib=10,
                    candidates=[cold_pageable, hot_pinned],
                    estimated_request_duration_ns=1_000_000_000,
                    candidate_latency_savings_ns={
                        cold_pageable.option_key(): 10_000_000,
                        hot_pinned.option_key(): 500_000_000,
                    },
                )
            ]
        )

        # The encoder's gain is below the risk floor, but its hostpin release
        # makes the valuable DiT placement feasible. A joint solver must keep
        # this cross-component trade instead of filtering it locally.
        assert [candidate.component_name for candidate in plan.changes] == [
            "hot_dit",
            "cold_encoder",
        ]
        assert plan.resource_budget_bytes["hostpin:node0"] == 0

    def test_hostpin_repack_must_fit_transition_headroom(self):
        pin_more = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=100 * GIB_BYTES,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((0,),),
            pinned_host_delta_bytes=4 * GIB_BYTES,
            host_pin_scratch_bytes=6 * GIB_BYTES,
        )
        plan = plan_auto_residency(
            reports=[
                _report(
                    host_pin_capacity_gib=10,
                    host_transition_headroom_gib=5,
                    candidates=[pin_more],
                )
            ]
        )

        assert plan.changes == []
        assert plan.resource_budget_bytes["hostram:node0:pin"] == 5 * GIB_BYTES

    def test_component_demotion_must_fit_host_materialization_headroom(self):
        demote = ResidencyTarget(
            component_name="text_encoder",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=100 * GIB_BYTES,
            host_materialize_scratch_bytes=6 * GIB_BYTES,
            device_transition_delta_bytes=-6 * GIB_BYTES,
        )
        plan = plan_auto_residency(
            reports=[
                _report(
                    host_transition_headroom_gib=5,
                    candidates=[demote],
                )
            ]
        )

        assert plan.changes == []
        assert plan.resource_budget_bytes["hostram:node0:materialize"] == 5 * GIB_BYTES

    def test_layerwise_materialization_must_fit_transition_vram(self):
        resident = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=10 * GIB_BYTES,
            h2d_bytes_per_request=100 * GIB_BYTES,
            target_layerwise_resident_layers=(10,),
            target_layerwise_pinned_layers=((),),
            device_transition_delta_bytes=10 * GIB_BYTES,
        )
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=100,
                    estimated_gib=50,
                    device_transition_allocated_gib=90,
                    target_workload_measured=True,
                    candidates=[resident],
                )
            ]
        )

        assert plan.changes == []
        assert (
            plan.resource_budget_bytes["gpu:rank0:placement-transition"]
            == 5 * GIB_BYTES
        )

    def test_device_release_can_fund_later_layerwise_materialization(self):
        release = ResidencyTarget(
            component_name="cold_transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=-10 * GIB_BYTES,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((),),
            device_transition_delta_bytes=-10 * GIB_BYTES,
        )
        materialize = ResidencyTarget(
            component_name="hot_transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=10 * GIB_BYTES,
            h2d_bytes_per_request=100 * GIB_BYTES,
            target_layerwise_resident_layers=(10,),
            target_layerwise_pinned_layers=((),),
            device_transition_delta_bytes=10 * GIB_BYTES,
        )
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=100,
                    estimated_gib=50,
                    device_transition_allocated_gib=95,
                    target_workload_measured=True,
                    candidates=[release, materialize],
                )
            ]
        )

        assert {candidate.component_name for candidate in plan.changes} == {
            "cold_transformer",
            "hot_transformer",
        }

    def test_complete_state_frontier_can_replace_an_earlier_resident(self):
        transformer_offload = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=10 * GIB_BYTES,
            h2d_bytes_per_request=0,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((),),
            active_device_delta_bytes=-20 * GIB_BYTES,
            target_device_weight_bytes=10 * GIB_BYTES,
        )
        transformer_resident = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=30 * GIB_BYTES,
            h2d_bytes_per_request=40 * GIB_BYTES,
            target_layerwise_resident_layers=(20,),
            target_layerwise_pinned_layers=((),),
            current_placement=True,
            target_device_weight_bytes=30 * GIB_BYTES,
        )
        encoder_offload = ResidencyTarget(
            component_name="text_encoder",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=0,
            current_placement=True,
        )
        encoder_resident = ResidencyTarget(
            component_name="text_encoder",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=30 * GIB_BYTES,
            h2d_bytes_per_request=80 * GIB_BYTES,
            inactive_device_delta_bytes=30 * GIB_BYTES,
            target_device_weight_bytes=30 * GIB_BYTES,
        )
        candidates = [
            transformer_offload,
            transformer_resident,
            encoder_offload,
            encoder_resident,
        ]
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=100,
                    estimated_gib=90,
                    target_workload_measured=True,
                    phase_peaks_gib={
                        "transformer_use": 70,
                        "encoder_use": 90,
                    },
                    phase_components={
                        "transformer_use": ("transformer",),
                        "encoder_use": ("text_encoder",),
                    },
                    candidates=candidates,
                )
            ]
        )

        assert {candidate.option_key() for candidate in plan.changes} == {
            "transformer:stage:layers=0:pins=-",
            "text_encoder:resident",
        }

    def test_component_offload_can_release_an_overlapped_layerwise_prefetch(self):
        layerwise = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=10 * GIB_BYTES,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((),),
            current_placement=True,
        )
        component_offload = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=0,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((),),
            concurrent_prefetch_device_delta_bytes=-10 * GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=18,
                    estimated_gib=18,
                    warmup_oom=True,
                    candidates=[layerwise, component_offload],
                    phase_peaks_gib={"encode_prefetch": 18},
                    phase_components={"encode_prefetch": ("text_encoder",)},
                    phase_present_components={
                        "encode_prefetch": ("text_encoder", "transformer"),
                    },
                    phase_prefetched_components={
                        "encode_prefetch": ("transformer",),
                    },
                    component_active_weight_gib={
                        "text_encoder": 8,
                        "transformer": 10,
                    },
                )
            ]
        )

        assert plan.skip_reason is None
        assert [candidate.target_mode() for candidate in plan.changes] == [
            COMPONENT_OFFLOAD
        ]

    def test_dynamic_hostpin_uses_one_node_budget_and_assigns_rank_quotas(self):
        pin_more = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=100 * GIB_BYTES,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((0,),),
            pinned_host_delta_bytes=10 * GIB_BYTES,
        )
        plan = plan_auto_residency(
            reports=[
                _report(
                    rank=0,
                    host_pin_capacity_gib=20,
                    candidates=[pin_more],
                ),
                _report(
                    rank=1,
                    host_pin_capacity_gib=20,
                    candidates=[pin_more],
                ),
            ]
        )

        assert [candidate.component_name for candidate in plan.changes] == [
            "transformer"
        ]
        assert plan.resource_budget_bytes["hostpin:node0"] == 40 * GIB_BYTES
        assert plan.host_pin_target_bytes_by_rank == {
            0: 10 * GIB_BYTES,
            1: 10 * GIB_BYTES,
        }

    def test_equal_latency_plan_avoids_strategy_repacking(self):
        current = ResidencyTarget(
            component_name="vae",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=0,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((),),
            current_placement=True,
        )
        component_offload = ResidencyTarget(
            component_name="vae",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=GIB_BYTES,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((),),
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    candidates=[current, component_offload],
                    estimated_request_duration_ns=1_000_000_000,
                    candidate_latency_savings_ns={
                        current.option_key(): 0,
                        component_offload.option_key(): 0,
                    },
                )
            ]
        )

        assert plan.changes == []

    def test_failed_candidate_margin_only_constrains_future_growth(self):
        candidate = _candidate("transformer", weight_gib=5, h2d_gib=5)
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=40,
                    estimated_gib=30,
                    planning_headroom_correction_gib=3,
                    target_workload_measured=True,
                    candidates=[candidate],
                    phase_peaks_gib={"denoise": 30},
                )
            ]
        )

        assert plan.current_placement_reserve_shortfall_bytes == 0
        assert plan.resource_budget_bytes["gpu:rank0:denoise"] == 3 * GIB_BYTES
        assert plan.changes == []

    def test_free_host_pin_gain_is_selected(self):
        pageable = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=5 * GIB_BYTES,
            h2d_bytes_per_request=10 * GIB_BYTES,
            target_layerwise_resident_layers=(5,),
            target_layerwise_pinned_layers=((),),
            target_device_weight_bytes=5 * GIB_BYTES,
            current_placement=True,
        )
        pinned = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=5 * GIB_BYTES,
            h2d_bytes_per_request=11 * GIB_BYTES,
            target_layerwise_resident_layers=(5,),
            target_layerwise_pinned_layers=((0,),),
            pinned_host_delta_bytes=GIB_BYTES,
            target_device_weight_bytes=5 * GIB_BYTES,
            target_pinned_host_bytes=GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=100,
                    estimated_gib=30,
                    candidates=[pageable, pinned],
                    host_pin_capacity_gib=10,
                    estimated_request_duration_ns=1_000_000_000,
                    candidate_latency_savings_ns={
                        pageable.option_key(): 100_000_000,
                        pinned.option_key(): 110_000_000,
                    },
                )
            ]
        )

        assert plan.changes == [pinned]

    def test_hostpin_capacity_is_scoped_per_node(self):
        pin_more = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=100 * GIB_BYTES,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((0,),),
            pinned_host_delta_bytes=30 * GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    rank=0,
                    node_rank=0,
                    host_pin_capacity_gib=20,
                    candidates=[pin_more],
                ),
                _report(
                    rank=1,
                    node_rank=1,
                    host_pin_capacity_gib=20,
                    candidates=[pin_more],
                ),
            ]
        )

        assert plan.changes == []
        assert plan.resource_budget_bytes["hostpin:node0"] == 20 * GIB_BYTES
        assert plan.resource_budget_bytes["hostpin:node1"] == 20 * GIB_BYTES

    def test_latency_equivalent_partial_dit_prefers_strided_schedule(self):
        common = dict(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=4 * GIB_BYTES,
            h2d_bytes_per_request=10 * GIB_BYTES,
            target_layerwise_resident_layers=(8,),
            target_layerwise_pinned_layers=((),),
            target_device_weight_bytes=4 * GIB_BYTES,
        )
        leading = ResidencyTarget(
            **common,
            target_layerwise_residency_policies=(RESIDENCY_POLICY_LEADING,),
            current_placement=True,
        )
        strided = ResidencyTarget(
            **common,
            target_layerwise_residency_policies=(RESIDENCY_POLICY_STRIDED,),
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    estimated_gib=30,
                    candidates=[leading, strided],
                    target_workload_measured=True,
                )
            ]
        )

        assert len(plan.changes) == 1
        assert plan.changes[0].target_layerwise_residency_policies == (
            RESIDENCY_POLICY_STRIDED,
        )

    def test_long_request_uses_available_vram_for_encoder_gain(self):
        transformer = _candidate(
            "transformer",
            mode=LAYERWISE_OFFLOAD,
            weight_gib=40,
            h2d_gib=40 * 40,
        )
        text_encoder = _candidate("text_encoder", weight_gib=8, h2d_gib=8)
        candidates = [transformer, text_encoder]
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=80,
                    estimated_gib=20,
                    candidates=candidates,
                    estimated_request_duration_ns=66_000_000_000,
                    candidate_latency_savings_ns={
                        transformer.option_key(): 10_000_000_000,
                        text_encoder.option_key(): 83_000_000,
                    },
                )
            ]
        )

        assert [candidate.component_name for candidate in plan.changes] == [
            "transformer",
            "text_encoder",
        ]

    def test_lower_latency_utility_keeps_the_measured_placement(self):
        resident = ResidencyTarget(
            component_name="transformer",
            residency_mode=RESIDENT,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=20 * GIB_BYTES,
            h2d_bytes_per_request=20 * GIB_BYTES,
            permanent_residency=True,
            target_device_weight_bytes=20 * GIB_BYTES,
            current_placement=True,
        )
        offloaded = ResidencyTarget(
            component_name="transformer",
            residency_mode=RESIDENT,
            target_residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=19 * GIB_BYTES,
            device_transition_delta_bytes=-20 * GIB_BYTES,
            inactive_device_delta_bytes=-20 * GIB_BYTES,
            present_device_delta_bytes=-20 * GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=100,
                    estimated_gib=30,
                    candidates=[resident, offloaded],
                    estimated_request_duration_ns=1_000_000_000,
                    candidate_latency_savings_ns={
                        resident.option_key(): 500_000_000,
                        offloaded.option_key(): 475_000_000,
                    },
                )
            ]
        )

        assert plan.changes == []

    def test_material_latency_gain_can_replace_the_measured_placement(self):
        resident = ResidencyTarget(
            component_name="transformer",
            residency_mode=RESIDENT,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=20 * GIB_BYTES,
            h2d_bytes_per_request=20 * GIB_BYTES,
            permanent_residency=True,
            target_device_weight_bytes=20 * GIB_BYTES,
            current_placement=True,
        )
        offloaded = ResidencyTarget(
            component_name="transformer",
            residency_mode=RESIDENT,
            target_residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=19 * GIB_BYTES,
            device_transition_delta_bytes=-20 * GIB_BYTES,
            inactive_device_delta_bytes=-20 * GIB_BYTES,
            present_device_delta_bytes=-20 * GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=100,
                    estimated_gib=30,
                    candidates=[resident, offloaded],
                    estimated_request_duration_ns=1_000_000_000,
                    candidate_latency_savings_ns={
                        resident.option_key(): 460_000_000,
                        offloaded.option_key(): 500_000_000,
                    },
                )
            ]
        )

        assert plan.changes == [offloaded]

    def test_measured_small_card_reserve_drops_to_ten_percent(self):
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=12,
                    estimated_gib=10,
                    target_workload_measured=True,
                )
            ]
        )
        assert plan.reserve_bytes == int(1.2 * GIB_BYTES)

    def test_node_hostpin_can_assign_asymmetric_rank_quotas(self):
        rank0 = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=100 * GIB_BYTES,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((0,),),
            pinned_host_delta_bytes=30 * GIB_BYTES,
        )
        rank1 = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=100 * GIB_BYTES,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((0,),),
            pinned_host_delta_bytes=5 * GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    rank=0,
                    host_pin_capacity_gib=20,
                    candidates=[rank0],
                ),
                _report(
                    rank=1,
                    host_pin_capacity_gib=20,
                    candidates=[rank1],
                ),
            ]
        )

        assert [candidate.component_name for candidate in plan.changes] == [
            "transformer"
        ]
        assert plan.host_pin_target_bytes_by_rank == {
            0: 30 * GIB_BYTES,
            1: 5 * GIB_BYTES,
        }

    def test_oom_phase_forces_a_lower_memory_complete_target(self):
        resident = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=20 * GIB_BYTES,
            h2d_bytes_per_request=20 * GIB_BYTES,
            permanent_residency=True,
            target_device_weight_bytes=20 * GIB_BYTES,
            current_placement=True,
        )
        layerwise = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=2 * GIB_BYTES,
            h2d_bytes_per_request=10 * GIB_BYTES,
            target_layerwise_resident_layers=(2,),
            target_layerwise_pinned_layers=((),),
            device_transition_delta_bytes=-15 * GIB_BYTES,
            active_device_delta_bytes=-15 * GIB_BYTES,
            present_device_delta_bytes=-15 * GIB_BYTES,
            inactive_device_delta_bytes=-20 * GIB_BYTES,
            target_device_weight_bytes=2 * GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=30,
                    estimated_gib=29,
                    warmup_oom=True,
                    phase_peaks_gib={"denoise": 29},
                    phase_components={"denoise": ("transformer",)},
                    phase_present_components={"denoise": ("transformer",)},
                    component_weight_gib={"transformer": 20},
                    candidates=[resident, layerwise],
                )
            ]
        )

        assert plan.recovering_from_oom
        assert plan.resource_budget_bytes["gpu:rank0:denoise"] == -3 * GIB_BYTES
        assert [candidate.option_key() for candidate in plan.changes] == [
            layerwise.option_key()
        ]

    def test_oom_recovery_state_survives_an_infeasible_plan(self):
        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=30,
                    estimated_gib=29,
                    warmup_oom=True,
                    phase_peaks_gib={"denoise": 29},
                    candidates=[_candidate("vae")],
                )
            ]
        )

        assert plan.recovering_from_oom
        assert plan.skip_reason == "no placement satisfies all resource budgets"
        assert not plan.changes

    def test_uncalibrated_small_card_reserve_has_three_gib_floor(self):
        plan = plan_auto_residency(reports=[_report(budget_gib=12, estimated_gib=5)])
        assert plan.reserve_bytes == 3 * GIB_BYTES

    def test_unobserved_phase_excludes_another_components_transient_weights(self):
        transformer_component = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=0,
            current_placement=True,
        )
        transformer_layerwise = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=10 * GIB_BYTES,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((),),
            active_device_delta_bytes=-10 * GIB_BYTES,
            inactive_device_delta_bytes=GIB_BYTES,
        )
        vae = ResidencyTarget(
            component_name="vae",
            residency_mode=RESIDENT,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=GIB_BYTES,
            h2d_bytes_per_request=GIB_BYTES,
            current_placement=True,
            permanent_residency=True,
            target_device_weight_bytes=GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=12,
                    estimated_gib=11.75,
                    warmup_oom=True,
                    candidates=[transformer_component, transformer_layerwise, vae],
                    phase_peaks_gib={
                        "setup": 5,
                        "prefetch:transformer": 11.75,
                    },
                    phase_components={
                        "prefetch:transformer": (),
                    },
                    phase_present_components={
                        "setup": ("vae",),
                        "prefetch:transformer": ("transformer", "vae"),
                    },
                    phase_prefetched_components={
                        "prefetch:transformer": ("transformer",),
                    },
                    component_weight_gib={"vae": 1},
                    component_active_weight_gib={"transformer": 11, "vae": 1},
                )
            ]
        )

        assert plan.skip_reason is None
        assert [candidate.target_mode() for candidate in plan.changes] == [
            LAYERWISE_OFFLOAD
        ]
        assert plan.resource_budget_bytes["gpu:rank0:unobserved:vae"] == (
            12 * GIB_BYTES - 5 * GIB_BYTES - plan.reserve_bytes
        )

    def test_untracked_oom_peak_constrains_synthetic_component_phase(self):
        component_offload = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=0,
            current_placement=True,
        )
        layerwise = ResidencyTarget(
            component_name="transformer",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=GIB_BYTES,
            h2d_bytes_per_request=10 * GIB_BYTES,
            target_layerwise_resident_layers=(0,),
            target_layerwise_pinned_layers=((),),
            active_device_delta_bytes=-10 * GIB_BYTES,
            inactive_device_delta_bytes=GIB_BYTES,
        )

        plan = plan_auto_residency(
            reports=[
                _report(
                    budget_gib=12,
                    estimated_gib=11.75,
                    phase_peaks_gib={"request:untracked": 11.75},
                    phase_present_components={"request:untracked": ("vae",)},
                    candidates=[component_offload, layerwise],
                    warmup_oom=True,
                )
            ]
        )

        assert [candidate.target_mode() for candidate in plan.changes] == [
            LAYERWISE_OFFLOAD
        ]

    def test_validation_keeps_conservative_reserve_after_warmup_oom(self):
        report = _report(
            budget_gib=30,
            estimated_gib=20,
            target_workload_measured=True,
            phase_peaks_gib={"decode": 28},
            warmup_oom=True,
        )

        assert current_placement_reserve_shortfall_bytes([report]) == 2 * GIB_BYTES


class _FakeLayerwiseManager:
    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        resident_layers: int = 0,
        layers_attr_str: str = "layers",
        event_log: list[str] | None = None,
        event_name: str = "manager",
    ):
        self.enabled = True
        self._configured = True
        self._tensors = tensors
        self.layers_attr_str = layers_attr_str
        self.num_layers = max(1, len(tensors))
        self.resident_layers = resident_layers
        self.residency_policy = "leading"
        self.pin_cpu_memory = True
        self._pinned_layers: tuple[int, ...] = ()
        self._event_log = event_log
        self._event_name = event_name
        self.fail_load = False
        self.load_all_layers_calls = 0
        self.remove_hooks_calls = 0
        self.register_hooks_calls = 0
        self.sync_to_cpu_calls = 0
        self.release_all_calls = 0
        self.release_host_stores_calls = 0

    def iter_cpu_weights(self):
        yield from self._tensors.items()

    def offloaded_weight_bytes(self):
        return sum(
            tensor.numel() * tensor.element_size() for tensor in self._tensors.values()
        )

    def resident_weight_bytes(self, resident_layers=None, residency_policy=None):
        del residency_policy
        count = self.resident_layers if resident_layers is None else resident_layers
        return (
            self.offloaded_weight_bytes()
            * min(count, self.num_layers)
            // self.num_layers
        )

    def peak_managed_device_weight_bytes(
        self, resident_layers=None, residency_policy=None
    ):
        del residency_policy
        count = self.resident_layers if resident_layers is None else resident_layers
        return (
            self.offloaded_weight_bytes()
            * min(count + 1, self.num_layers)
            // self.num_layers
        )

    def layer_weight_bytes(self):
        total = self.offloaded_weight_bytes()
        base, remainder = divmod(total, self.num_layers)
        return {
            layer_idx: base + (1 if layer_idx < remainder else 0)
            for layer_idx in range(self.num_layers)
        }

    def layer_host_store_bytes(self):
        return self.layer_weight_bytes()

    def pinned_host_weight_bytes(self):
        layer_bytes = self.layer_weight_bytes()
        return sum(layer_bytes[layer_idx] for layer_idx in self._pinned_layers)

    def pinnable_layer_indices(self):
        return tuple(range(self.num_layers))

    def pinned_layer_indices(self):
        return self._pinned_layers

    def set_pinned_layers(self, pinned_layers):
        previous = self._pinned_layers
        target = tuple(sorted(pinned_layers))
        if target != previous and self._event_log is not None:
            action = "pin" if target else "unpin"
            self._event_log.append(f"{self._event_name}:{action}")
        self._pinned_layers = target
        return previous

    def set_resident_layers(self, resident_layers):
        previous, _ = self.set_residency_layout(resident_layers, self.residency_policy)
        return previous

    def set_residency_layout(self, resident_layers, residency_policy):
        previous = (self.resident_layers, self.residency_policy)
        self.resident_layers = min(max(0, resident_layers), self.num_layers)
        self.residency_policy = residency_policy
        return previous

    def load_all_layers(self):
        self.load_all_layers_calls += 1
        if self._event_log is not None:
            self._event_log.append(f"{self._event_name}:load")
        if self.fail_load:
            raise RuntimeError("CUDA out of memory")

    def remove_forward_hooks(self):
        self.remove_hooks_calls += 1

    def register_forward_hooks(self):
        self.register_hooks_calls += 1

    def sync_all_layers_to_cpu(self):
        self.sync_to_cpu_calls += 1
        if self._event_log is not None:
            self._event_log.append(f"{self._event_name}:sync")

    def release_all(self):
        self.release_all_calls += 1

    def release_host_stores(self):
        if self.enabled:
            raise RuntimeError("offload is still enabled")
        self.release_host_stores_calls += 1
        self._pinned_layers = ()


class _FakeLayerwiseDit(LayerwiseOffloadableModuleMixin, nn.Module):
    def __init__(self, managers: list[_FakeLayerwiseManager]):
        nn.Module.__init__(self)
        self.layerwise_offload_managers = managers


class _FakeLazyLayerwiseDit(LayerwiseOffloadableModuleMixin, nn.Module):
    layer_names = ["layers"]

    def __init__(self, num_layers: int = 4):
        nn.Module.__init__(self)
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(num_layers))
        self.layerwise_offload_managers = []

    def configure_layerwise_offload(
        self,
        server_args,
        *,
        pin_budget=None,
        component_name=None,
        pin_during_initialization=True,
    ):
        del server_args, pin_budget, component_name, pin_during_initialization
        tensors = {
            f"layers.{index}.weight": layer.weight
            for index, layer in enumerate(self.layers)
        }
        self.layerwise_offload_managers = [_FakeLayerwiseManager(tensors)]


class _StubResidencyArgs:
    """Duck-typed stand-in for the two ServerArgs hooks adjustments use."""

    def __init__(self, host_pin_budget=None):
        self.auto_modes: dict[str, str] = {}
        self._host_pin_budget = host_pin_budget

    def auto_residency_mode(self, component_name):
        return self.auto_modes.get(component_name)

    def residency_mode(self, component_name):
        return self.auto_modes.get(component_name, COMPONENT_OFFLOAD)

    def set_auto_residency_mode(self, component_name, mode):
        self.auto_modes[component_name] = mode

    def clear_auto_residency_mode(self, component_name):
        self.auto_modes.pop(component_name, None)

    def host_pin_budget(self):
        return self._host_pin_budget


def _plan_for(candidates: list[ResidencyTarget]) -> AutoResidencyPlan:
    return AutoResidencyPlan(
        estimated_peak_bytes=GIB_BYTES,
        reserve_bytes=MIN_VRAM_RESERVE_BYTES,
        budget_bytes=100 * GIB_BYTES,
        changes=candidates,
    )


class TestSizeAccounting:
    def test_component_offload_counts_params_and_buffers(self):
        module = nn.Linear(8, 8)
        module.register_buffer("stats", torch.zeros(4))
        expected = sum(t.numel() * t.element_size() for t in module.parameters()) + sum(
            t.numel() * t.element_size() for t in module.buffers()
        )
        assert component_resident_size_bytes(module, COMPONENT_OFFLOAD) == expected

    def test_layerwise_counts_manager_cpu_buffers_not_placeholders(self):
        cpu_weights = {"layers.0.w": torch.zeros(1024), "layers.1.w": torch.zeros(1024)}
        module = _FakeLayerwiseDit([_FakeLayerwiseManager(cpu_weights)])
        expected = sum(t.numel() * t.element_size() for t in cpu_weights.values())
        assert component_resident_size_bytes(module, LAYERWISE_OFFLOAD) == expected

    def test_layerwise_runtime_floor_uses_only_the_streaming_window(self):
        manager = _FakeLayerwiseManager(
            {
                "layers.0.w": torch.zeros(16),
                "layers.1.w": torch.zeros(16),
                "layers.2.w": torch.zeros(16),
            }
        )
        module = _FakeLayerwiseDit([manager])
        module.register_parameter("unmanaged", nn.Parameter(torch.zeros(8)))

        runtime = component_runtime_weight_bytes({"transformer": module})

        assert runtime["transformer"] == (
            module.unmanaged.numel() * module.unmanaged.element_size()
            + manager.peak_managed_device_weight_bytes()
        )


class TestCollectResidencyTargets:
    def _modes(self, mapping):
        return lambda name: mapping.get(name, RESIDENT)

    def test_resident_frontier_includes_absolute_and_ratio_cli_states(self):
        managers = [
            _FakeLayerwiseManager(
                {f"layers.{index}.w": torch.zeros(16) for index in range(4)}
            ),
            _FakeLayerwiseManager(
                {f"layers.{index}.w": torch.zeros(16) for index in range(8)}
            ),
        ]

        targets = _layerwise_resident_targets(managers)

        assert (2, 2) in targets
        assert (1, 2) in targets

    def test_measured_resident_frontier_allocates_repeated_groups_independently(self):
        managers = [
            _FakeLayerwiseManager(
                {f"first.{index}.w": torch.zeros(16) for index in range(3)}
            ),
            _FakeLayerwiseManager(
                {f"second.{index}.w": torch.zeros(16) for index in range(2)}
            ),
        ]

        targets = _layerwise_resident_targets(
            managers,
            layer_uses=((8, 8, 8), (8, 8)),
        )

        assert (0, 2) in targets
        assert (3, 0) in targets

    def test_measured_resident_frontier_streams_one_shot_and_unused_groups(self):
        managers = [
            _FakeLayerwiseManager(
                {f"encoder.{index}.w": torch.zeros(16) for index in range(3)}
            ),
            _FakeLayerwiseManager(
                {f"decoder.{index}.w": torch.zeros(16) for index in range(2)}
            ),
        ]

        targets = _layerwise_resident_targets(
            managers,
            layer_uses=((0, 0, 0), (1, 1)),
        )

        assert targets == [(0, 0)]

    def test_host_pin_frontier_includes_non_prefix_packings(self):
        manager = SimpleNamespace(
            num_layers=3,
            residency_policy="leading",
            pin_cpu_memory=True,
            layer_weight_bytes=lambda: {0: 6, 1: 4, 2: 4},
            layer_host_store_bytes=lambda: {0: 6, 1: 4, 2: 4},
            pinnable_layer_indices=lambda: (0, 1, 2),
        )

        targets = _layerwise_pin_targets(
            managers=[manager],
            resident_layers=(0,),
            current_pinned_layers=((),),
            uses_per_streamed_layer=1,
        )

        assert ((0,),) in targets
        assert ((1, 2),) in targets

    def test_host_pin_frontier_coalesces_equal_transformer_layers(self):
        layer_bytes = {index: 8 for index in range(40)}
        manager = SimpleNamespace(
            num_layers=40,
            residency_policy="leading",
            pin_cpu_memory=True,
            layer_weight_bytes=lambda: layer_bytes,
            layer_host_store_bytes=lambda: layer_bytes,
            pinnable_layer_indices=lambda: tuple(layer_bytes),
        )

        targets = _layerwise_pin_targets(
            managers=[manager],
            resident_layers=(0,),
            current_pinned_layers=((),),
            uses_per_streamed_layer=20,
        )

        assert len(targets) == 41

    def test_nonbinding_host_resources_keep_only_maximum_pin_utility(self):
        manager = SimpleNamespace(
            num_layers=3,
            residency_policy="leading",
            pin_cpu_memory=True,
            layer_weight_bytes=lambda: {0: 6, 1: 4, 2: 4},
            layer_host_store_bytes=lambda: {0: 6, 1: 4, 2: 4},
            pinnable_layer_indices=lambda: (0, 1, 2),
        )

        targets = _layerwise_pin_targets(
            managers=[manager],
            resident_layers=(1,),
            current_pinned_layers=((0, 1, 2),),
            uses_per_streamed_layer=1,
            layer_uses=((0, 3, 1),),
            constrain_host_transitions=False,
            maximum_utility_only=True,
        )

        assert targets == [((1, 2),)]

    def test_collapsed_pin_frontier_retains_the_measured_state(self):
        manager = _FakeLayerwiseManager(
            {f"layers.{index}.w": torch.zeros(16) for index in range(3)}
        )
        manager._pinned_layers = (0,)
        module = _FakeLayerwiseDit([manager])

        candidates = collect_residency_targets(
            modules={"transformer": module},
            residency_mode_of=lambda _name: LAYERWISE_OFFLOAD,
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=3,
            used_components={"transformer"},
            layerwise_layer_uses={"transformer": {"layers": (3, 3, 3)}},
            host_transition_headroom_bytes=GIB_BYTES,
            host_pin_headroom_bytes=GIB_BYTES,
            request_duration_ns=1_000_000_000,
        )

        current = [candidate for candidate in candidates if candidate.current_placement]
        assert len(candidates) <= 8
        assert len(current) == 1
        assert current[0].target_layerwise_pinned_layers == ((0,),)

    def test_host_pin_frontier_drops_scratch_only_tradeoffs_when_nonbinding(self):
        manager = SimpleNamespace(
            num_layers=3,
            residency_policy="leading",
            pin_cpu_memory=True,
            layer_weight_bytes=lambda: {0: 6, 1: 4, 2: 4},
            layer_host_store_bytes=lambda: {0: 6, 1: 4, 2: 4},
            pinnable_layer_indices=lambda: (0, 1, 2),
        )

        constrained = _layerwise_pin_targets(
            managers=[manager],
            resident_layers=(0,),
            current_pinned_layers=((0, 1, 2),),
            uses_per_streamed_layer=1,
            layer_uses=((2, 1, 1),),
        )
        unconstrained = _layerwise_pin_targets(
            managers=[manager],
            resident_layers=(0,),
            current_pinned_layers=((0, 1, 2),),
            uses_per_streamed_layer=1,
            layer_uses=((2, 1, 1),),
            constrain_host_transitions=False,
        )

        assert ((1, 2),) in constrained
        assert ((1, 2),) not in unconstrained
        assert ((0,),) in unconstrained

    def test_filters_and_sizes(self):
        te = nn.Linear(8, 8)
        vae = nn.Linear(4, 4)
        dit = _FakeLayerwiseDit(
            [_FakeLayerwiseManager({"layers.0.w": torch.zeros(1024)})]
        )
        modules = {
            "text_encoder": te,
            "vae": vae,
            "transformer": dit,
            "explicit_encoder": nn.Linear(2, 2),
            "custom": nn.Linear(2, 2),
            "scheduler": object(),
        }
        modes = self._modes(
            {
                "text_encoder": COMPONENT_OFFLOAD,
                "transformer": LAYERWISE_OFFLOAD,
                "explicit_encoder": COMPONENT_OFFLOAD,
                "custom": COMPONENT_OFFLOAD,
            }
        )
        candidates = collect_residency_targets(
            modules=modules,
            residency_mode_of=modes,
            explicit_residency_mode_of=lambda name: (
                COMPONENT_OFFLOAD if name == "explicit_encoder" else None
            ),
            custom_strategy_names={"custom"},
            num_inference_steps=50,
        )
        permanent_candidates = {
            candidate.component_name: candidate
            for candidate in candidates
            if candidate.permanent_residency
        }
        # Unset resident components remain tunable. Explicit placement,
        # custom strategy, and non-module objects stay outside the frontier.
        assert set(permanent_candidates) == {"text_encoder", "transformer", "vae"}
        assert permanent_candidates["vae"].current_placement
        assert permanent_candidates["text_encoder"].h2d_bytes_per_request == (
            permanent_candidates["text_encoder"].target_resident_weight_bytes
        )
        # The all-pageable layerwise DiT re-streams its layers every step.
        assert permanent_candidates["transformer"].h2d_bytes_per_request == (
            permanent_candidates["transformer"].target_resident_weight_bytes
            * 50
            * PAGEABLE_H2D_COST_MULTIPLIER
        )
        assert any(
            candidate.component_name == "transformer"
            and candidate.target_layerwise_resident_layers == (1,)
            for candidate in candidates
        )

    def test_mechanism_mismatch_is_not_a_candidate(self):
        # A module the compile window manages layerwise while its configured
        # mode says component-offload: adjustment would be a silent no-op.
        mismatch = _FakeLayerwiseDit(
            [_FakeLayerwiseManager({"layers.0.w": torch.zeros(16)})]
        )
        candidates = collect_residency_targets(
            modules={"mismatch": mismatch},
            residency_mode_of=self._modes({"mismatch": COMPONENT_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=10,
        )
        assert candidates == []

    def test_layerwise_dit_can_release_existing_resident_layers(self):
        manager = _FakeLayerwiseManager(
            {
                "layers.0.w": torch.zeros(16),
                "layers.1.w": torch.zeros(16),
                "layers.2.w": torch.zeros(16),
                "layers.3.w": torch.zeros(16),
            },
            resident_layers=3,
        )
        module = _FakeLayerwiseDit([manager])

        candidates = collect_residency_targets(
            modules={"transformer": module},
            residency_mode_of=self._modes({"transformer": LAYERWISE_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=10,
        )

        target = next(
            candidate
            for candidate in candidates
            if not candidate.permanent_residency
            and candidate.target_layerwise_resident_layers == (1,)
            and candidate.target_layerwise_pinned_layers == ((),)
        )
        assert target.target_resident_weight_bytes > 0
        assert target.active_device_delta_bytes < 0
        assert target.h2d_bytes_per_request > 0

    def test_fully_pinned_component_can_release_hostpin_for_a_hotter_component(self):
        manager = _FakeLayerwiseManager(
            {
                "layers.0.w": torch.zeros(16),
                "layers.1.w": torch.zeros(16),
            }
        )
        manager._pinned_layers = (0, 1)
        module = _FakeLayerwiseDit([manager])

        candidates = collect_residency_targets(
            modules={"text_encoder": module},
            residency_mode_of=self._modes({"text_encoder": LAYERWISE_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=10,
        )

        assert any(
            candidate.target_layerwise_pinned_layers == ((),)
            and candidate.pinned_host_delta_bytes < 0
            and candidate.host_unpin_scratch_bytes > 0
            for candidate in candidates
        )

    def test_multi_worker_candidate_keeps_existing_hostpin_placement(self):
        manager = _FakeLayerwiseManager(
            {
                "layers.0.w": torch.zeros(16),
                "layers.1.w": torch.zeros(16),
            }
        )
        manager._pinned_layers = (0, 1)
        module = _FakeLayerwiseDit([manager])

        candidates = collect_residency_targets(
            modules={"transformer": module},
            residency_mode_of=self._modes({"transformer": LAYERWISE_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=10,
            allow_host_pin_reallocation=False,
        )
        permanent = next(
            candidate for candidate in candidates if candidate.permanent_residency
        )

        assert permanent.target_layerwise_pinned_layers == ((0, 1),)
        assert permanent.pinned_host_delta_bytes == 0

    def test_mixed_dtype_layerwise_component_is_not_permanently_promoted(self):
        manager = _FakeLayerwiseManager(
            {
                "layers.0.w": torch.zeros(16),
                "layers.1.w": torch.zeros(16),
            }
        )
        module = _FakeLayerwiseDit([manager])

        candidates = collect_residency_targets(
            modules={"vae": module},
            residency_mode_of=self._modes({"vae": LAYERWISE_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=10,
            mixed_dtype_components={"vae"},
        )

        assert candidates
        assert {candidate.target_mode() for candidate in candidates} == {
            LAYERWISE_OFFLOAD
        }

    def test_non_dit_layerwise_frontier_includes_partial_residency(self):
        manager = _FakeLayerwiseManager(
            {f"layers.{index}.w": torch.zeros(16) for index in range(4)}
        )
        module = _FakeLayerwiseDit([manager])

        candidates = collect_residency_targets(
            modules={"text_encoder": module},
            residency_mode_of=self._modes({"text_encoder": LAYERWISE_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=1,
        )

        assert any(
            candidate.target_layerwise_resident_layers == (2,)
            and candidate.target_mode() == LAYERWISE_OFFLOAD
            for candidate in candidates
        )
        component = next(
            candidate
            for candidate in candidates
            if candidate.target_mode() == COMPONENT_OFFLOAD
        )
        assert component.target_layerwise_pinned_layers == ((),)
        assert component.pinned_host_delta_bytes == 0

    def test_measured_vae_frontier_does_not_spend_on_unused_encoder_layers(self):
        encoder = _FakeLayerwiseManager(
            {f"encoder.{index}.w": torch.zeros(16) for index in range(2)},
            layers_attr_str="encoder",
        )
        decoder = _FakeLayerwiseManager(
            {f"decoder.{index}.w": torch.zeros(16) for index in range(3)},
            layers_attr_str="decoder",
        )
        module = _FakeLayerwiseDit([encoder, decoder])

        candidates = collect_residency_targets(
            modules={"vae": module},
            residency_mode_of=self._modes({"vae": LAYERWISE_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=20,
            used_components={"vae"},
            layerwise_layer_uses={
                "vae": {
                    "encoder": (0, 0),
                    "decoder": (1, 1, 1),
                }
            },
        )
        layerwise = [
            candidate
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
        ]

        assert layerwise
        assert {
            candidate.target_layerwise_resident_layers for candidate in layerwise
        } == {(0, 0)}
        assert all(
            candidate.target_layerwise_pinned_layers[0] == () for candidate in layerwise
        )

    def test_component_offload_frontier_can_expand_layerwise_lazily(self):
        module = _FakeLazyLayerwiseDit(num_layers=4)

        candidates = collect_residency_targets(
            modules={"transformer": module},
            residency_mode_of=self._modes({"transformer": COMPONENT_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=10,
            layerwise_tuning_of=lambda _name, _dit_group: (0.0, 0.0, "leading"),
        )

        component = next(
            candidate
            for candidate in candidates
            if candidate.target_mode() == COMPONENT_OFFLOAD
        )
        full_stage = next(
            candidate
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
            and candidate.target_layerwise_resident_layers == (4,)
        )
        assert component.current_placement
        assert full_stage.h2d_bytes_per_request == component.h2d_bytes_per_request
        assert all(
            candidate.h2d_bytes_per_request <= component.h2d_bytes_per_request
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
        )
        assert any(
            candidate.target_layerwise_resident_layers == (2,)
            for candidate in candidates
        )
        resident = next(
            candidate for candidate in candidates if candidate.target_mode() == RESIDENT
        )
        assert (
            resident.device_transition_delta_bytes
            == resident.target_device_weight_bytes
        )
        assert (
            resident.concurrent_prefetch_delta() == resident.active_device_delta_bytes
        )

    def test_virtual_layerwise_prefetch_is_runtime_not_transition_memory(self):
        module = _FakeLazyLayerwiseDit(num_layers=3)
        candidates = collect_residency_targets(
            modules={"transformer": module},
            residency_mode_of=self._modes({"transformer": COMPONENT_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=4,
            layerwise_tuning_of=lambda _name, _dit_group: (1.0, 0.0, "leading"),
        )

        streamed = next(
            candidate
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
            and candidate.target_layerwise_resident_layers == (0,)
        )
        assert streamed.device_transition_delta_bytes == 0
        assert streamed.active_device_delta_bytes < 0
        assert (
            streamed.concurrent_prefetch_delta() == streamed.active_device_delta_bytes
        )

    def test_lazy_layerwise_frontier_keeps_buffers_out_of_managed_weights(self):
        module = _FakeLazyLayerwiseDit(num_layers=2)
        module.layers[0].register_buffer("cache", torch.zeros(4096))

        candidates = collect_residency_targets(
            modules={"transformer": module},
            residency_mode_of=self._modes({"transformer": COMPONENT_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=10,
            layerwise_tuning_of=lambda _name, _dit_group: (0.0, 0.0, "leading"),
        )

        full_stage = next(
            candidate
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
            and candidate.target_layerwise_resident_layers == (2,)
        )
        # Each Linear packs a 16-element weight and 4-element bias. The buffer
        # remains resident and is not copied into the layerwise store.
        assert full_stage.target_resident_weight_bytes == 2 * 20 * 4
        assert full_stage.target_device_weight_bytes == (
            full_stage.target_resident_weight_bytes
            + module.layers[0].cache.untyped_storage().nbytes()
        )

    def test_component_weight_footprint_deduplicates_tied_storage(self):
        module = nn.Module()
        backing = torch.empty(1024)
        module.register_parameter("left", nn.Parameter(backing[:512]))
        module.register_parameter("right", nn.Parameter(backing[512:]))

        assert component_runtime_weight_bytes({"transformer": module}) == {
            "transformer": backing.untyped_storage().nbytes()
        }

    def test_layerwise_frontier_accounts_for_immediate_materialization(self):
        manager = _FakeLayerwiseManager(
            {
                "layers.0.w": torch.zeros(16),
                "layers.1.w": torch.zeros(16),
            }
        )
        module = _FakeLayerwiseDit([manager])

        candidates = collect_residency_targets(
            modules={"transformer": module},
            residency_mode_of=self._modes({"transformer": LAYERWISE_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=10,
        )
        resident = next(
            candidate for candidate in candidates if candidate.permanent_residency
        )
        component = next(
            candidate
            for candidate in candidates
            if candidate.target_mode() == COMPONENT_OFFLOAD
        )
        assert (
            component.concurrent_prefetch_delta() == component.active_device_delta_bytes
        )
        assert (
            resident.device_transition_delta_bytes
            == resident.target_device_weight_bytes
        )
        module.disable_offload()
        module.register_parameter(
            "materialized",
            nn.Parameter(torch.zeros(manager.offloaded_weight_bytes() // 4)),
        )
        candidates = collect_residency_targets(
            modules={"transformer": module},
            residency_mode_of=self._modes({"transformer": RESIDENT}),
            baseline_residency_mode_of=self._modes({"transformer": LAYERWISE_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=10,
        )
        streamed = next(
            candidate
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
        )
        assert (
            streamed.device_transition_delta_bytes == -manager.offloaded_weight_bytes()
        )
        component = next(
            candidate
            for candidate in candidates
            if candidate.target_mode() == COMPONENT_OFFLOAD
        )
        assert (
            component.device_transition_delta_bytes
            == -resident.target_device_weight_bytes
        )

    def test_auto_dit_frontier_tunes_each_layer_group_policy(self):
        managers = [
            _FakeLayerwiseManager(
                {f"layers.{index}.w": torch.zeros(16) for index in range(4)},
                resident_layers=2,
            )
            for _ in range(2)
        ]
        module = _FakeLayerwiseDit(managers)

        candidates = collect_residency_targets(
            modules={"transformer": module},
            residency_mode_of=self._modes({"transformer": LAYERWISE_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=10,
            layerwise_policy_is_explicit=lambda _name, _dit_group: False,
        )

        policies = {
            candidate.target_layerwise_residency_policies
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
            and candidate.target_layerwise_resident_layers == (2, 2)
            and candidate.target_layerwise_pinned_layers == ((), ())
        }
        assert policies == {
            (RESIDENCY_POLICY_LEADING, RESIDENCY_POLICY_LEADING),
            (RESIDENCY_POLICY_LEADING, RESIDENCY_POLICY_STRIDED),
            (RESIDENCY_POLICY_STRIDED, RESIDENCY_POLICY_LEADING),
            (RESIDENCY_POLICY_STRIDED, RESIDENCY_POLICY_STRIDED),
        }

    def test_auto_dit_frontier_tunes_policy_unless_explicit(self):
        manager = _FakeLayerwiseManager(
            {f"layers.{index}.w": torch.zeros(16) for index in range(4)},
            resident_layers=2,
        )
        module = _FakeLayerwiseDit([manager])
        common = dict(
            modules={"transformer": module},
            residency_mode_of=self._modes({"transformer": LAYERWISE_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=10,
        )

        candidates = collect_residency_targets(
            **common,
            layerwise_policy_is_explicit=lambda _name, _dit_group: False,
        )
        policies = {
            candidate.target_layerwise_residency_policies
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
            and candidate.target_layerwise_resident_layers == (2,)
            and candidate.target_layerwise_pinned_layers == ((),)
        }
        assert policies == {
            (RESIDENCY_POLICY_LEADING,),
            (RESIDENCY_POLICY_STRIDED,),
        }

        explicit_candidates = collect_residency_targets(
            **common,
            layerwise_policy_is_explicit=lambda _name, _dit_group: True,
        )
        assert all(
            candidate.target_layerwise_residency_policies is None
            for candidate in explicit_candidates
        )

    def test_component_latency_cap_keeps_the_full_pin_frontier(self):
        manager = _FakeLayerwiseManager(
            {f"layers.{index}.w": torch.zeros(16) for index in range(3)}
        )
        layer_bytes = {index: GIB_BYTES for index in range(3)}
        manager.layer_weight_bytes = lambda: layer_bytes
        manager.layer_host_store_bytes = lambda: layer_bytes
        module = _FakeLayerwiseDit([manager])

        candidates = collect_residency_targets(
            modules={"transformer": module},
            residency_mode_of=lambda _name: LAYERWISE_OFFLOAD,
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=3,
            used_components={"transformer"},
            layerwise_layer_uses={"transformer": {"layers": (3, 3, 3)}},
            host_transition_headroom_bytes=16 * GIB_BYTES,
            host_pin_headroom_bytes=16 * GIB_BYTES,
            request_duration_ns=1_000_000_000,
            latency_upper_bound_ns_by_component={"transformer": 1},
        )

        assert len(candidates) > 8

    def test_default_resident_component_keeps_offload_as_a_target(self):
        module = nn.Linear(4, 4)
        candidates = collect_residency_targets(
            modules={"text_encoder": module},
            residency_mode_of=self._modes({"text_encoder": RESIDENT}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=1,
        )

        by_mode = {candidate.target_mode(): candidate for candidate in candidates}
        assert set(by_mode) == {COMPONENT_OFFLOAD, RESIDENT}
        assert by_mode[RESIDENT].current_placement
        assert by_mode[COMPONENT_OFFLOAD].inactive_device_delta_bytes < 0
        assert by_mode[COMPONENT_OFFLOAD].device_transition_delta_bytes < 0
        assert by_mode[COMPONENT_OFFLOAD].host_materialize_scratch_bytes > 0

    def test_default_resident_dit_exposes_complete_layerwise_frontier(self):
        module = _FakeLazyLayerwiseDit(num_layers=3)
        candidates = collect_residency_targets(
            modules={"transformer_2": module},
            residency_mode_of=self._modes({"transformer_2": RESIDENT}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=4,
            layerwise_tuning_of=lambda _name, _dit_group: (1.0, 0.0, "leading"),
        )

        assert {candidate.target_mode() for candidate in candidates} == {
            COMPONENT_OFFLOAD,
            LAYERWISE_OFFLOAD,
            RESIDENT,
        }
        assert {
            candidate.target_layerwise_resident_layers
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
        } == {(0,), (1,), (2,), (3,)}
        assert all(
            candidate.residency_mode == RESIDENT
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
        )
        resident = next(
            candidate for candidate in candidates if candidate.target_mode() == RESIDENT
        )
        assert {
            candidate.host_materialize_scratch_bytes
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
        } == {resident.target_device_weight_bytes}

    @pytest.mark.parametrize(
        "component_name",
        (
            "text_encoder",
            "text_encoder_2",
            "image_encoder",
            "image_encoder_2",
            "transformer",
            "transformer_2",
            "video_dit",
            "video_dit_2",
            "audio_dit",
            "audio_dit_2",
            "connectors",
            "dual_tower_bridge",
            "vae",
            "vae_2",
            "video_vae",
            "audio_vae",
            "condition_image_encoder",
        ),
    )
    @pytest.mark.parametrize(
        "startup_mode", (COMPONENT_OFFLOAD, LAYERWISE_OFFLOAD, RESIDENT)
    )
    def test_every_serving_component_group_gets_a_complete_frontier(
        self, component_name, startup_mode
    ):
        module = (
            _FakeLayerwiseDit(
                [
                    _FakeLayerwiseManager(
                        {f"layers.{index}.w": torch.zeros(16) for index in range(4)}
                    )
                ]
            )
            if startup_mode == LAYERWISE_OFFLOAD
            else _FakeLazyLayerwiseDit(num_layers=4)
        )

        candidates = collect_residency_targets(
            modules={component_name: module},
            residency_mode_of=self._modes({component_name: startup_mode}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=8,
            layerwise_tuning_of=lambda _name, _dit_group: (
                0.0,
                0.0,
                "leading",
            ),
        )

        assert {candidate.target_mode() for candidate in candidates} == {
            COMPONENT_OFFLOAD,
            LAYERWISE_OFFLOAD,
            RESIDENT,
        }
        assert any(
            candidate.target_mode() == LAYERWISE_OFFLOAD
            and candidate.target_layerwise_resident_layers == (2,)
            for candidate in candidates
        )

    def test_large_host_pin_frontier_is_bounded_and_keeps_extremes(self):
        layer_bytes = {index: 1 << index for index in range(24)}
        manager = SimpleNamespace(
            num_layers=24,
            residency_policy="leading",
            pin_cpu_memory=True,
            layer_weight_bytes=lambda: layer_bytes,
            layer_host_store_bytes=lambda: layer_bytes,
            pinnable_layer_indices=lambda: tuple(layer_bytes),
        )

        targets = _layerwise_pin_targets(
            managers=[manager],
            resident_layers=(0,),
            current_pinned_layers=((),),
            uses_per_streamed_layer=1,
            max_targets=16,
        )

        assert len(targets) <= 16
        assert ((),) in targets
        assert (tuple(layer_bytes),) in targets

    def test_large_measured_resident_frontier_is_bounded_and_keeps_anchors(self):
        managers = [
            _FakeLayerwiseManager(
                {f"first.{index}.w": torch.zeros(16) for index in range(70)}
            ),
            _FakeLayerwiseManager(
                {f"second.{index}.w": torch.zeros(16) for index in range(70)}
            ),
        ]

        targets = _layerwise_resident_targets(
            managers,
            layer_uses=(tuple([8] * 70), tuple([8] * 70)),
            current_resident_layers=(13, 17),
        )

        assert len(targets) <= MAX_LAYERWISE_RESIDENT_TARGETS
        assert (0, 0) in targets
        assert (70, 70) in targets
        assert (70, 0) in targets
        assert (0, 70) in targets
        assert (13, 17) in targets

    def test_large_policy_frontier_is_bounded_and_keeps_extremes(self):
        managers = [
            SimpleNamespace(num_layers=4, residency_policy=RESIDENCY_POLICY_LEADING)
            for _ in range(80)
        ]

        targets = _layerwise_policy_targets(
            managers=managers,
            resident_layers=tuple(2 for _ in managers),
            tune_policy=True,
        )

        assert len(targets) == MAX_LAYERWISE_POLICY_TARGETS
        assert tuple(RESIDENCY_POLICY_LEADING for _ in managers) in targets
        assert tuple(RESIDENCY_POLICY_STRIDED for _ in managers) in targets

    def test_measured_resident_frontier_keeps_current_one_shot_layout(self):
        managers = [
            _FakeLayerwiseManager(
                {f"encoder.{index}.w": torch.zeros(16) for index in range(3)}
            ),
            _FakeLayerwiseManager(
                {f"decoder.{index}.w": torch.zeros(16) for index in range(2)}
            ),
        ]

        targets = _layerwise_resident_targets(
            managers,
            layer_uses=((0, 0, 0), (1, 1)),
            current_resident_layers=(2, 1),
        )

        assert targets == [(0, 0), (2, 1)]

    def test_mixed_dtype_coarse_component_does_not_add_virtual_layerwise(self):
        module = _FakeLazyLayerwiseDit()

        candidates = collect_residency_targets(
            modules={"vae": module},
            residency_mode_of=self._modes({"vae": COMPONENT_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=10,
            mixed_dtype_components={"vae"},
            layerwise_tuning_of=lambda _name, _dit_group: (
                1.0,
                0.0,
                "leading",
            ),
        )

        assert {candidate.target_mode() for candidate in candidates} == {
            COMPONENT_OFFLOAD,
            RESIDENT,
        }

    def test_repeated_non_dit_group_can_tune_policy(self):
        manager = _FakeLayerwiseManager(
            {f"layers.{index}.w": torch.zeros(16) for index in range(4)},
            resident_layers=2,
        )
        module = _FakeLayerwiseDit([manager])

        candidates = collect_residency_targets(
            modules={"custom_refiner": module},
            residency_mode_of=self._modes({"custom_refiner": LAYERWISE_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=10,
            layerwise_policy_is_explicit=lambda _name, _dit_group: False,
            layerwise_layer_uses={"custom_refiner": {"layers": (4, 4, 4, 4)}},
        )

        policies = {
            candidate.target_layerwise_residency_policies
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
            and candidate.target_layerwise_resident_layers == (2,)
            and candidate.target_layerwise_pinned_layers == ((),)
        }
        assert policies == {
            (RESIDENCY_POLICY_LEADING,),
            (RESIDENCY_POLICY_STRIDED,),
        }

    def test_required_resident_component_has_no_tunable_frontier(self):
        candidates = collect_residency_targets(
            modules={"transformer_2": nn.Linear(4, 4)},
            residency_mode_of=self._modes({"transformer_2": RESIDENT}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=1,
            required_resident_components={"transformer_2"},
        )

        assert candidates == []

    def test_resident_seed_preserves_unconfigured_layerwise_frontier(self):
        module = _FakeLazyLayerwiseDit(num_layers=3)
        candidates = collect_residency_targets(
            modules={"text_encoder": module},
            residency_mode_of=self._modes({"text_encoder": RESIDENT}),
            baseline_residency_mode_of=self._modes({"text_encoder": LAYERWISE_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=1,
            layerwise_tuning_of=lambda _name, _dit_group: (1.0, 0.0, "leading"),
        )

        assert {candidate.target_mode() for candidate in candidates} == {
            COMPONENT_OFFLOAD,
            LAYERWISE_OFFLOAD,
            RESIDENT,
        }
        assert all(candidate.residency_mode == RESIDENT for candidate in candidates)
        assert next(
            candidate for candidate in candidates if candidate.target_mode() == RESIDENT
        ).current_placement

    def test_virtual_layerwise_frontier_includes_host_pin_prefixes(self):
        module = _FakeLazyLayerwiseDit(num_layers=3)
        candidates = collect_residency_targets(
            modules={"transformer": module},
            residency_mode_of=self._modes({"transformer": COMPONENT_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=4,
            layerwise_tuning_of=lambda _name, _dit_group: (1.0, 0.0, "leading"),
        )

        streamed = [
            candidate
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
            and candidate.target_layerwise_resident_layers == (0,)
        ]
        assert {candidate.target_layerwise_pinned_layers for candidate in streamed} == {
            ((),),
            ((0,),),
            ((0, 1),),
            ((0, 1, 2),),
        }
        fully_pinned = next(
            candidate
            for candidate in streamed
            if candidate.target_layerwise_pinned_layers == ((0, 1, 2),)
        )
        assert fully_pinned.pinned_host_delta_bytes > 0
        assert (
            fully_pinned.host_pin_scratch_bytes == fully_pinned.pinned_host_delta_bytes
        )

    def test_virtual_layerwise_reuses_component_offload_host_weights(self):
        module = _FakeLazyLayerwiseDit(num_layers=3)
        candidates = collect_residency_targets(
            modules={"transformer": module},
            residency_mode_of=self._modes({"transformer": COMPONENT_OFFLOAD}),
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=4,
            layerwise_tuning_of=lambda _name, _dit_group: (1.0, 0.0, "leading"),
        )

        streamed = next(
            candidate
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
            and candidate.target_layerwise_resident_layers == (0,)
        )
        assert streamed.device_transition_delta_bytes == 0
        assert streamed.active_device_delta_bytes < 0
        resident = next(
            candidate for candidate in candidates if candidate.target_mode() == RESIDENT
        )
        assert (
            0
            < streamed.host_materialize_scratch_bytes
            < resident.target_device_weight_bytes
        )


class TestApplyAndRollback:
    def test_lazy_layerwise_configuration_rolls_back_to_component_offload(self):
        module = _FakeLazyLayerwiseDit(num_layers=4)
        args = _StubResidencyArgs()
        candidates = collect_residency_targets(
            modules={"transformer": module},
            residency_mode_of=lambda _name: COMPONENT_OFFLOAD,
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=10,
            layerwise_tuning_of=lambda _name, _dit_group: (0.0, 0.0, "leading"),
        )
        target = next(
            candidate
            for candidate in candidates
            if candidate.target_mode() == LAYERWISE_OFFLOAD
            and candidate.target_layerwise_resident_layers == (2,)
            and candidate.target_layerwise_pinned_layers == ((0, 1, 2, 3),)
        )

        applied = apply_residency_changes(
            plan=_plan_for([target]),
            modules={"transformer": module},
            server_args=args,
        )

        assert len(module.layerwise_offload_managers) == 1
        assert module.layerwise_offload_managers[0].resident_layers == 2
        assert module.layerwise_offload_managers[0].pinned_layer_indices() == (
            0,
            1,
            2,
            3,
        )
        assert args.auto_modes == {"transformer": LAYERWISE_OFFLOAD}
        manager = module.layerwise_offload_managers[0]

        rollback_residency_changes(
            applied=applied,
            modules={"transformer": module},
            server_args=args,
        )
        assert module.layerwise_offload_managers == []
        assert args.auto_modes == {}
        assert module.layers[0].weight.device.type == "cpu"
        assert manager.load_all_layers_calls == 0

    def test_layerwise_can_become_component_offload_and_rollback(self):
        module = _FakeLazyLayerwiseDit(num_layers=4)
        args = _StubResidencyArgs()
        module.configure_layerwise_offload(args)
        manager = module.layerwise_offload_managers[0]
        manager.set_pinned_layers((0,))
        args.auto_modes["text_encoder"] = LAYERWISE_OFFLOAD
        candidates = collect_residency_targets(
            modules={"text_encoder": module},
            residency_mode_of=lambda _name: LAYERWISE_OFFLOAD,
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=(),
            num_inference_steps=1,
        )
        target = next(
            candidate
            for candidate in candidates
            if candidate.target_mode() == COMPONENT_OFFLOAD
        )

        applied = apply_residency_changes(
            plan=_plan_for([target]),
            modules={"text_encoder": module},
            server_args=args,
        )

        assert module.layerwise_offload_managers == []
        assert args.auto_modes == {"text_encoder": COMPONENT_OFFLOAD}
        assert module.layers[0].weight.device.type == "cpu"
        assert target.target_resident_weight_bytes == 0
        assert manager.load_all_layers_calls == 0

        rollback_residency_changes(
            applied=applied,
            # Lazy stage-owned components can disappear from the manager's
            # per-request name index before validation rollback.
            modules={},
            server_args=args,
        )
        assert len(module.layerwise_offload_managers) == 1
        assert args.auto_modes == {"text_encoder": LAYERWISE_OFFLOAD}
        assert module.layerwise_offload_managers[0].pinned_layer_indices() == (0,)

    def test_component_offload_promotion_materializes_before_validation(self):
        class RecordingLinear(nn.Linear):
            def __init__(self):
                super().__init__(4, 4)
                self.to_targets = []

            def to(self, *args, **kwargs):
                self.to_targets.append(args[0])
                return super().to(*args, **kwargs)

        module = RecordingLinear()
        args = _StubResidencyArgs()
        applied = apply_residency_changes(
            plan=_plan_for([_candidate("text_encoder", weight_gib=1)]),
            modules={"text_encoder": module},
            server_args=args,
        )
        assert args.auto_modes == {"text_encoder": RESIDENT}
        assert module.to_targets == [current_platform.get_local_torch_device()]
        assert [p.component_name for p in applied] == ["text_encoder"]

        rollback_residency_changes(
            applied=applied, modules={"text_encoder": module}, server_args=args
        )
        assert args.auto_modes == {}
        assert module.to_targets[-1] == "cpu"

    def test_component_resident_can_be_demoted_and_rollback_restores_it(self):
        module = nn.Linear(4, 4)
        args = _StubResidencyArgs()
        args.auto_modes["text_encoder"] = RESIDENT
        candidate = ResidencyTarget(
            component_name="text_encoder",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=0,
            inactive_device_delta_bytes=-GIB_BYTES,
        )

        applied = apply_residency_changes(
            plan=_plan_for([candidate]),
            modules={"text_encoder": module},
            server_args=args,
        )
        assert args.auto_modes == {"text_encoder": COMPONENT_OFFLOAD}

        rollback_residency_changes(
            applied=applied,
            modules={"text_encoder": module},
            server_args=args,
        )
        assert args.auto_modes == {"text_encoder": RESIDENT}

    def test_layerwise_promotion_loads_all_layers_and_rollback_rearms(self):
        manager = _FakeLayerwiseManager({"layers.0.w": torch.zeros(16)})
        module = _FakeLayerwiseDit([manager])
        args = _StubResidencyArgs()
        candidate = _candidate("transformer", mode=LAYERWISE_OFFLOAD, weight_gib=1)

        applied = apply_residency_changes(
            plan=_plan_for([candidate]),
            modules={"transformer": module},
            server_args=args,
        )
        assert manager.load_all_layers_calls == 1
        assert manager.remove_hooks_calls == 1
        assert manager.enabled is False

        rollback_residency_changes(
            applied=applied, modules={"transformer": module}, server_args=args
        )
        assert manager.enabled is True
        assert manager.register_hooks_calls == 1
        assert args.auto_modes == {}

    def test_resident_only_promotion_keeps_pins_until_validation_commit(self):
        manager = _FakeLayerwiseManager({"layers.0.w": torch.zeros(16)})
        manager._pinned_layers = (0,)
        module = _FakeLayerwiseDit([manager])
        args = _StubResidencyArgs()
        candidate = _candidate("text_encoder", mode=LAYERWISE_OFFLOAD, weight_gib=1)

        applied = apply_residency_changes(
            plan=_plan_for([candidate]),
            modules={"text_encoder": module},
            server_args=args,
        )

        assert manager.enabled is False
        assert manager.pinned_layer_indices() == (0,)
        assert manager.release_host_stores_calls == 0

        commit_residency_changes(
            applied=applied,
            modules={"text_encoder": module},
            server_args=args,
        )

        assert manager.release_host_stores_calls == 1
        assert module.layerwise_offload_managers == []

    def test_partial_layerwise_promotion_restores_exact_group_counts(self):
        managers = [
            _FakeLayerwiseManager(
                {
                    "layers.0.w": torch.zeros(16),
                    "layers.1.w": torch.zeros(16),
                    "layers.2.w": torch.zeros(16),
                },
                resident_layers=1,
            ),
            _FakeLayerwiseManager(
                {
                    "layers.0.w": torch.zeros(16),
                    "layers.1.w": torch.zeros(16),
                }
            ),
        ]
        module = _FakeLayerwiseDit(managers)
        args = _StubResidencyArgs()
        candidate = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=GIB_BYTES,
            h2d_bytes_per_request=10 * GIB_BYTES,
            target_layerwise_resident_layers=(2, 2),
            target_layerwise_pinned_layers=((), ()),
        )

        applied = apply_residency_changes(
            plan=_plan_for([candidate]),
            modules={"transformer": module},
            server_args=args,
        )

        assert [manager.resident_layers for manager in managers] == [2, 2]
        assert args.auto_modes == {"transformer": LAYERWISE_OFFLOAD}
        rollback_residency_changes(
            applied=applied,
            modules={"transformer": module},
            server_args=args,
        )
        assert [manager.resident_layers for manager in managers] == [1, 0]

    def test_permanent_layerwise_can_be_demoted_and_rollback_restores_it(self):
        manager = _FakeLayerwiseManager(
            {
                "layers.0.w": torch.zeros(16),
                "layers.1.w": torch.zeros(16),
            },
            resident_layers=2,
        )
        module = _FakeLayerwiseDit([manager])
        module.disable_offload()
        args = _StubResidencyArgs()
        args.auto_modes["transformer"] = RESIDENT
        candidate = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=0,
            target_layerwise_resident_layers=(1,),
            target_layerwise_pinned_layers=((),),
        )

        applied = apply_residency_changes(
            plan=_plan_for([candidate]),
            modules={"transformer": module},
            server_args=args,
        )

        assert manager.enabled is True
        assert manager.resident_layers == 1
        assert args.auto_modes == {"transformer": LAYERWISE_OFFLOAD}

        rollback_residency_changes(
            applied=applied,
            modules={"transformer": module},
            server_args=args,
        )
        assert manager.enabled is False
        assert manager.resident_layers == 2
        assert args.auto_modes == {"transformer": RESIDENT}

    def test_mid_failure_rolls_back_already_applied_promotions(self):
        manager = _FakeLayerwiseManager({"layers.0.w": torch.zeros(16)})
        module = _FakeLayerwiseDit([manager])
        args = _StubResidencyArgs()
        candidates = [
            _candidate("transformer", mode=LAYERWISE_OFFLOAD, weight_gib=1),
            _candidate("missing_component", weight_gib=1),
        ]
        with pytest.raises(RuntimeError, match="missing_component"):
            apply_residency_changes(
                plan=_plan_for(candidates),
                modules={"transformer": module},
                server_args=args,
            )
        assert args.auto_modes == {}
        assert manager.enabled is True
        # Validation finishes before any placement is touched.
        assert manager.register_hooks_calls == 0

    def test_hostpin_handoff_releases_all_components_before_acquiring(
        self, monkeypatch
    ):
        events: list[str] = []
        source_manager = _FakeLayerwiseManager(
            {"layers.0.w": torch.zeros(16)},
            event_log=events,
            event_name="source",
        )
        source_manager._pinned_layers = (0,)
        target_manager = _FakeLayerwiseManager(
            {"layers.0.w": torch.zeros(16)},
            event_log=events,
            event_name="target",
        )
        modules = {
            "source": _FakeLayerwiseDit([source_manager]),
            "target": _FakeLayerwiseDit([target_manager]),
        }
        candidates = [
            ResidencyTarget(
                component_name="source",
                residency_mode=LAYERWISE_OFFLOAD,
                target_resident_weight_bytes=0,
                h2d_bytes_per_request=-1,
                target_layerwise_resident_layers=(0,),
                target_layerwise_pinned_layers=((),),
                pinned_host_delta_bytes=-source_manager.offloaded_weight_bytes(),
            ),
            ResidencyTarget(
                component_name="target",
                residency_mode=LAYERWISE_OFFLOAD,
                target_resident_weight_bytes=0,
                h2d_bytes_per_request=10,
                target_layerwise_resident_layers=(0,),
                target_layerwise_pinned_layers=((0,),),
                pinned_host_delta_bytes=target_manager.offloaded_weight_bytes(),
            ),
        ]
        monkeypatch.setattr(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "auto_residency.release_unused_pinned_memory",
            lambda: events.append("flush"),
        )

        applied = apply_residency_changes(
            plan=_plan_for(candidates),
            modules=modules,
            server_args=_StubResidencyArgs(),
        )

        assert events == ["source:unpin", "flush", "target:pin"]
        assert all(adjustment.pinned_host_changed for adjustment in applied)

    def test_vram_handoff_releases_before_materializing(self):
        events: list[str] = []
        source_manager = _FakeLayerwiseManager(
            {"layers.0.w": torch.zeros(16)},
            event_log=events,
            event_name="source",
        )
        target_manager = _FakeLayerwiseManager(
            {"layers.0.w": torch.zeros(16)},
            event_log=events,
            event_name="target",
        )
        source = _FakeLayerwiseDit([source_manager])
        target = _FakeLayerwiseDit([target_manager])
        source.disable_offload()
        events.clear()
        candidates = [
            ResidencyTarget(
                component_name="source",
                residency_mode=LAYERWISE_OFFLOAD,
                target_residency_mode=LAYERWISE_OFFLOAD,
                target_resident_weight_bytes=0,
                h2d_bytes_per_request=-1,
                target_layerwise_resident_layers=(0,),
                target_layerwise_pinned_layers=((),),
                device_transition_delta_bytes=-source_manager.offloaded_weight_bytes(),
                active_device_delta_bytes=100,
            ),
            ResidencyTarget(
                component_name="target",
                residency_mode=LAYERWISE_OFFLOAD,
                target_residency_mode=RESIDENT,
                target_resident_weight_bytes=target_manager.offloaded_weight_bytes(),
                h2d_bytes_per_request=10,
                target_layerwise_resident_layers=(1,),
                target_layerwise_pinned_layers=((),),
                device_transition_delta_bytes=target_manager.offloaded_weight_bytes(),
                active_device_delta_bytes=-100,
            ),
        ]

        apply_residency_changes(
            plan=_plan_for(candidates),
            modules={"source": source, "target": target},
            server_args=_StubResidencyArgs(),
        )

        assert events == ["source:sync", "target:load"]

    def test_vram_rollback_releases_before_restoring_resident_layers(self):
        events: list[str] = []

        class _TrackedModule(nn.Module):
            def to(self, *args, **kwargs):
                events.append("text_encoder:release")
                return self

        manager = _FakeLayerwiseManager(
            {"layers.0.w": torch.zeros(16)},
            event_log=events,
            event_name="transformer",
        )
        transformer = _FakeLayerwiseDit([manager])
        applied = [
            AppliedResidencyChange(
                component_name="text_encoder",
                residency_mode=COMPONENT_OFFLOAD,
                applied_device_delta_bytes=10,
            ),
            AppliedResidencyChange(
                component_name="transformer",
                residency_mode=LAYERWISE_OFFLOAD,
                previous_layerwise_resident_layers=(1,),
                previous_layerwise_pinned_layers=((),),
                previous_layerwise_offload_enabled=False,
                applied_device_delta_bytes=-10,
            ),
        ]

        rollback_residency_changes(
            applied=applied,
            modules={
                "text_encoder": _TrackedModule(),
                "transformer": transformer,
            },
            server_args=_StubResidencyArgs(),
        )

        assert events == ["text_encoder:release", "transformer:load"]

    def test_hostpin_rollback_releases_new_owner_before_restoring_old_owner(
        self, monkeypatch
    ):
        events: list[str] = []
        source_manager = _FakeLayerwiseManager(
            {"layers.0.w": torch.zeros(16)},
            event_log=events,
            event_name="source",
        )
        target_manager = _FakeLayerwiseManager(
            {"layers.0.w": torch.zeros(16)},
            event_log=events,
            event_name="target",
        )
        target_manager._pinned_layers = (0,)
        modules = {
            "source": _FakeLayerwiseDit([source_manager]),
            "target": _FakeLayerwiseDit([target_manager]),
        }
        applied = [
            AppliedResidencyChange(
                component_name="source",
                residency_mode=LAYERWISE_OFFLOAD,
                previous_layerwise_resident_layers=(0,),
                previous_layerwise_pinned_layers=((0,),),
                pinned_host_changed=True,
            ),
            AppliedResidencyChange(
                component_name="target",
                residency_mode=LAYERWISE_OFFLOAD,
                previous_layerwise_resident_layers=(0,),
                previous_layerwise_pinned_layers=((),),
                pinned_host_changed=True,
            ),
        ]
        monkeypatch.setattr(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "auto_residency.release_unused_pinned_memory",
            lambda: events.append("flush"),
        )

        rollback_residency_changes(
            applied=applied,
            modules=modules,
            server_args=_StubResidencyArgs(),
        )

        assert events == ["target:unpin", "flush", "source:pin"]

    def test_enable_offload_does_not_double_register_hooks(self):
        manager = _FakeLayerwiseManager({"layers.0.w": torch.zeros(16)})
        module = _FakeLayerwiseDit([manager])
        # never disabled: re-enabling must not stack a second set of hooks
        module.enable_offload()
        assert manager.register_hooks_calls == 0

    def test_disable_offload_failure_rearms_the_manager(self):
        # load_all_layers is the step most likely to OOM; a failed disable
        # must leave the manager streaming (hooks re-registered), not in a
        # hook-less enabled=True state serving (1,) placeholders.
        manager = _FakeLayerwiseManager({"layers.0.w": torch.zeros(16)})
        manager.fail_load = True
        module = _FakeLayerwiseDit([manager])
        with pytest.raises(RuntimeError, match="out of memory"):
            module.disable_offload()
        assert manager.enabled is True
        assert manager.register_hooks_calls == 1
        assert manager.release_all_calls == 1

    def test_disable_offload_failure_rearms_earlier_layer_groups(self):
        first = _FakeLayerwiseManager({"layers.0.w": torch.zeros(16)})
        failing = _FakeLayerwiseManager({"layers.1.w": torch.zeros(16)})
        failing.fail_load = True
        module = _FakeLayerwiseDit([first, failing])

        with pytest.raises(RuntimeError, match="out of memory"):
            module.disable_offload()

        assert first.enabled is True
        assert first.register_hooks_calls == 1
        assert first.release_all_calls == 1
        assert failing.enabled is True
        assert failing.register_hooks_calls == 1
        assert failing.release_all_calls == 1

    def test_failed_rollback_raises_rollback_error_with_visible_cause(self):
        class _BrokenArgs(_StubResidencyArgs):
            def clear_auto_residency_mode(self, component_name):
                # str(AssertionError()) is "" -- describe_error must keep it
                # visible so no truthiness filter can drop it
                raise AssertionError()

        args = _BrokenArgs()
        failing_manager = _FakeLayerwiseManager({"layers.0.w": torch.zeros(16)})
        failing_manager.fail_load = True
        candidates = [
            _candidate("text_encoder", weight_gib=1),
            _candidate("transformer", mode=LAYERWISE_OFFLOAD, weight_gib=1),
        ]
        with pytest.raises(AutoResidencyRollbackError) as exc_info:
            apply_residency_changes(
                plan=_plan_for(candidates),
                modules={
                    "text_encoder": nn.Linear(2, 2),
                    "transformer": _FakeLayerwiseDit([failing_manager]),
                },
                server_args=args,
            )
        assert "AssertionError" in str(exc_info.value)
        assert "CUDA out of memory" in str(exc_info.value)

    def test_rollback_keeps_going_past_a_broken_component(self):
        released: list[str] = []

        class _PartialArgs(_StubResidencyArgs):
            def clear_auto_residency_mode(self, component_name):
                if component_name == "vae":
                    raise RuntimeError("boom")
                released.append(component_name)

        args = _PartialArgs()
        applied = [
            AppliedResidencyChange(
                component_name="text_encoder", residency_mode=COMPONENT_OFFLOAD
            ),
            AppliedResidencyChange(
                component_name="vae", residency_mode=COMPONENT_OFFLOAD
            ),
        ]
        with pytest.raises(RuntimeError, match="vae"):
            rollback_residency_changes(
                applied=applied,
                modules={},
                server_args=args,
            )
        # the earlier-applied adjustment was still undone
        assert released == ["text_encoder"]

    def test_apply_and_rollback_restore_rank_hostpin_quota(self):
        budget = HostPinBudget(
            available_bytes=40 * GIB_BYTES,
            reserve_bytes=2 * GIB_BYTES,
        )
        args = _StubResidencyArgs(host_pin_budget=budget)
        target = ResidencyTarget(
            component_name="text_encoder",
            residency_mode=COMPONENT_OFFLOAD,
            target_residency_mode=RESIDENT,
            target_resident_weight_bytes=GIB_BYTES,
            h2d_bytes_per_request=GIB_BYTES,
        )
        plan = AutoResidencyPlan(
            host_pin_target_bytes_by_rank={0: 12 * GIB_BYTES},
            changes=[target],
        )

        applied = apply_residency_changes(
            plan=plan,
            modules={"text_encoder": nn.Linear(2, 2)},
            server_args=args,
            rank=0,
        )
        assert budget.available_bytes == 12 * GIB_BYTES
        assert budget.reserve_bytes == 0

        rollback_residency_changes(
            applied=applied,
            modules={"text_encoder": nn.Linear(2, 2)},
            server_args=args,
        )
        assert budget.available_bytes == 40 * GIB_BYTES
        assert budget.reserve_bytes == 2 * GIB_BYTES
        assert budget.planning_capacity_bytes == 38 * GIB_BYTES

    def test_apply_failure_restores_rank_hostpin_quota(self):
        budget = HostPinBudget(
            available_bytes=40 * GIB_BYTES,
            reserve_bytes=2 * GIB_BYTES,
        )
        args = _StubResidencyArgs(host_pin_budget=budget)
        plan = AutoResidencyPlan(
            host_pin_target_bytes_by_rank={0: 12 * GIB_BYTES},
            changes=[
                ResidencyTarget(
                    component_name="missing",
                    residency_mode=COMPONENT_OFFLOAD,
                    target_residency_mode=RESIDENT,
                    target_resident_weight_bytes=GIB_BYTES,
                    h2d_bytes_per_request=GIB_BYTES,
                )
            ],
        )

        with pytest.raises(RuntimeError, match="is missing"):
            apply_residency_changes(
                plan=plan,
                modules={},
                server_args=args,
                rank=0,
            )

        assert budget.available_bytes == 40 * GIB_BYTES
        assert budget.reserve_bytes == 2 * GIB_BYTES
        assert budget.planning_capacity_bytes == 38 * GIB_BYTES

    def test_partial_layerwise_policy_change_rolls_back_exact_layout(self):
        manager = _FakeLayerwiseManager(
            {f"layers.{index}.w": torch.zeros(16) for index in range(4)},
            resident_layers=2,
        )
        module = _FakeLayerwiseDit([manager])
        args = _StubResidencyArgs()
        candidate = ResidencyTarget(
            component_name="transformer",
            residency_mode=LAYERWISE_OFFLOAD,
            target_resident_weight_bytes=manager.resident_weight_bytes(2),
            h2d_bytes_per_request=GIB_BYTES,
            target_layerwise_resident_layers=(2,),
            target_layerwise_residency_policies=(RESIDENCY_POLICY_STRIDED,),
            target_layerwise_pinned_layers=((),),
        )

        applied = apply_residency_changes(
            plan=_plan_for([candidate]),
            modules={"transformer": module},
            server_args=args,
        )
        assert manager.residency_policy == RESIDENCY_POLICY_STRIDED

        rollback_residency_changes(
            applied=applied,
            modules={"transformer": module},
            server_args=args,
        )
        assert manager.resident_layers == 2
        assert manager.residency_policy == RESIDENCY_POLICY_LEADING


class TestCandidateRanking:
    def test_post_request_hint_order_matches_promotion_order(self):
        # The recommendation log and the adjustment plan share one ranking, so
        # the hint lists components in the order auto mode would promote them.
        candidates = [
            _candidate("vae", weight_gib=1),
            _candidate(
                "transformer", mode=LAYERWISE_OFFLOAD, weight_gib=10, h2d_gib=500
            ),
            _candidate("text_encoder", weight_gib=7),
        ]
        ranked = rank_candidates_by_h2d_savings(candidates)
        assert [candidate.component_name for candidate in ranked] == [
            "transformer",
            "text_encoder",
            "vae",
        ]
        plan = plan_auto_residency(
            reports=[_report(budget_gib=1000, estimated_gib=50, candidates=candidates)]
        )
        assert [candidate.component_name for candidate in plan.changes] == [
            candidate.component_name for candidate in ranked
        ]


class TestAppliedChangesLog:
    pass

    def test_reports_runtime_transition_and_kill_switch(self):
        plan = _plan_for(
            [
                _candidate("text_encoder", mode=LAYERWISE_OFFLOAD, weight_gib=7),
                _candidate("vae", weight_gib=1),
            ]
        )
        message = format_applied_changes(plan=plan)
        assert "text_encoder: layerwise-offload -> resident" in message
        assert "vae: component-offload -> resident" in message
        assert "Startup flags may use a different load path" in message
        assert "--component-residency" not in message
        assert "SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY=1" in message


class TestWarmupFrameAdjustment:
    def _server_args(self, *, bcg: bool = False, num_gpus: int = 1) -> SimpleNamespace:
        return SimpleNamespace(
            pipeline_config=LongLive2T2VConfig(),
            pipeline_class_name=None,
            enable_breakable_cuda_graph=bcg,
            num_gpus=num_gpus,
        )

    def _defaults(self) -> SimpleNamespace:
        return SimpleNamespace(
            num_frames=61,
            adjust_frames=True,
            enable_sequence_shard=None,
            num_frames_round_down=False,
        )

    def test_capped_frames_keep_the_model_frame_contract(self):
        # LongLive2 default 61 frames is capped to 17, whose 5 latent frames
        # break the 8-frame causal block; the builder must re-align to 29.
        assert SERVER_WARMUP_MAX_VIDEO_FRAMES == 17
        num_frames = _resolve_warmup_num_frames(
            self._server_args(), self._defaults(), server_based_warmup=True
        )
        assert num_frames == 29

    def test_bcg_keeps_full_serving_frames(self):
        num_frames = _resolve_warmup_num_frames(
            self._server_args(bcg=True), self._defaults(), server_based_warmup=True
        )
        assert num_frames == 61

    def test_non_server_warmup_keeps_default_frames(self):
        num_frames = _resolve_warmup_num_frames(
            self._server_args(), self._defaults(), server_based_warmup=False
        )
        assert num_frames == 61

    def test_capped_frames_get_the_gpu_alignment_real_requests_get(self):
        # a frame-aligning pipeline (no sequence shard) on multiple GPUs:
        # 17 frames -> 5 latent frames -> ceil to 6 latents on 2 GPUs -> 21
        args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                task_type=ModelTaskType.T2V,
                adjust_num_frames=lambda n: n,
                vae_config=SimpleNamespace(
                    use_temporal_scaling_frames=True,
                    arch_config=SimpleNamespace(temporal_compression_ratio=4),
                ),
            ),
            pipeline_class_name=None,
            enable_breakable_cuda_graph=False,
            num_gpus=2,
        )
        defaults = SimpleNamespace(
            num_frames=81,
            adjust_frames=True,
            enable_sequence_shard=None,
            num_frames_round_down=False,
        )
        num_frames = _resolve_warmup_num_frames(
            args, defaults, server_based_warmup=True
        )
        assert num_frames == 21


class TestAutoResidencyWarmupShape:
    def _patch_gate(self, monkeypatch, reason: str | None = None) -> None:
        monkeypatch.setattr(
            "sglang.multimodal_gen.runtime.warmup_request_builder.auto_residency_args_skip_reason",
            lambda _args: reason,
        )

    def _wan_like_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            pipeline_config=SimpleNamespace(
                task_type=ModelTaskType.T2V,
                adjust_num_frames=lambda n: n,
            ),
            num_gpus=1,
        )

    def _defaults(
        self,
        num_frames: int,
        *,
        width: int | None = 1280,
        height: int | None = 720,
        supported_resolutions=None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            width=width,
            height=height,
            num_frames=num_frames,
            supported_resolutions=supported_resolutions,
            adjust_frames=True,
            enable_sequence_shard=None,
            num_frames_round_down=False,
        )

    def test_capped_video_gets_a_full_shape_probe(self, monkeypatch):
        self._patch_gate(monkeypatch)
        probe = _resolve_auto_residency_warmup_shape(
            self._wan_like_args(),
            self._defaults(81),
            warmup_shape=(832, 480, 17),
            server_based_warmup=True,
        )
        assert probe == (1280, 720, 81)

    def test_matching_warmup_needs_no_probe(self, monkeypatch):
        self._patch_gate(monkeypatch)
        probe = _resolve_auto_residency_warmup_shape(
            self._wan_like_args(),
            self._defaults(17, width=832, height=480),
            warmup_shape=(832, 480, 17),
            server_based_warmup=True,
        )
        assert probe is None

    def test_unknown_target_resolution_skips_probe(self, monkeypatch):
        self._patch_gate(monkeypatch)
        probe = _resolve_auto_residency_warmup_shape(
            self._wan_like_args(),
            self._defaults(81, width=None, height=None),
            warmup_shape=(832, 480, 17),
            server_based_warmup=True,
        )
        assert probe is None

    def test_skip_gate_disables_probe(self, monkeypatch):
        # Full-shape calibration must share the adjustment's own gate (kill
        # switch, quantized, manual, ...).
        self._patch_gate(monkeypatch, reason="performance_mode=manual")
        probe = _resolve_auto_residency_warmup_shape(
            self._wan_like_args(),
            self._defaults(81),
            warmup_shape=(832, 480, 17),
            server_based_warmup=True,
        )
        assert probe is None

    def test_supported_resolution_fills_missing_target_size(self, monkeypatch):
        self._patch_gate(monkeypatch)
        probe = _resolve_auto_residency_warmup_shape(
            self._wan_like_args(),
            self._defaults(
                81,
                width=None,
                height=None,
                supported_resolutions=[(832, 480), (1024, 1024)],
            ),
            warmup_shape=(832, 480, 17),
            server_based_warmup=True,
        )
        assert probe == (1024, 1024, 81)


class TestAutoResidencySkipReason:
    def _base_args(self, **overrides) -> SimpleNamespace:
        args = SimpleNamespace(
            performance_mode="auto",
            warmup_mode="server",
            warmup_resolutions=None,
            disagg_role="monolithic",
            backend="sglang",
            enable_breakable_cuda_graph=False,
            enable_torch_compile=False,
            batching_max_size=1,
            dp_size=1,
            ulysses_degree=1,
            use_fsdp_inference=False,
            quantization=None,
            component_quantizations={},
            transformer_weights_path=None,
            nunchaku_config=None,
            direct_gpu_weight_loading=False,
            ltx2_two_stage_device_mode=None,
            pipeline_class_name=None,
            pipeline_config=SimpleNamespace(
                task_type=ModelTaskType.T2V,
                supports_auto_residency=True,
            ),
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def _skip_reason(self, args):
        from sglang.multimodal_gen.runtime.server_warmup import (
            auto_residency_skip_reason,
        )

        return auto_residency_skip_reason(args)

    def test_env_kill_switch(self, monkeypatch):
        monkeypatch.setenv("SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY", "1")
        reason = self._skip_reason(self._base_args())
        assert (
            reason is not None and "SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY" in reason
        )

    def test_manual_performance_mode(self, monkeypatch):
        monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY", raising=False)
        reason = self._skip_reason(self._base_args(performance_mode="manual"))
        assert reason == "performance_mode=manual"

    @pytest.mark.parametrize(
        "overrides, expected_fragment",
        [
            ({"warmup_mode": "request"}, "server warmup"),
            ({"disagg_role": "denoiser"}, "server warmup"),
            ({"backend": "diffusers"}, "diffusers"),
            (
                {"ltx2_two_stage_device_mode": "original"},
                "LTX-2 original two-stage placement",
            ),
            (
                {"pipeline_class_name": "LTX2TwoStagePipeline"},
                "legacy LTX-2 two-stage placement",
            ),
            ({"enable_breakable_cuda_graph": True}, "CUDA graph"),
            # compile warmup strips the memory layout (layerwise DiT +
            # resident aux components on CPU): its peaks are not serving peaks
            ({"enable_torch_compile": True}, "stripped memory layout"),
            ({"batching_max_size": 4}, "batching"),
            (
                {
                    "pipeline_config": SimpleNamespace(
                        task_type=ModelTaskType.T2V,
                        supports_auto_residency=False,
                    )
                },
                "post-warmup residency changes",
            ),
        ],
    )
    def test_excluded_paths(self, monkeypatch, overrides, expected_fragment):
        monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY", raising=False)
        monkeypatch.delenv("SGLANG_CACHE_DIT_ENABLED", raising=False)
        reason = self._skip_reason(self._base_args(**overrides))
        assert reason is not None and expected_fragment in reason

    @pytest.mark.parametrize(
        "overrides",
        [
            {"quantization": "fp8"},
            {"component_quantizations": {"image_encoder": "fp8"}},
            {"transformer_weights_path": "/x.safetensors"},
            {"direct_gpu_weight_loading": True},
        ],
    )
    def test_fixed_loading_paths_still_calibrate_other_components(
        self, monkeypatch, overrides
    ):
        monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY", raising=False)
        monkeypatch.delenv("SGLANG_CACHE_DIT_ENABLED", raising=False)
        reason = self._skip_reason(self._base_args(**overrides))
        assert reason is None or reason == "requires CUDA"

    def test_fixed_loading_components_are_excluded_individually(self):
        from sglang.multimodal_gen.runtime.server_args.auto_tune import (
            fixed_loading_residency_components,
        )

        component_names = ("transformer", "transformer_2", "image_encoder", "vae")
        assert fixed_loading_residency_components(
            self._base_args(quantization="fp8"), component_names
        ) == {"transformer", "transformer_2"}
        assert fixed_loading_residency_components(
            self._base_args(component_quantizations={"image_encoder": "fp8"}),
            component_names,
        ) == {"image_encoder"}
        assert (
            fixed_loading_residency_components(
                self._base_args(transformer_weights_path="/x.safetensors"),
                component_names,
            )
            == set()
        )
        assert fixed_loading_residency_components(
            self._base_args(ltx2_two_stage_device_mode="original"),
            component_names,
        ) == {"transformer", "transformer_2"}
        assert (
            fixed_loading_residency_components(
                self._base_args(ltx2_two_stage_device_mode="resident"),
                component_names,
            )
            == set()
        )

    def test_cache_dit_excluded(self, monkeypatch):
        monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY", raising=False)
        monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")
        reason = self._skip_reason(self._base_args())
        assert reason is not None and "cache-dit" in reason

    @pytest.mark.parametrize(
        "overrides",
        [
            {"dp_size": 2},
            {"ulysses_degree": 2},
            {"use_fsdp_inference": True},
        ],
    )
    def test_parallel_paths_reach_platform_gate(self, monkeypatch, overrides):
        monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY", raising=False)
        monkeypatch.delenv("SGLANG_CACHE_DIT_ENABLED", raising=False)
        reason = self._skip_reason(self._base_args(**overrides))
        assert reason is None or reason == "requires CUDA"

    def test_eligible_path_reaches_platform_gate(self, monkeypatch):
        monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY", raising=False)
        monkeypatch.delenv("SGLANG_CACHE_DIT_ENABLED", raising=False)
        reason = self._skip_reason(self._base_args())
        # on a CUDA host everything passes; CPU CI stops at the platform gate
        assert reason is None or reason == "requires CUDA"

    def test_transformer_weight_override_reaches_platform_gate(self, monkeypatch):
        monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY", raising=False)
        monkeypatch.delenv("SGLANG_CACHE_DIT_ENABLED", raising=False)
        reason = self._skip_reason(
            self._base_args(transformer_weights_path="/x.safetensors")
        )
        assert reason is None or reason == "requires CUDA"


class TestEstimateAllocatorHeadroom:
    def test_uses_covering_workload_allocator_gap(self):
        small = _record(
            num_frames=9,
            peak_gib=10,
            peak_reserved_gib=11,
        )
        target = _record(
            num_frames=17,
            peak_gib=20,
            peak_reserved_gib=26,
        )

        assert (
            estimate_allocator_headroom_bytes(
                records=[small, target], target_units=target.workload_units()
            )
            == 6 * GIB_BYTES
        )

    def test_keeps_largest_observed_gap_without_covering_workload(self):
        first = _record(peak_gib=10, peak_reserved_gib=12)
        second = _record(peak_gib=15, peak_reserved_gib=19)

        assert (
            estimate_allocator_headroom_bytes(
                records=[first, second], target_units=second.workload_units() * 2
            )
            == 4 * GIB_BYTES
        )
