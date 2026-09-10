# SPDX-License-Identifier: Apache-2.0
"""Unit tests for warmup-calibrated auto residency adjustment."""

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.longlive2 import LongLive2T2VConfig
from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
    ACTIVATION_EXTRAPOLATION_MARGIN,
    GIB_BYTES,
    DefaultWorkload,
    WarmupMemoryRecord,
    estimate_default_workload_peak_bytes,
    estimate_default_workload_timing,
    estimate_layerwise_layer_uses,
    estimate_workload_phase_peaks,
    resolve_measured_default_workload,
)
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
    peak_reserved_gib=0,
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
        peak_reserved_bytes=peak_reserved_gib * GIB_BYTES,
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

        peaks, active, used, _ = estimate_workload_phase_peaks(
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

        peaks, _, _, _ = estimate_workload_phase_peaks(
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

        peaks, _, _, transitions = estimate_workload_phase_peaks(
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

        peaks, active, used, _ = estimate_workload_phase_peaks(
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

        peaks, active, used, _ = estimate_workload_phase_peaks(
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
