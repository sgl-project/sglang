# SPDX-License-Identifier: Apache-2.0
"""Unit tests for warmup-calibrated auto residency adjustment."""

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.longlive2 import LongLive2T2VConfig
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
    ACTIVATION_EXTRAPOLATION_MARGIN,
    GIB_BYTES,
    WarmupMemoryRecord,
    estimate_default_workload_peak_bytes,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.warmup_request_builder import (
    SERVER_WARMUP_MAX_VIDEO_FRAMES,
    SERVER_WARMUP_VIDEO_STEPS,
    _resolve_auto_residency_warmup_shape,
    _resolve_warmup_num_frames,
    build_warmup_reqs,
)

PLANNER_UNAVAILABLE = "residency planner not available"


def _record(
    *,
    width=832,
    height=480,
    num_frames=17,
    baseline_gib=10,
    peak_gib=12,
    succeeded=True,
) -> WarmupMemoryRecord:
    return WarmupMemoryRecord(
        width=width,
        height=height,
        num_frames=num_frames,
        baseline_allocated_bytes=baseline_gib * GIB_BYTES,
        peak_allocated_bytes=peak_gib * GIB_BYTES,
        succeeded=succeeded,
    )


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
        monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
        reason = self._skip_reason(self._base_args(**overrides))
        assert reason == PLANNER_UNAVAILABLE

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
    def test_parallel_paths_reach_planner_gate(self, monkeypatch, overrides):
        monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY", raising=False)
        monkeypatch.delenv("SGLANG_CACHE_DIT_ENABLED", raising=False)
        monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
        reason = self._skip_reason(self._base_args(**overrides))
        assert reason == PLANNER_UNAVAILABLE

    def test_eligible_path_reaches_planner_gate(self, monkeypatch):
        monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY", raising=False)
        monkeypatch.delenv("SGLANG_CACHE_DIT_ENABLED", raising=False)
        monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
        reason = self._skip_reason(self._base_args())
        assert reason == PLANNER_UNAVAILABLE

    def test_eligible_server_warmup_runs_no_probe_or_step_floor(self, monkeypatch):
        # no calibration consumer yet: one bounded warmup, not probe + 4-step floor
        monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY", raising=False)
        monkeypatch.delenv("SGLANG_CACHE_DIT_ENABLED", raising=False)
        monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
        args = self._base_args(
            warmup_steps=1,
            enable_cfg_parallel=False,
            num_gpus=1,
            pipeline_config=SimpleNamespace(
                task_type=ModelTaskType.T2V,
                supports_auto_residency=True,
                adjust_num_frames=lambda value: value,
                vae_stride=None,
                vae_scale_factor=None,
                vae_config=SimpleNamespace(arch_config=None),
            ),
            is_arg_explicitly_set=lambda _name: False,
        )
        monkeypatch.setattr(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            lambda _args: SamplingParams(
                width=1280,
                height=720,
                num_frames=81,
                num_inference_steps=35,
                adjust_frames=False,
                supported_resolutions=[(1280, 720), (832, 480)],
            ),
        )
        reqs = build_warmup_reqs(
            args, warmup_resolutions=None, server_based_warmup=True
        )
        assert len(reqs) == 1
        assert not reqs[0].extra.get("auto_residency_full_shape_probe")
        assert reqs[0].num_inference_steps == SERVER_WARMUP_VIDEO_STEPS
