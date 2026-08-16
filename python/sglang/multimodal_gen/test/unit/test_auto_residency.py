# SPDX-License-Identifier: Apache-2.0
"""Unit tests for warmup-calibrated auto residency promotion."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.longlive2 import LongLive2T2VConfig
from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
    ACTIVATION_EXTRAPOLATION_MARGIN,
    GIB_BYTES,
    MIN_VRAM_RESERVE_BYTES,
    AutoResidencyPlan,
    PromotionCandidate,
    RankResidencyReport,
    WarmupMemoryRecord,
    apply_promotions,
    collect_promotion_candidates,
    component_resident_size_bytes,
    estimate_default_workload_peak_bytes,
    plan_auto_residency,
    rollback_promotions,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
    RESIDENT,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.warmup_request_builder import (
    SERVER_WARMUP_MAX_VIDEO_FRAMES,
    _resolve_calibration_num_frames,
    _resolve_warmup_num_frames,
)


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
        peak_reserved_bytes=peak_gib * GIB_BYTES,
        succeeded=succeeded,
    )


class TestEstimateDefaultWorkloadPeak:
    def test_same_shape_uses_measured_peak(self):
        record = _record()
        estimate = estimate_default_workload_peak_bytes(
            records=[record], target_units=record.workload_units()
        )
        assert estimate == record.peak_reserved_bytes

    def test_unknown_target_uses_measured_peak(self):
        record = _record()
        estimate = estimate_default_workload_peak_bytes(
            records=[record], target_units=None
        )
        assert estimate == record.peak_reserved_bytes

    def test_single_capped_record_scales_only_the_activation_part(self):
        # Warmup capped to 832x480x17; Wan-class default is 704x1280x121.
        record = _record()
        target_units = 704 * 1280 * 121
        ratio = target_units / record.workload_units()
        assert ratio > 10  # the cap ratio this formula exists for

        estimate = estimate_default_workload_peak_bytes(
            records=[record], target_units=target_units
        )
        activation = record.peak_reserved_bytes - record.baseline_allocated_bytes
        expected = record.baseline_allocated_bytes + int(
            activation * ratio * ACTIVATION_EXTRAPOLATION_MARGIN
        )
        assert estimate == expected
        # Scaling the whole peak would inflate the estimate by the resident
        # weights times the cap ratio and promotion would never trigger.
        naive = int(record.peak_reserved_bytes * ratio)
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
        slope = (large.peak_reserved_bytes - small.peak_reserved_bytes) / (
            large.workload_units() - small.workload_units()
        )
        constant = large.peak_reserved_bytes - slope * large.workload_units()
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

    def test_covering_measurement_bounds_the_target(self):
        capped = _record(num_frames=17, peak_gib=12)
        full = _record(width=1280, height=720, num_frames=81, peak_gib=30)
        estimate = estimate_default_workload_peak_bytes(
            records=[capped, full], target_units=1280 * 720 * 81
        )
        assert estimate == full.peak_reserved_bytes

    def test_multiple_records_take_the_max(self):
        low = _record(peak_gib=12)
        high = _record(peak_gib=20)
        estimate = estimate_default_workload_peak_bytes(
            records=[low, high], target_units=low.workload_units()
        )
        assert estimate == high.peak_reserved_bytes

    def test_failed_warmup_disables_estimation(self):
        records = [_record(), _record(succeeded=False)]
        assert (
            estimate_default_workload_peak_bytes(records=records, target_units=None)
            is None
        )

    def test_no_records_disables_estimation(self):
        assert (
            estimate_default_workload_peak_bytes(records=[], target_units=None) is None
        )


def _candidate(
    name: str,
    *,
    mode: str = COMPONENT_OFFLOAD,
    weight_gib: int = 10,
    h2d_gib: int | None = None,
) -> PromotionCandidate:
    return PromotionCandidate(
        component_name=name,
        residency_mode=mode,
        promoted_weight_bytes=weight_gib * GIB_BYTES,
        h2d_bytes_per_request=(h2d_gib if h2d_gib is not None else weight_gib)
        * GIB_BYTES,
    )


def _report(
    *,
    rank: int = 0,
    budget_gib: int = 100,
    estimated_gib: int | None = 50,
    candidates: list[PromotionCandidate] | None = None,
    skip_reason: str | None = None,
) -> RankResidencyReport:
    return RankResidencyReport(
        rank=rank,
        budget_bytes=budget_gib * GIB_BYTES,
        estimated_peak_bytes=(
            None if estimated_gib is None else estimated_gib * GIB_BYTES
        ),
        candidates=candidates if candidates is not None else [_candidate("vae")],
        skip_reason=skip_reason,
    )


class TestPlanAutoResidency:
    def test_rank_skip_reason_propagates(self):
        plan = plan_auto_residency(
            reports=[_report(), _report(rank=1, skip_reason="no measurements")]
        )
        assert plan.skip_reason == "rank 1: no measurements"
        assert not plan.promotions

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

    def test_greedy_promotion_by_h2d_savings(self):
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
        assert [c.component_name for c in plan.promotions] == ["dit"]

    def test_smaller_component_still_fits_after_skipping_a_big_one(self):
        candidates = [
            _candidate("text_encoder", weight_gib=45, h2d_gib=100),
            _candidate("vae", weight_gib=5, h2d_gib=5),
        ]
        plan = plan_auto_residency(
            reports=[_report(budget_gib=100, estimated_gib=50, candidates=candidates)]
        )
        assert [c.component_name for c in plan.promotions] == ["vae"]

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
        names = [c.component_name for c in plan.promotions]
        assert names == ["vae"]
        # worst-case size across ranks
        assert plan.promotions[0].promoted_weight_bytes == 12 * GIB_BYTES

    def test_no_candidates_skips(self):
        plan = plan_auto_residency(reports=[_report(candidates=[])])
        assert plan.skip_reason is not None


class _FakeLayerwiseManager:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.enabled = True
        self._configured = True
        self._tensors = tensors
        self.load_all_layers_calls = 0
        self.remove_hooks_calls = 0
        self.register_hooks_calls = 0
        self.sync_to_cpu_calls = 0
        self.release_all_calls = 0

    def iter_cpu_weights(self):
        yield from self._tensors.items()

    def load_all_layers(self):
        self.load_all_layers_calls += 1

    def remove_forward_hooks(self):
        self.remove_hooks_calls += 1

    def register_forward_hooks(self):
        self.register_hooks_calls += 1

    def sync_all_layers_to_cpu(self):
        self.sync_to_cpu_calls += 1

    def release_all(self):
        self.release_all_calls += 1


class _FakeLayerwiseDit(LayerwiseOffloadableModuleMixin, nn.Module):
    def __init__(self, managers: list[_FakeLayerwiseManager]):
        nn.Module.__init__(self)
        self.layerwise_offload_managers = managers


class _StubResidencyArgs:
    """Duck-typed stand-in for the two ServerArgs hooks promotions use."""

    def __init__(self):
        self.required: set[str] = set()

    def require_component_resident(self, component_name, *, feature_name):
        self.required.add(component_name)

    def release_required_component_residency(self, component_name):
        self.required.discard(component_name)


def _plan_for(candidates: list[PromotionCandidate]) -> AutoResidencyPlan:
    return AutoResidencyPlan(
        estimated_peak_bytes=GIB_BYTES,
        reserve_bytes=MIN_VRAM_RESERVE_BYTES,
        budget_bytes=100 * GIB_BYTES,
        promotions=candidates,
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


class TestCollectPromotionCandidates:
    def _modes(self, mapping):
        return lambda name: mapping.get(name, RESIDENT)

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
        candidates = collect_promotion_candidates(
            modules=modules,
            residency_mode_of=modes,
            explicit_residency_mode_of=lambda name: (
                COMPONENT_OFFLOAD if name == "explicit_encoder" else None
            ),
            custom_strategy_names={"custom"},
            num_inference_steps=50,
        )
        by_name = {c.component_name: c for c in candidates}
        # resident vae, explicit placement, custom strategy, non-module: out
        assert set(by_name) == {"text_encoder", "transformer"}
        assert by_name["text_encoder"].h2d_bytes_per_request == (
            by_name["text_encoder"].promoted_weight_bytes
        )
        # layerwise DiT re-streams its layers once per denoise step
        assert by_name["transformer"].h2d_bytes_per_request == (
            by_name["transformer"].promoted_weight_bytes * 50
        )


class TestApplyAndRollback:
    def test_component_offload_promotion_marks_resident_without_moving(self):
        module = nn.Linear(4, 4)
        args = _StubResidencyArgs()
        applied = apply_promotions(
            plan=_plan_for([_candidate("text_encoder", weight_gib=1)]),
            modules={"text_encoder": module},
            server_args=args,
        )
        assert args.required == {"text_encoder"}
        assert [p.component_name for p in applied] == ["text_encoder"]

        rollback_promotions(
            applied=applied, modules={"text_encoder": module}, server_args=args
        )
        assert args.required == set()

    def test_layerwise_promotion_loads_all_layers_and_rollback_rearms(self):
        manager = _FakeLayerwiseManager({"layers.0.w": torch.zeros(16)})
        module = _FakeLayerwiseDit([manager])
        args = _StubResidencyArgs()
        candidate = _candidate("transformer", mode=LAYERWISE_OFFLOAD, weight_gib=1)

        applied = apply_promotions(
            plan=_plan_for([candidate]),
            modules={"transformer": module},
            server_args=args,
        )
        assert manager.load_all_layers_calls == 1
        assert manager.remove_hooks_calls == 1
        assert manager.enabled is False

        rollback_promotions(
            applied=applied, modules={"transformer": module}, server_args=args
        )
        assert manager.enabled is True
        assert manager.register_hooks_calls == 1
        assert args.required == set()

    def test_mid_failure_rolls_back_already_applied_promotions(self):
        manager = _FakeLayerwiseManager({"layers.0.w": torch.zeros(16)})
        module = _FakeLayerwiseDit([manager])
        args = _StubResidencyArgs()
        candidates = [
            _candidate("transformer", mode=LAYERWISE_OFFLOAD, weight_gib=1),
            _candidate("missing_component", weight_gib=1),
        ]
        with pytest.raises(RuntimeError, match="missing_component"):
            apply_promotions(
                plan=_plan_for(candidates),
                modules={"transformer": module},
                server_args=args,
            )
        assert args.required == set()
        assert manager.enabled is True
        assert manager.register_hooks_calls == 1

    def test_enable_offload_does_not_double_register_hooks(self):
        manager = _FakeLayerwiseManager({"layers.0.w": torch.zeros(16)})
        module = _FakeLayerwiseDit([manager])
        # never disabled: re-enabling must not stack a second set of hooks
        module.enable_offload()
        assert manager.register_hooks_calls == 0


class TestWarmupFrameAdjustment:
    def _server_args(self, *, bcg: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            pipeline_config=LongLive2T2VConfig(),
            enable_breakable_cuda_graph=bcg,
        )

    def _defaults(self) -> SimpleNamespace:
        return SimpleNamespace(num_frames=61)

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


class TestCalibrationFrames:
    def _wan_like_args(self, *, performance_mode: str = "auto") -> SimpleNamespace:
        return SimpleNamespace(
            pipeline_config=SimpleNamespace(
                task_type=ModelTaskType.T2V,
                adjust_num_frames=lambda n: n,
            ),
            enable_breakable_cuda_graph=False,
            performance_mode=performance_mode,
        )

    def test_capped_video_gets_a_smaller_calibration_size(self):
        calibration = _resolve_calibration_num_frames(
            self._wan_like_args(),
            SimpleNamespace(num_frames=81),
            17,
            server_based_warmup=True,
        )
        assert calibration == 9

    def test_uncapped_video_needs_no_calibration(self):
        calibration = _resolve_calibration_num_frames(
            self._wan_like_args(),
            SimpleNamespace(num_frames=13),
            13,
            server_based_warmup=True,
        )
        assert calibration is None

    def test_manual_mode_skips_calibration(self):
        calibration = _resolve_calibration_num_frames(
            self._wan_like_args(performance_mode="manual"),
            SimpleNamespace(num_frames=81),
            17,
            server_based_warmup=True,
        )
        assert calibration is None

    def test_frame_contract_collapse_skips_calibration(self):
        # LongLive2 aligns both 17 and 9 to 29 latent-block frames: a second
        # measurement at the same size adds nothing.
        args = SimpleNamespace(
            pipeline_config=LongLive2T2VConfig(),
            enable_breakable_cuda_graph=False,
            performance_mode="auto",
        )
        calibration = _resolve_calibration_num_frames(
            args, SimpleNamespace(num_frames=61), 29, server_based_warmup=True
        )
        assert calibration is None


class TestAutoResidencySkipReason:
    def _base_args(self, **overrides) -> SimpleNamespace:
        args = SimpleNamespace(
            performance_mode="auto",
            warmup_mode="server",
            warmup_resolutions=None,
            backend="sglang",
            enable_breakable_cuda_graph=False,
            batching_max_size=1,
            dp_size=1,
            use_fsdp_inference=False,
            quantization=None,
            transformer_weights_path=None,
            nunchaku_config=None,
            pipeline_config=SimpleNamespace(task_type=ModelTaskType.T2V),
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
            ({"backend": "diffusers"}, "diffusers"),
            ({"enable_breakable_cuda_graph": True}, "CUDA graph"),
            ({"batching_max_size": 4}, "batching"),
            ({"dp_size": 2}, "dp replicas"),
            ({"use_fsdp_inference": True}, "FSDP"),
            ({"quantization": "fp8"}, "quantized"),
            ({"transformer_weights_path": "/x.safetensors"}, "quantized"),
        ],
    )
    def test_excluded_paths(self, monkeypatch, overrides, expected_fragment):
        monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY", raising=False)
        monkeypatch.delenv("SGLANG_CACHE_DIT_ENABLED", raising=False)
        reason = self._skip_reason(self._base_args(**overrides))
        assert reason is not None and expected_fragment in reason

    def test_cache_dit_excluded(self, monkeypatch):
        monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY", raising=False)
        monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")
        reason = self._skip_reason(self._base_args())
        assert reason is not None and "cache-dit" in reason

    def test_eligible_path_reaches_platform_gate(self, monkeypatch):
        monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY", raising=False)
        monkeypatch.delenv("SGLANG_CACHE_DIT_ENABLED", raising=False)
        reason = self._skip_reason(self._base_args())
        # on a CUDA host everything passes; CPU CI stops at the platform gate
        assert reason is None or reason == "requires CUDA"
