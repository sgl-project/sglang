"""A warmup probe that does not fit is retried smaller instead of abandoned."""

from types import SimpleNamespace

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.warmup_request_builder import lighten_warmup_req


def _server_args(temporal_compression_ratio: int = 4) -> SimpleNamespace:
    arch_config = SimpleNamespace(
        temporal_compression_ratio=temporal_compression_ratio,
        vae_scale_factor=8,
        spatial_compression_ratio=8,
    )
    return SimpleNamespace(
        pipeline_class_name=None,
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(arch_config=arch_config),
            vae_scale_factor=8,
        ),
    )


def _req(width: int, height: int, num_frames: int) -> Req:
    return Req(
        sampling_params=SamplingParams(
            width=width, height=height, num_frames=num_frames
        )
    )


class TestLightenWarmupReq:
    def test_video_probe_halves_latent_frames_first(self):
        lighter = lighten_warmup_req(_server_args(), _req(832, 480, 17))
        assert (lighter.width, lighter.height) == (832, 480)
        assert lighter.num_frames == 9

    def test_frames_step_down_to_a_single_frame(self):
        server_args = _server_args()
        frames = []
        req = _req(832, 480, 17)
        for _ in range(4):
            req = lighten_warmup_req(server_args, req)
            if req is None:
                break
            frames.append(req.num_frames)
        assert frames[:3] == [9, 5, 1]

    def test_image_probe_halves_the_area(self):
        lighter = lighten_warmup_req(_server_args(), _req(1024, 1024, 1))
        assert lighter.num_frames == 1
        assert lighter.width * lighter.height <= 1024 * 1024 // 2
        assert lighter.width % 16 == 0 and lighter.height % 16 == 0

    def test_the_original_request_is_left_alone(self):
        req = _req(832, 480, 17)
        lighten_warmup_req(_server_args(), req)
        assert req.num_frames == 17

    def test_a_probe_at_the_floor_cannot_shrink(self):
        assert lighten_warmup_req(_server_args(), _req(16, 16, 1)) is None

    def test_frames_follow_the_model_frame_contract(self):
        # LongLive2-style contract: latent frames come in causal blocks of 8,
        # so with a temporal ratio of 4 only 29, 61, 93, ... frames are valid.
        server_args = _server_args()

        def adjust_num_frames(num_frames: int) -> int:
            latent = (num_frames - 1) // 4 + 1
            if latent % 8 == 0:
                return num_frames
            return (max(8, latent // 8 * 8) - 1) * 4 + 1

        server_args.pipeline_config.adjust_num_frames = adjust_num_frames

        lighter = lighten_warmup_req(server_args, _req(960, 928, 61))
        assert lighter.num_frames == 29
        assert (lighter.width, lighter.height) == (960, 928)

        # At the smallest valid frame count the probe shrinks the area instead.
        floor = lighten_warmup_req(server_args, lighter)
        assert floor.num_frames == 29
        assert floor.width * floor.height <= 960 * 928 // 2


def _record(width: int, height: int, num_frames: int, *, peak_gib: float):
    from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
        WarmupMemoryRecord,
    )

    return WarmupMemoryRecord(
        width=width,
        height=height,
        num_frames=num_frames,
        baseline_allocated_bytes=2 << 30,
        peak_allocated_bytes=int(peak_gib * (1 << 30)),
        succeeded=True,
    )


class TestFitAutoResidencyProbe:
    def test_probe_shrinks_until_its_extrapolated_peak_fits(self):
        from sglang.multimodal_gen.runtime.managers.gpu_worker import (
            fit_auto_residency_probe,
        )

        fitted, estimate, steps = fit_auto_residency_probe(
            _req(1280, 720, 81),
            records=[_record(832, 480, 81, peak_gib=20.0)],
            free_bytes=40 << 30,
            total_bytes=80 << 30,
            server_args=_server_args(),
        )
        assert steps >= 1
        assert (fitted.width, fitted.height) == (1280, 720)
        assert fitted.num_frames < 81
        assert estimate is not None and estimate <= 40 << 30

    def test_probe_that_fits_runs_at_full_shape(self):
        from sglang.multimodal_gen.runtime.managers.gpu_worker import (
            fit_auto_residency_probe,
        )

        fitted, _, steps = fit_auto_residency_probe(
            _req(1280, 720, 81),
            records=[_record(832, 480, 81, peak_gib=20.0)],
            free_bytes=79 << 30,
            total_bytes=80 << 30,
            server_args=_server_args(),
        )
        assert steps == 0
        assert fitted.num_frames == 81

    def test_probe_never_shrinks_below_the_bounded_warmup_shape(self):
        from sglang.multimodal_gen.runtime.managers.gpu_worker import (
            fit_auto_residency_probe,
        )

        # Nothing fits the extrapolation, but the bounded 832x480x17f warmup
        # already ran, so the ladder (81 -> 41 -> 21 -> 9 frames) stops at the
        # first shape at or below it instead of reaching a 16x16x1f probe.
        fitted, _, steps = fit_auto_residency_probe(
            _req(832, 480, 81),
            records=[_record(832, 480, 17, peak_gib=30.0)],
            free_bytes=8 << 30,
            total_bytes=80 << 30,
            server_args=_server_args(),
        )
        assert steps >= 1
        assert (fitted.width, fitted.height) == (832, 480)
        assert fitted.num_frames == 9

    def test_without_a_trusted_estimate_the_probe_runs_as_requested(self):
        from sglang.multimodal_gen.runtime.managers.gpu_worker import (
            fit_auto_residency_probe,
        )

        fitted, estimate, steps = fit_auto_residency_probe(
            _req(1280, 720, 81),
            records=[],
            free_bytes=1 << 30,
            total_bytes=80 << 30,
            server_args=_server_args(),
        )
        assert (steps, estimate) == (0, None)
        assert fitted.num_frames == 81


class TestOutOfMemoryClassification:
    def test_allocation_failures_from_libraries_count_as_out_of_memory(self):
        from sglang.multimodal_gen.runtime.server_warmup import _is_out_of_memory

        assert _is_out_of_memory("CUDA error: out of memory")
        assert _is_out_of_memory("cuBLAS error: CUBLAS_STATUS_ALLOC_FAILED")
        assert _is_out_of_memory(
            "RuntimeError: cudaErrorMemoryAllocation: out of memory"
        )
        assert not _is_out_of_memory("shape mismatch in attention")
