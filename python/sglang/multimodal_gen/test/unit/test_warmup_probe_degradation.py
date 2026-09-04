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
