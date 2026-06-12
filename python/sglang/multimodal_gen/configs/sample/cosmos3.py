# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 sampling parameters.

A single ``SamplingParams`` class serves T2V, I2V, and T2I — the per-request
mode is dispatched in the pipeline from ``num_frames`` (``== 1`` → T2I) and
``image_path`` (set → I2V). For ``num_frames == 1`` the output ``data_type``
flips to ``IMAGE`` so the file extension and decode path agree.
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class Cosmos3SamplingParams(SamplingParams):
    """Cosmos3 sampling parameters (T2V defaults; also used for I2V / T2I)."""

    height: int = 720
    width: int = 1280
    num_frames: int = 81
    fps: int = 24

    guidance_scale: float = 4.0
    num_inference_steps: int = 35

    negative_prompt: str = ""

    generate_sound: bool = False

    # Action conditioning (Policy checkpoints). Mode is one of forward_dynamics,
    # inverse_dynamics, policy. The embodiment selects the per-domain action
    # heads. Normalization stats are dataset-derived and optional. Without a
    # stats file actions stay in the model's normalized [-1, 1] space.
    action_mode: str | None = None
    action_embodiment: str = "droid_lerobot"
    action_chunk_size: int = 16
    action_path: str | None = None
    action_stats_path: str | None = None
    action_normalization: str = "quantile"
    action_view_point: str = "ego_view"
    action_resolution: str = "480"

    # Video-to-video (video2video): condition on the first latent frames of the
    # input video given via ``image_path`` (a video file), then generate the
    # rest. ``None`` falls back to [0] for an image (I2V) and [0, 1] for a video
    # (V2V), matching the reference DEFAULT_CONDITION_FRAME_INDEXES_VISION.
    condition_frame_indexes_vision: list[int] | None = None

    # T2I defaults to the native Cosmos3 CFG window below. T2V / I2V leave it
    # unset for full-range CFG.
    guidance_interval: tuple[float, float] | None = None

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1280, 720),
            (720, 1280),
            (832, 480),
            (480, 832),
            (1024, 1024),
        ]
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # T2I skips CFG below t=400 (the native [400, 1000] window), halving the
        # low-noise forwards at no quality cost.
        if self.num_frames == 1 and self.guidance_interval is None:
            self.guidance_interval = (400.0, 1000.0)

    def _set_output_file_name(self) -> None:
        # The pipeline config's ``task_type=TI2V`` drives ``data_type`` to
        # VIDEO, but a single-frame request is a T2I and must pick the IMAGE
        # extension. Flip before the base derives the file name.
        if self.num_frames == 1:
            self.data_type = DataType.IMAGE
        super()._set_output_file_name()
