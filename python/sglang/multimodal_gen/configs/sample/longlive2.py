# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.wan import Wan2_2_TI2V_5B_SamplingParam


@dataclass
class LongLive2SamplingParams(Wan2_2_TI2V_5B_SamplingParam):
    height: int = 704
    width: int = 1280
    fps: int = 24
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    num_frames: int = 61
    shot_prompts: list[str] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    shot_durations: list[int] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    chunks_per_shot: int = field(default=0, metadata={"batch_sig_exclude": True})
    scene_cut_prefix: str = field(
        default="The scene transitions. ", metadata={"batch_sig_exclude": True}
    )
    multi_shot_sink: bool = field(default=True, metadata={"batch_sig_exclude": True})
    multi_shot_rope_offset: float = field(
        default=8.0, metadata={"batch_sig_exclude": True}
    )

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1280, 704),
            (704, 1280),
            (832, 480),
            (480, 832),
        ]
    )

    def _validate(self):
        super()._validate()

        if self.shot_prompts is not None:
            if not isinstance(self.shot_prompts, list) or not self.shot_prompts:
                raise ValueError("shot_prompts must be a non-empty list of strings")
            if not all(
                isinstance(prompt, str) and prompt for prompt in self.shot_prompts
            ):
                raise ValueError("shot_prompts must contain non-empty strings")

        if self.shot_durations is not None:
            if not isinstance(self.shot_durations, list) or not self.shot_durations:
                raise ValueError("shot_durations must be a non-empty list of ints")
            if self.shot_prompts is not None and len(self.shot_durations) != len(
                self.shot_prompts
            ):
                raise ValueError("shot_durations must match shot_prompts length")
            if not all(
                isinstance(duration, int) and duration > 0
                for duration in self.shot_durations
            ):
                raise ValueError("shot_durations must contain positive ints")

        if self.chunks_per_shot < 0:
            raise ValueError("chunks_per_shot must be non-negative")

        if self.scene_cut_prefix is None:
            self.scene_cut_prefix = ""
        if self.multi_shot_rope_offset < 0:
            raise ValueError("multi_shot_rope_offset must be non-negative")

    def _adjust(self, server_args):
        if self.shot_prompts is not None and self.prompt is None:
            self.prompt = self.shot_prompts[0]
        super()._adjust(server_args)
