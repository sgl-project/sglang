# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams sampling parameters.

Forward dynamics only: a conditioning image, a prompt, and per-pixel-frame
actions produce a video. ``action`` rows follow the checkpoint's action
contract (camera pose: 9-D ``[translation(3), rot6d(6)]`` deltas, one row per
pixel frame after the first); ``domain_id`` / ``domain_name`` pick the
embodiment, defaulting to the contract's default embodiment. The prompt is
wrapped in the structured JSON caption the checkpoint was trained on unless
``format_prompt_as_json`` is disabled. The image is fitted into a trained
canvas with reflection padding, but like training only the content region is
generated, so the output size is the fitted image rounded down to the VAE
stride (for example 768x480 for a 16:10 image on the 832x480 canvas).
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams

# Reference inference default and the mRoPE base rate; training clips kept their
# native frame rate, so no single value is "the" trained fps.
COSMOS_DREAMS_DEFAULT_FPS = 24

# Canvases of the 480 tier the checkpoint was trained on, widest-landscape first.
COSMOS_DREAMS_480P_CANVASES: tuple[tuple[int, int], ...] = (
    (832, 480),
    (480, 832),
    (640, 640),
    (736, 544),
    (544, 736),
)


@dataclass
class CosmosDreamsSamplingParams(Cosmos3SamplingParams):
    fps: int = COSMOS_DREAMS_DEFAULT_FPS
    num_frames: int = 81
    # Guidance is distilled into the weights; the rollout runs one branch.
    guidance_scale: float = 1.0

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: list(COSMOS_DREAMS_480P_CANVASES)
    )

    # Wrap the prompt in the training-time JSON caption (framing, description,
    # resolution, aspect ratio). Disable to tokenize the prompt verbatim.
    format_prompt_as_json: bool = True

    # Set during adjustment when the request left height/width unset: the image
    # stage then snaps the canvas to the trained size closest to the image aspect.
    canvas_from_image: bool = False

    @classmethod
    def video_request_extra_fields(cls) -> frozenset[str]:
        return super().video_request_extra_fields() | frozenset(
            {"format_prompt_as_json"}
        )

    def _validate(self) -> None:
        super()._validate()
        if self.video_path is not None:
            raise ValueError(
                "Cosmos-Dreams conditions on a single image; video_path is not supported."
            )
        if self._resolve_control_paths():
            raise ValueError(
                "Cosmos-Dreams does not support transfer (control_path) inference."
            )
        if float(self.sound_duration or 0.0) > 0.0:
            raise ValueError("Cosmos-Dreams does not generate sound.")
        if self.action_mode not in (None, "forward_dynamics"):
            raise ValueError(
                "Cosmos-Dreams only runs forward dynamics (actions condition the video); "
                f"got action_mode={self.action_mode!r}."
            )
        if self.num_frames == 1:
            raise ValueError("Cosmos-Dreams generates video; num_frames must exceed 1.")
        if isinstance(self.image_path, list) and len(self.image_path) > 1:
            raise ValueError("Cosmos-Dreams accepts exactly one conditioning image.")
        if not isinstance(self.format_prompt_as_json, bool):
            raise ValueError("format_prompt_as_json must be a boolean.")

    def _adjust(self, server_args) -> None:
        # The base adjustment fills a default canvas below; remember whether the
        # request left it open so the image can pick the canvas instead.
        self.canvas_from_image = self.height is None and self.width is None
        super()._adjust(server_args)
        # Guidance is distilled into the weights and the step count is fixed by
        # the checkpoint's sampler; the rollout stage ignores other values.
        if self.guidance_scale != 1.0:
            raise ValueError(
                f"Cosmos-Dreams distilled inference requires guidance_scale=1.0, got {self.guidance_scale}."
            )
