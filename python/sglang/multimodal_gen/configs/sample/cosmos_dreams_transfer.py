# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams-Transfer sampling parameters.

Control-video transfer only: one pre-computed control clip (``control_path``)
of one hint type (``control_hint``: edge, blur, depth, or seg) plus a prompt
produce a video whose length follows the control clip. The target is generated
from noise (no conditioning image), guidance is distilled into the weights,
and the caption receives the training-time duration/resolution metadata plus
the control-adherence sentence.
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import TRANSFER_HINTS
from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.configs.sample.cosmos_dreams import (
    COSMOS_DREAMS_480P_CANVASES,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

# Longest clip of the legacy chunkwise 480p Transfer recipe, in pixel frames.
COSMOS_DREAMS_TRANSFER_MAX_FRAMES = 601


@dataclass
class CosmosDreamsTransferSamplingParams(Cosmos3SamplingParams):
    # Cap on control frames consumed; the clip is trimmed to the chunk partition.
    num_frames: int = COSMOS_DREAMS_TRANSFER_MAX_FRAMES
    # Replaced by the control clip's frame rate unless the request sets it.
    fps: int = 24
    # Guidance is distilled into the weights; the rollout runs one branch.
    guidance_scale: float = 1.0

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: list(COSMOS_DREAMS_480P_CANVASES)
    )

    # Append the control-adherence sentence to the prompt; None follows the
    # checkpoint contract.
    emphasize_control_in_prompt: bool | None = None

    # Set during adjustment when the request left height/width unset: the canvas
    # is then snapped to the trained size closest to the control clip's aspect.
    canvas_from_control: bool = False

    @classmethod
    def video_request_extra_fields(cls) -> frozenset[str]:
        return super().video_request_extra_fields() | frozenset(
            {"emphasize_control_in_prompt"}
        )

    @property
    def resolved_control_path(self) -> str:
        paths = self._resolve_control_paths()
        if len(paths) != 1:
            raise ValueError(
                "Cosmos-Dreams-Transfer requires exactly one control_path (a pre-computed "
                f"control video), got {len(paths)}."
            )
        return paths[0]

    @property
    def resolved_control_hint(self) -> str:
        hints = self._resolve_control_hints()
        if len(hints) != 1:
            raise ValueError(
                "Cosmos-Dreams-Transfer requires exactly one control_hint "
                f"({', '.join(TRANSFER_HINTS)}), got {len(hints)}."
            )
        if hints[0] not in TRANSFER_HINTS:
            raise ValueError(
                f"Cosmos-Dreams-Transfer control_hint must be one of {list(TRANSFER_HINTS)}, "
                f"got {hints[0]!r}."
            )
        return hints[0]

    def is_explicit(self, name: str) -> bool:
        """Whether the request set ``name`` itself (tracked as in Cosmos3SamplingParams)."""
        explicit = getattr(self, "_explicit_fields", None)
        return explicit is not None and name in explicit

    def _validate(self) -> None:
        super()._validate()
        # Default construction (offload planning) carries no control input; a
        # request must, which _validate_with_pipeline_config enforces.
        if self._resolve_control_paths() or self._resolve_control_hints():
            _ = self.resolved_control_path
            _ = self.resolved_control_hint
        if self.image_path is not None:
            raise ValueError(
                "Cosmos-Dreams-Transfer generates the target from noise; image_path is not supported."
            )
        if self.video_path is not None:
            raise ValueError(
                "Cosmos-Dreams-Transfer takes pre-computed control clips only; video_path is not supported."
            )
        if self.action is not None or self.action_mode is not None:
            raise ValueError("Cosmos-Dreams-Transfer does not take actions.")
        if float(self.sound_duration or 0.0) > 0.0:
            raise ValueError("Cosmos-Dreams-Transfer does not generate sound.")
        if self.num_first_chunk_conditional_frames != 0:
            raise ValueError(
                "Cosmos-Dreams-Transfer requires num_first_chunk_conditional_frames=0."
            )
        if float(self.control_guidance) != 1.0:
            raise ValueError("Cosmos-Dreams-Transfer requires control_guidance=1.0.")
        if self.num_frames == 1:
            raise ValueError(
                "Cosmos-Dreams-Transfer generates video; num_frames must exceed 1."
            )
        if self.emphasize_control_in_prompt is not None and not isinstance(
            self.emphasize_control_in_prompt, bool
        ):
            raise ValueError("emphasize_control_in_prompt must be a boolean or null.")

    def _validate_with_pipeline_config(self, pipeline_config) -> None:
        super()._validate_with_pipeline_config(pipeline_config)
        _ = self.resolved_control_path
        _ = self.resolved_control_hint

    def _adjust(self, server_args) -> None:
        self.canvas_from_control = self.height is None and self.width is None
        # Cosmos3SamplingParams._adjust rejects transfer on distilled checkpoints
        # and applies the bidirectional per-hint guidance defaults; the Dreams
        # rollout is fixed-step and unguided, so only the base adjustment applies.
        SamplingParams._adjust(self, server_args)
        distilled_sigmas = server_args.pipeline_config.distilled_sigmas
        if distilled_sigmas is None:
            raise ValueError(
                "Cosmos-Dreams-Transfer requires a distilled fixed-step scheduler."
            )
        self.num_inference_steps = len(distilled_sigmas)
        if self.is_explicit("guidance_scale") and self.guidance_scale != 1.0:
            raise ValueError(
                f"Cosmos-Dreams-Transfer distilled inference requires guidance_scale=1.0, got {self.guidance_scale}."
            )
        self.guidance_scale = 1.0
