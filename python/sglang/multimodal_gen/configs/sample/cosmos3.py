# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 sampling parameters.

A single ``SamplingParams`` class serves T2V, I2V, V2V, T2I, and
action-conditioned variants.  Per-request mode is dispatched in the pipeline
from ``num_frames`` (``== 1`` → T2I), ``image_path`` (set → I2V),
``video_path`` (set → V2V), and ``action_mode`` (set → action-conditioned).
For ``num_frames == 1`` the output ``data_type`` flips to ``IMAGE``
so the file extension and decode path agree.
"""

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)

COSMOS3_DEFAULT_GUIDANCE_SCALE = 4.0
COSMOS3_EDGE_T2I_GUIDANCE_SCALE = 7.0
COSMOS3_EDGE_T2V_GUIDANCE_SCALE = 5.0
COSMOS3_EDGE_T2V_WIDTH = 832
COSMOS3_EDGE_T2V_HEIGHT = 480
COSMOS3_EDGE_T2I_SIZE = 640

# Image generation applies guidance only over a high-noise window; guiding the
# low-noise steps degrades sample quality (Kynkaeaenniemi et al. 2024). Video
# modes guide at every step.
COSMOS3_T2I_GUIDANCE_INTERVAL = (400.0, 1000.0)

# Edge is trained at 256p/480p only; larger frames push the spatial mRoPE grid
# past its trained range and shatter the output.
COSMOS3_EDGE_SUPPORTED_RESOLUTIONS = [
    (832, 480),
    (480, 832),
    (640, 480),
    (480, 640),
    (480, 480),
    (640, 640),
    (448, 256),
    (256, 448),
    (256, 256),
]


@dataclass
class Cosmos3SamplingParams(SamplingParams):
    """Cosmos3 sampling parameters (T2V defaults; also used for I2V / V2V / T2I).

    ``height``/``width`` default to ``None`` so the variant (Edge vs. base) can
    pick the right resolution at request time in
    :meth:`_resolve_variant_defaults`.
    """

    height: int | None = None
    width: int | None = None
    num_frames: int = 81
    fps: int = 24

    guidance_scale: float = COSMOS3_DEFAULT_GUIDANCE_SCALE
    num_inference_steps: int = 35

    negative_prompt: str = ""

    # Optional CFG window — T2I requests typically pass e.g. ``(400, 1000)`` to
    # skip guidance at low noise levels. T2V / I2V / V2V leave it unset.
    guidance_interval: tuple[float, float] | None = None

    # V2V conditioning: which latent-frame indices stay locked to the input
    # video. ``None`` resolves to ``[0]`` for I2V (single frame) and ``[0, 1]``
    # for V2V. ``condition_video_keep`` controls whether the first or last
    # source frames are used when the input video is longer than needed.
    condition_frame_indexes: list[int] | None = None
    condition_video_keep: str = "first"

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1280, 720),
            (720, 1280),
            (832, 480),
            (480, 832),
            (1024, 1024),
            (640, 640),
        ]
    )

    # Action modality (requires action_gen=True in the model checkpoint)
    # action_mode: "forward_dynamics" | "policy" | "inverse_dynamics"
    action_mode: str | None = None
    domain_id: int | None = None
    domain_name: str | None = None
    raw_action_dim: int | None = None
    action_fps: float | None = None
    # Action data for forward_dynamics: [T, D] nested list (API) or JSON string
    # (CLI via --action). Ignored by the other action modes.
    action: Any = None
    # Viewpoint phrasing for the structured action caption.
    action_view_point: str = "ego_view"
    # Optional dataset-derived action stats (JSON) for (de)normalization. When
    # set, input actions are normalized and predicted actions de-normalized
    # into physical units with ``action_normalization``.
    action_stats_path: str | None = None
    action_normalization: str = "quantile"

    def _adjust(self, server_args) -> None:
        # adjust distil and edge args — read from the pre-computed config fields
        # so no checkpoint download happens at request time.
        pipeline_config = server_args.pipeline_config
        distilled_sigmas = pipeline_config.distilled_sigmas
        if distilled_sigmas is not None:
            self.num_inference_steps = len(distilled_sigmas)
        self._resolve_variant_defaults(
            bool(pipeline_config.is_edge),
            is_distilled=distilled_sigmas is not None,
        )

        # adjust action args
        action_output = False
        if self.action_mode is not None:
            self.action_mode = str(self.action_mode).strip().lower()
            if self.action_mode not in (
                "policy",
                "forward_dynamics",
                "inverse_dynamics",
            ):
                raise ValueError(
                    f"Unsupported action_mode={self.action_mode!r}; expected "
                    "'policy', 'forward_dynamics', or 'inverse_dynamics'."
                )
            action_output = self.action_mode != "forward_dynamics"

        super()._adjust(server_args)

        # Policy and inverse dynamics produce actions. Forward dynamics consumes
        # actions to produce video and therefore remains a visual request.
        if action_output:
            self.data_type = DataType.ACTION
            self.save_output = False
            self.return_file_paths_only = False
            self.return_frames = False
            self.output_file_name = None
            self.output_compression = 0

    def _guidance_is_explicit(self) -> bool:
        explicit = getattr(self, "_explicit_fields", None)
        return explicit is not None and "guidance_scale" in explicit

    def _resolve_variant_defaults(
        self, is_edge: bool, is_distilled: bool = False
    ) -> None:
        """Fill unset resolution/guidance with the variant's defaults.

        Base resolution defaulting (``supported_resolutions[0]``) covers the
        non-Edge path; only Edge and guidance need explicit handling here.
        """
        is_t2i = self.num_frames == 1
        if is_distilled:
            # Guidance is distilled into the model; run a single forward.
            self.guidance_scale = 1.0
        elif is_edge and not self._guidance_is_explicit():
            self.guidance_scale = (
                COSMOS3_EDGE_T2I_GUIDANCE_SCALE
                if is_t2i
                else COSMOS3_EDGE_T2V_GUIDANCE_SCALE
            )
        if is_t2i and not is_distilled and self.guidance_interval is None:
            self.guidance_interval = COSMOS3_T2I_GUIDANCE_INTERVAL
        if is_edge:
            self.supported_resolutions = COSMOS3_EDGE_SUPPORTED_RESOLUTIONS
            if self.height is None and self.width is None:
                if is_t2i:
                    self.width = self.height = COSMOS3_EDGE_T2I_SIZE
                else:
                    self.width, self.height = (
                        COSMOS3_EDGE_T2V_WIDTH,
                        COSMOS3_EDGE_T2V_HEIGHT,
                    )

    def _set_output_file_name(self) -> None:
        # Action outputs never need a visual filename. This also avoids hashing
        # in-memory observation images while base visual adjustment is running.
        if self.action_mode in ("policy", "inverse_dynamics"):
            return
        # The pipeline config's ``task_type=TI2V`` drives ``data_type`` to
        # VIDEO, but a single-frame request is a T2I and must pick the IMAGE
        # extension. Flip before the base derives the file name.
        if self.num_frames == 1:
            self.data_type = DataType.IMAGE
        super()._set_output_file_name()
