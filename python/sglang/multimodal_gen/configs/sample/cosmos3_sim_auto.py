# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Nano-Sim-Auto sampling parameters.

The request shape is the Multiview-AV one (``multiview.views[]`` with a WSM
``control_path``, an optional ``vision_path`` and a ``prompt`` per camera,
``wsm``, optional ``lidar``). The student fixes what the bidirectional model
exposes as knobs: the four-step SDE schedule, guidance 1, no flow shift, one
conditioning frame per camera and one measured LiDAR sweep; requests that set
them differently are rejected rather than silently ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sglang.multimodal_gen.configs.models.dits.cosmos3_sim_auto import (
    Cosmos3SimAutoManifest,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_sim_auto import (
    Cosmos3SimAutoConfig,
)
from sglang.multimodal_gen.configs.sample.cosmos3_multiview import (
    Cosmos3MultiviewSamplingParams,
)

# Reference rollout horizon (297 frames = 1 + 4 * 74 at 30 fps, 9.9 s).
COSMOS3_SIM_AUTO_DEFAULT_NUM_FRAMES = 297
COSMOS3_SIM_AUTO_DEFAULT_FPS = 30


@dataclass
class Cosmos3SimAutoSamplingParams(Cosmos3MultiviewSamplingParams):
    num_frames: int = COSMOS3_SIM_AUTO_DEFAULT_NUM_FRAMES
    fps: int = COSMOS3_SIM_AUTO_DEFAULT_FPS
    # Guidance is distilled into the weights; the schedule length is refreshed
    # from the artifact on adjustment.
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    # The student conditions each camera on its first frame only, like the
    # reference rollout.
    condition_video_as_image: bool | None = True

    def _validate(self) -> None:
        super()._validate()
        if (
            self.per_view_negative_prompt is not None
            and self.per_view_negative_prompt.strip()
        ):
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto runs no unconditional branch; per_view_negative_prompt "
                "is not supported."
            )
        if self.normalize_cfg:
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto runs guidance 1; normalize_cfg must stay off."
            )
        lidar = self.resolved_lidar()
        if lidar is not None and lidar["num_conditional_sweeps"] not in (None, 1):
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto conditions on exactly one measured LiDAR sweep, got "
                f"num_conditional_sweeps={lidar['num_conditional_sweeps']}."
            )

    def _adjust(self, server_args) -> None:
        pipeline_config = server_args.pipeline_config
        if not isinstance(pipeline_config, Cosmos3SimAutoConfig):
            raise TypeError(
                "Cosmos3SimAutoSamplingParams requires Cosmos3SimAutoConfig, got "
                f"{type(pipeline_config).__name__}."
            )
        manifest = pipeline_config.sim_auto_manifest
        if manifest is None:
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto pipeline config has not resolved its artifact; "
                "model_path must be set before requests are adjusted."
            )
        super()._adjust(server_args)
        self._enforce_student_contract(manifest)

    def _enforce_student_contract(self, manifest: Cosmos3SimAutoManifest) -> None:
        """Reject the knobs the distilled student has no freedom in, after the defaults were applied."""
        if float(self.guidance_scale) != 1.0:
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto is a distilled student and runs guidance_scale=1.0 only, "
                f"got {self.guidance_scale}."
            )
        if int(self.num_inference_steps) != len(manifest.t_list):
            raise ValueError(
                f"Cosmos3-Nano-Sim-Auto uses its fixed {len(manifest.t_list)}-step schedule; "
                f"got num_inference_steps={self.num_inference_steps}."
            )
        if self.flow_shift is not None and float(self.flow_shift) != 1.0:
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto's fixed-step schedule takes no flow_shift, "
                f"got {self.flow_shift}."
            )
        if self.lidar is not None and not math.isclose(
            float(self.fps), manifest.default_fps
        ):
            raise ValueError(
                f"Joint camera/LiDAR requests need fps={manifest.default_fps:g} to match the "
                f"LiDAR source clock, got {self.fps:g}."
            )
        if not self.resolved_condition_video_as_image():
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto conditions each camera on its first frame only; "
                "condition_video_as_image must stay true."
            )
        indexes = self.resolved_local_condition_indexes()
        if indexes not in (None, [], [0]):
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto conditions on latent frame 0 of each camera only, got "
                f"condition_frame_indexes_vision={indexes}."
            )
