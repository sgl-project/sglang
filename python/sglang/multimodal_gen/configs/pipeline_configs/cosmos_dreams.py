# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams pipeline configuration.

Cosmos-Dreams checkpoints are distilled Cosmos3 Omni checkpoints with a
``cosmos_dreams`` artifact in ``transformer/config.json``. The config reuses
``Cosmos3Config`` (Wan VAE, tokenizer, distilled sigmas) and swaps in the
causal transformer class.
"""

import math
from dataclasses import dataclass

from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    ACTION_CONDITIONING_MODE,
    load_cosmos_dreams_manifest,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import (
    Cosmos3Config,
    _transformer_config,
)

# Deployment default from the reference Cosmos-Dreams deployment: 720x1280.
COSMOS_DREAMS_MAX_PIXELS = 921_600


@dataclass
class CosmosDreamsConfig(Cosmos3Config):
    transformer_class_override: str | None = "CosmosDreamsTransformer"

    # Conditioning contract this pipeline drives; the Transfer subclass overrides it.
    conditioning_mode: str = ACTION_CONDITIONING_MODE

    # The Dreams stages own prompt formatting (training-time JSON caption); the
    # Cosmos3 system prompt and duration suffix were not used in training.
    use_duration_template: bool = False
    use_system_prompt: bool = False

    # Largest canvas admitted per request (height * width).
    max_pixels: int = COSMOS_DREAMS_MAX_PIXELS

    # Resolution tier whose trained canvases are used when a request leaves
    # height/width unset. The Sim-Bimanual checkpoint trained on the 480 tier only.
    canvas_tier: str = "480"

    def update_config_from_dict(self, args, prefix: str = "") -> None:
        super().update_config_from_dict(args, prefix)
        if self.model_path:
            self._validate_checkpoint(self.model_path)

    def _validate_checkpoint(self, model_path: str) -> None:
        """Fail in the launcher before 30 GB of weights are loaded."""
        manifest = load_cosmos_dreams_manifest(_transformer_config(model_path))
        if manifest.conditioning_mode != self.conditioning_mode:
            raise ValueError(
                f"Checkpoint {manifest.checkpoint_id} is conditioned on "
                f"{manifest.conditioning_mode!r} but {type(self).__name__} drives "
                f"{self.conditioning_mode!r}; use CosmosDreamsPipeline for action "
                "checkpoints and CosmosDreamsTransferPipeline for control-video checkpoints."
            )
        if self.distilled_sigmas is None:
            raise ValueError(
                "Cosmos-Dreams requires a distilled fixed-step scheduler "
                "(FlowMatchEulerDiscreteScheduler with fixed_step_sampler_config.t_list)."
            )
        if len(self.distilled_sigmas) != len(manifest.t_list) or any(
            not math.isclose(scheduler_sigma, manifest_sigma, rel_tol=0.0, abs_tol=1e-8)
            for scheduler_sigma, manifest_sigma in zip(
                self.distilled_sigmas, manifest.t_list, strict=True
            )
        ):
            raise ValueError(
                "Cosmos-Dreams scheduler and transformer manifests define different "
                f"fixed-step schedules: scheduler={self.distilled_sigmas}, "
                f"transformer={list(manifest.t_list)}."
            )

    def supports_action_endpoint(self) -> bool:
        # Forward dynamics only: actions condition the video, none are produced.
        return False

    def supports_dynamic_batching(self) -> bool:
        return False
