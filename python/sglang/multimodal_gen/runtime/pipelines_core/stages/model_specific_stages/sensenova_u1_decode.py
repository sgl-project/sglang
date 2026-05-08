# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_prepare import (
    SenseNovaU1PixelFlowPrepared,
)


@dataclass(frozen=True, slots=True)
class SenseNovaU1GeneratedSegment:
    type: str
    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    commit_image: Any | None = None


class SenseNovaU1PixelFlowDecoder:
    def forward(
        self,
        prepared: SenseNovaU1PixelFlowPrepared,
        image_prediction: Any,
    ) -> SenseNovaU1GeneratedSegment:
        import numpy as np
        import torch
        from PIL import Image

        array = (
            (image_prediction[0].float() * 0.5 + 0.5)
            .clamp(0, 1)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        image = Image.fromarray((array * 255.0).round().astype(np.uint8), "RGB")
        commit_image = {
            "pixel_values": image_prediction.detach().to(torch.bfloat16).cpu(),
            "value_range": "minus_one_to_one",
            "grid_hw": prepared.gen_grid_hw[:1].detach().cpu(),
        }
        cfg = prepared.cfg
        return SenseNovaU1GeneratedSegment(
            type="image",
            image=image,
            metadata={
                "g_kind": "pixel_flow",
                "native_context_pixel_flow": True,
                "temporary_context_kv": True,
                "timesteps": prepared.steps,
                "seed": prepared.seed,
                "width": prepared.width,
                "height": prepared.height,
                "grid": (prepared.token_h, prepared.token_w),
                "g_position_start": prepared.condition.position_count,
                "condition_position_count": prepared.condition.position_count,
                "cfg_img_condition_position_count": _forward_context_position(
                    prepared.img_condition
                ),
                "cfg_uncondition_position_count": _forward_context_position(
                    prepared.uncondition
                ),
                "noise_scale": prepared.noise_scale,
                "cfg_text_scale": cfg.text_scale,
                "cfg_img_scale": cfg.img_scale,
                "cfg_renorm_type": cfg.renorm_type if cfg.needs_cfg else "none",
            },
            commit_image=commit_image,
        )


def _forward_context_position(context: Any | None) -> int | None:
    if context is None:
        return None
    return context.position_count
