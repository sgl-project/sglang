# SPDX-License-Identifier: Apache-2.0
"""
Shape conditioning stage for Hunyuan3D pipelines.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _move_to_device(payload, device, dtype):
    if isinstance(payload, torch.Tensor):
        return payload.to(device=device, dtype=dtype)
    if isinstance(payload, dict):
        return {k: _move_to_device(v, device, dtype) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_move_to_device(v, device, dtype) for v in payload]
    return payload


class ShapeConditioningStage(PipelineStage):
    def __init__(self, conditioner: Any, model: Any) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.model = model

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        image = batch.extra["shape_image"]
        cond_inputs = batch.extra["shape_cond_inputs"]
        device = self.device
        dtype = next(self.model.parameters()).dtype

        image = _move_to_device(image, device, dtype)
        cond_inputs = _move_to_device(cond_inputs, device, dtype)

        do_cfg = batch.guidance_scale >= 0 and not (
            hasattr(self.model, "guidance_embed") and self.model.guidance_embed is True
        )

        cond = self.conditioner(image=image, **cond_inputs)
        if do_cfg:
            un_cond = self.conditioner.unconditional_embedding(
                image.shape[0], **cond_inputs
            )

            def cat_recursive(a, b):
                if isinstance(a, torch.Tensor):
                    return torch.cat([a, b], dim=0).to(dtype)
                out = {}
                for key in a.keys():
                    out[key] = cat_recursive(a[key], b[key])
                return out

            cond = cat_recursive(cond, un_cond)

        batch.extra["shape_cond"] = cond
        batch.extra["shape_do_cfg"] = do_cfg
        batch.extra["shape_image"] = image
        return batch
