# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class ConditionEncodingStage(PipelineStage):
    """Base class for stages that materialize denoiser conditions on the batch."""


class AuxiliaryConditionEncodingStage(ConditionEncodingStage):
    """Apply pipeline-config prepared auxiliary conditioning tensors to the request."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dtype = dtype

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        condition = server_args.pipeline_config.prepare_world_condition(
            batch=batch,
            device=self.device,
            dtype=self.dtype,
        )
        if condition is None:
            return batch
        if not isinstance(condition, Mapping):
            raise TypeError("prepare_world_condition must return a mapping or None")
        for field_name, value in condition.items():
            setattr(batch, field_name, value)
        return batch
