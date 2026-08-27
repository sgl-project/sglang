# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.configs.sensenova_u1 import (
    DEFAULT_CFG_INTERVAL,
    DEFAULT_CFG_NORM,
    DEFAULT_ENABLE_TIMESTEP_SHIFT,
    DEFAULT_T_EPS,
    DEFAULT_THINK_MODE,
    DEFAULT_TIMESTEP_SHIFT,
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _denorm_sensenova_output(x: torch.Tensor) -> torch.Tensor:
    """Convert SenseNova's normalized image tensor from [-1, 1] to [0, 1]."""
    return ((x.float() + 1.0) * 0.5).clamp(0, 1)


@dataclass(frozen=True)
class SenseNovaU1GenerationOptions:
    cfg_norm: str = DEFAULT_CFG_NORM
    timestep_shift: float = DEFAULT_TIMESTEP_SHIFT
    enable_timestep_shift: bool = DEFAULT_ENABLE_TIMESTEP_SHIFT
    cfg_interval: tuple[float, float] = DEFAULT_CFG_INTERVAL
    t_eps: float = DEFAULT_T_EPS
    think_mode: bool = DEFAULT_THINK_MODE

    @classmethod
    def from_batch(cls, batch: Req) -> SenseNovaU1GenerationOptions:
        extra = batch.extra.get(SENSENOVA_U1_REQUEST_EXTRA_KEY, {})
        return cls(
            cfg_norm=extra.get("cfg_norm", DEFAULT_CFG_NORM),
            timestep_shift=float(extra.get("timestep_shift", DEFAULT_TIMESTEP_SHIFT)),
            enable_timestep_shift=bool(
                extra.get("enable_timestep_shift", DEFAULT_ENABLE_TIMESTEP_SHIFT)
            ),
            cfg_interval=tuple(extra.get("cfg_interval", DEFAULT_CFG_INTERVAL)),
            t_eps=float(extra.get("t_eps", DEFAULT_T_EPS)),
            think_mode=bool(extra.get("think_mode", DEFAULT_THINK_MODE)),
        )


class SenseNovaU1GenerationStage(PipelineStage):
    def __init__(self, model: torch.nn.Module, tokenizer: Any):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        del server_args
        options = SenseNovaU1GenerationOptions.from_batch(batch)
        if int(batch.num_outputs_per_prompt) != 1:
            raise ValueError(
                "SenseNova-U1 expects output expansion before generation; "
                f"got num_outputs_per_prompt={batch.num_outputs_per_prompt}."
            )
        seed = batch.seed[0] if isinstance(batch.seed, list) else int(batch.seed)

        out = self.model.t2i_generate(
            self.tokenizer,
            batch.prompt,
            image_size=(int(batch.width), int(batch.height)),
            cfg_scale=float(batch.guidance_scale),
            cfg_norm=options.cfg_norm,
            timestep_shift=options.timestep_shift,
            enable_timestep_shift=options.enable_timestep_shift,
            cfg_interval=options.cfg_interval,
            num_steps=int(batch.num_inference_steps),
            batch_size=1,
            t_eps=options.t_eps,
            think_mode=options.think_mode,
            seed=seed,
        )
        think_text = None
        if options.think_mode:
            images, think_text = out
        else:
            images = out

        images = _denorm_sensenova_output(images)
        samples = [sample.contiguous() for sample in images]
        usage = {"think_text": think_text} if think_text is not None else None
        return OutputBatch(
            output=samples,
            metrics=batch.metrics,
            usage=usage,
        )
