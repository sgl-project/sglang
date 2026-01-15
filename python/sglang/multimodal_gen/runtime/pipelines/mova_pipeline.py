# SPDX-License-Identifier: Apache-2.0
"""
MoVA pipeline integration (native SGLang pipeline).
"""

from __future__ import annotations

import os
import sys

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.mova import MovaPipelineConfig
from sglang.multimodal_gen.configs.sample.mova import MovaSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    MovaInferenceStage,
    MovaPreprocessStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class MovaPipeline(ComposedPipelineBase):
    """MoVA pipeline with SGLang stage orchestration."""

    pipeline_name = "MoVA"
    is_video_pipeline = True
    _required_config_modules: list[str] = []
    pipeline_config_cls = MovaPipelineConfig
    sampling_params_cls = MovaSamplingParams

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, object]:
        if loaded_modules is not None:
            return loaded_modules

        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../..")
        )
        mossvg_root = os.path.join(repo_root, "mossVG")
        # Prefer the top-level "mova" package to avoid double-importing
        # the same module as both "mossVG.mova" and "mova".
        if mossvg_root not in sys.path:
            sys.path.append(mossvg_root)

        from mova.diffusion.pipelines.mova import MoVA

        torch_dtype = PRECISION_TO_TYPE.get(
            server_args.pipeline_config.dit_precision, torch.bfloat16
        )
        # diffusers device_map only accepts "cuda"/"balanced"; use .to() for rank.
        pipe = MoVA.from_pretrained(
            self.model_path, device_map="cuda", torch_dtype=torch_dtype
        )
        pipe.to(f"cuda:{envs.LOCAL_RANK}")

        # MoVA's video_vae can be bf16 while latents are float32; align dtype at decode.
        try:
            vae = pipe.video_vae
            target_dtype = next(vae.parameters()).dtype
            orig_decode = vae.decode

            def _decode_with_cast(z, *args, **kwargs):
                if z.dtype != target_dtype:
                    z = z.to(target_dtype)
                return orig_decode(z, *args, **kwargs)

            vae.decode = _decode_with_cast  # type: ignore[assignment]
        except Exception as e:
            logger.warning("Failed to wrap MoVA video_vae decode for dtype cast: %s", e)
        pipe.eval()
        return {"mova": pipe}

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )
        self.add_stage(stage_name="mova_preprocess_stage", stage=MovaPreprocessStage())
        self.add_stage(
            stage_name="mova_inference_stage",
            stage=MovaInferenceStage(self.get_module("mova")),
        )


class MoVAPipelineAlias(MovaPipeline):
    pipeline_name = "MoVAPipeline"


EntryClass = [MovaPipeline, MoVAPipelineAlias]
