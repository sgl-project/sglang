# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import torch
from transformers import T5TokenizerFast, UMT5EncoderModel

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_va import (
    LingBotVALiberoConfig,
    LingBotVARoboTwinConfig,
)
from sglang.multimodal_gen.configs.sample.lingbot_va import LingBotVASamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
from sglang.multimodal_gen.runtime.models.dits.lingbot_va import LingBotVATransformer
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_va import (
    LingBotVAStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAActionPostprocessStage,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model


class LingBotVALiberoPipeline(ComposedPipelineBase):
    pipeline_name = "LingBotVALiberoPipeline"
    pipeline_config_cls = LingBotVALiberoConfig
    sampling_params_cls = LingBotVASamplingParams
    _required_config_modules = []

    def validate_disagg_role(self, role):
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "LingBot-VA currently runs all policy stages in one worker"
            )

    def load_modules(self, server_args, loaded_modules=None):
        if loaded_modules is not None:
            return loaded_modules
        root = Path(
            maybe_download_model(self.model_path, revision=server_args.revision)
        )
        device = get_local_torch_device()
        model, info = LingBotVATransformer.from_pretrained(
            root / "transformer", torch_dtype=torch.bfloat16, output_loading_info=True
        )
        # The release also contains the original Wan Conv3d; VA inference uses patch_embedding_mlp.
        info["unexpected_keys"] = sorted(
            set(info["unexpected_keys"])
            - {"patch_embedding.bias", "patch_embedding.weight"}
        )
        if any(
            info.get(k)
            for k in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(f"Incompatible LingBot-VA weights: {info}")
        return {
            "transformer": model.eval().to(device),
            "text_encoder": UMT5EncoderModel.from_pretrained(
                root / "text_encoder", dtype=torch.bfloat16
            )
            .eval()
            .to(device),
            "tokenizer": T5TokenizerFast.from_pretrained(root / "tokenizer"),
            "vae": VAELoader(device)
            .load_customized(str(root / "vae"), server_args, "vae")
            .eval()
            .to(device),
        }

    def create_pipeline_stages(self, server_args):
        self.add_stage(
            LingBotVAStage(self.modules, server_args.pipeline_config), "lingbot_va"
        )
        self.add_stage(VLAActionPostprocessStage(), "action_postprocess")


class LingBotVARoboTwinPipeline(LingBotVALiberoPipeline):
    pipeline_name = "LingBotVARoboTwinPipeline"
    pipeline_config_cls = LingBotVARoboTwinConfig


EntryClass = [LingBotVALiberoPipeline, LingBotVARoboTwinPipeline]
