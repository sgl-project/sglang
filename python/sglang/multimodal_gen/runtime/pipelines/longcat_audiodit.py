"""
LongCat-AudioDiT TTS / voice-cloning pipeline.

Loads the model as a HuggingFace PreTrainedModel and runs the full generation
(text encoding -> ODE solve -> VAE decode) in a single monolithic stage.
"""

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from sglang.multimodal_gen.configs.models.dits.longcat_audiodit import (
    LongCatAudioDiTConfig,
)
from sglang.multimodal_gen.runtime.models.dits.longcat_audiodit import (
    LongCatAudioDiTModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longcat_audiodit import (
    LongCatAudioDiTInferenceStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Register LongCatAudioDiTConfig / AudioDiTModel with HuggingFace AutoConfig / AutoModel
# so that from_pretrained works without the external audiodit package.
AutoConfig.register("audiodit", LongCatAudioDiTConfig, exist_ok=True)
AutoModel.register(LongCatAudioDiTConfig, LongCatAudioDiTModel, exist_ok=True)


class LongCatAudioDiTPipeline(ComposedPipelineBase):
    """SGLang pipeline for LongCat-AudioDiT.

    Loads the full ``AudioDiTModel`` (transformer + VAE + text encoder) directly
    via HuggingFace ``from_pretrained`` and runs inference through a single
    monolithic stage.  The model code is inlined in sglang — no external
    ``audiodit`` package is needed.
    """

    pipeline_name = "LongCatAudioDiTPipeline"

    # LongCat-AudioDiT is a monolithic HF PreTrainedModel — there are no
    # separate Diffusers components.  We populate modules ourselves in
    # load_modules() below.
    _required_config_modules: list[str] = []

    def load_modules(self, server_args: ServerArgs, loaded_modules=None):
        """Load ``AudioDiTModel`` and its tokenizer directly.

        Bypasses the standard Diffusers component-loading path because
        LongCat-AudioDiT ships as a single HuggingFace ``PreTrainedModel``
        without a ``model_index.json``.

        The model code is inlined under
        ``sglang.multimodal_gen.runtime.models.longcat_audiodit``, so no
        external ``audiodit`` package is required.
        """
        if loaded_modules:
            return loaded_modules

        logger.info("Loading AudioDiTModel from %s ...", self.model_path)
        model = LongCatAudioDiTModel.from_pretrained(self.model_path)

        # Move to GPU, apply precision per PipelineConfig:
        #   - transformer: bfloat16 (dit_precision = "bf16")
        #   - VAE: float16 (vae_precision = "fp16", matching reference inference.py)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.transformer.to(torch.bfloat16)
        model.vae.to_half()
        model.eval()

        logger.info("Loading tokenizer from %s ...", model.config.text_encoder_model)
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

        return {"model": model, "tokenizer": tokenizer}

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            LongCatAudioDiTInferenceStage(
                model=self.get_module("model"),
                tokenizer=self.get_module("tokenizer"),
            ),
            "longcat_audiodit_inference_stage",
        )


EntryClass = [LongCatAudioDiTPipeline]
