# SPDX-License-Identifier: Apache-2.0
"""LongCat-AudioDiT TTS / voice-cloning pipeline.

Stage layout matches Hunyuan3D shape: BeforeDenoising -> Denoising -> Decoding.
"""

import os

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from sglang.multimodal_gen.configs.models.dits.longcat_audiodit import (
    LongCatAudioDiTConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.dits.longcat_audiodit import (
    LongCatAudioDiTModel,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_audiodit_flow_match import (
    AudioDiTFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longcat_audiodit import (
    LongCatAudioDiTBeforeDenoisingStage,
    LongCatAudioDiTDecodingStage,
    LongCatAudioDiTDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

# Register the in-repo HF config/model so from_pretrained works without
# the external audiodit package (checkpoint model_type is still "audiodit").
AutoConfig.register("audiodit", LongCatAudioDiTConfig, exist_ok=True)
AutoModel.register(LongCatAudioDiTConfig, LongCatAudioDiTModel, exist_ok=True)


class LongCatAudioDiTPipeline(ComposedPipelineBase):
    """SGLang pipeline for LongCat-AudioDiT.

    Loads the full ``LongCatAudioDiTModel`` (transformer + VAE + text encoder)
    via HuggingFace ``from_pretrained``, then runs BeforeDenoising ->
    Denoising -> Decoding.
    """

    pipeline_name = "LongCatAudioDiTPipeline"

    # LongCat-AudioDiT is a monolithic HF PreTrainedModel — no separate
    # Diffusers components.  We populate modules ourselves in load_modules().
    _required_config_modules: list[str] = []

    def load_modules(self, server_args: ServerArgs, loaded_modules=None):
        """Load ``LongCatAudioDiTModel``, its tokenizer, and a scheduler instance."""
        if loaded_modules:
            return loaded_modules

        logger.info("Loading LongCatAudioDiTModel from %s ...", self.model_path)
        model = LongCatAudioDiTModel.from_pretrained(self.model_path)
        self._sync_dit_config_from_model(server_args, model)

        device = get_local_torch_device()
        model = model.to(device)
        dit_dtype = _DTYPE_MAP.get(
            (server_args.pipeline_config.dit_precision or "bf16").lower(),
            torch.bfloat16,
        )
        model.transformer.to(dit_dtype)
        vae_precision = (server_args.pipeline_config.vae_precision or "fp16").lower()
        if vae_precision in ("fp16", "float16"):
            model.vae.to_half()
        model.eval()

        logger.info("Loading tokenizer from %s ...", model.config.text_encoder_model)
        tokenizer = self._load_tokenizer(
            self.model_path, model.config.text_encoder_model
        )

        # Create the custom ascending-t scheduler.
        scheduler = AudioDiTFlowMatchScheduler()

        return {
            "model": model,
            "tokenizer": tokenizer,
            "transformer": model.transformer,
            "vae": model.vae,
            "scheduler": scheduler,
        }

    def create_pipeline_stages(self, server_args: ServerArgs):
        model = self.get_module("model")

        self.add_stage(
            LongCatAudioDiTBeforeDenoisingStage(
                model=model,
                tokenizer=self.get_module("tokenizer"),
                scheduler=self.get_module("scheduler"),
            ),
            "longcat_audiodit_before_denoising_stage",
        )
        self.add_stage(
            LongCatAudioDiTDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
            "longcat_audiodit_denoising_stage",
        )
        self.add_stage(
            LongCatAudioDiTDecodingStage(
                vae=self.get_module("vae"),
                model=model,
            ),
            "longcat_audiodit_decoding_stage",
        )

    @staticmethod
    def _sync_dit_config_from_model(
        server_args: ServerArgs, model: LongCatAudioDiTModel
    ) -> None:
        """Align pipeline dit_config with the loaded HF model (1B / 3.5B, etc.)."""
        arch = server_args.pipeline_config.dit_config.arch_config
        arch.hidden_size = model.config.dit_dim
        arch.num_attention_heads = model.config.dit_heads
        arch.num_channels_latents = model.config.latent_dim
        if arch.num_attention_heads > 0:
            arch.attention_head_dim = arch.hidden_size // arch.num_attention_heads

    @staticmethod
    def _has_local_tokenizer_files(model_path: str) -> bool:
        """True when the checkpoint ships tokenizer artifacts, not just config.json."""
        names = (
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "spiece.model",
            "sentencepiece.bpe.model",
            "vocab.txt",
        )
        return any(os.path.isfile(os.path.join(model_path, name)) for name in names)

    @staticmethod
    def _load_tokenizer(model_path: str, text_encoder_model: str):
        """Prefer tokenizer files shipped with the checkpoint; fall back to Hub.

        AudioDiT checkpoints often only contain ``config.json`` (with a nested
        ``tokenizer_class``). ``AutoTokenizer.from_pretrained(model_path)`` then
        tries to build a T5/UMT5 tokenizer from that config and raises
        ``ValueError`` if sentencepiece is missing — not ``OSError``.
        """
        if LongCatAudioDiTPipeline._has_local_tokenizer_files(model_path):
            try:
                return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            except (OSError, ValueError) as e:
                logger.info(
                    "Local tokenizer in %s failed (%s); loading %s",
                    model_path,
                    e,
                    text_encoder_model,
                )
        else:
            logger.info(
                "No tokenizer files in %s; loading %s",
                model_path,
                text_encoder_model,
            )
        return AutoTokenizer.from_pretrained(text_encoder_model)


EntryClass = [LongCatAudioDiTPipeline]
