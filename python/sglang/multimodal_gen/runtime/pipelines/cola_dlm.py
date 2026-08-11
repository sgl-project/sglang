"""Cola-DLM text diffusion pipeline.

Cola-DLM is a continuous latent diffusion language model that generates text
by performing block-wise diffusion in a continuous latent space. It uses two
sub-models: a DiT prior (ColaDiTModel) and a Text VAE (ColaTextVAEModel).

This pipeline overrides load_modules() to load the Cola-DLM sub-models
directly from nested checkpoint directories, bypassing model_index.json.

Reference: https://github.com/ByteDance-Seed/Cola-DLM/blob/main/cola_dlm/inference.py
"""

from __future__ import annotations

import logging
import os
from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cola_dlm import (
    ColaBlockDenoisingStage,
    ColaTextDecodingStage,
    ColaTokenizationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = logging.getLogger(__name__)


class ColaDLMPipeline(ComposedPipelineBase):
    """Cola-DLM text diffusion pipeline.

    Pipeline stages:
    1. ColaTokenizationStage — tokenize prompt, VAE encode, prepare first block
    2. ColaBlockDenoisingStage — block-wise ODE with CFG, VAE decode, sample tokens
    3. ColaTextDecodingStage — detokenize generated tokens to text
    """

    pipeline_name = "ColaDLMPipeline"
    _required_config_modules = []  # We load everything in load_modules()

    def _load_config(self) -> dict[str, Any]:
        """Return a synthetic config dict (no model_index.json needed)."""
        return {
            "_class_name": self.pipeline_name,
            "_diffusers_version": "0.0.0",
        }

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load Cola-DLM sub-models directly from nested checkpoint directories."""
        import torch

        from sglang.multimodal_gen.runtime.models.dits.cola_dlm import ColaDiTWrapper
        from sglang.multimodal_gen.runtime.models.vaes.cola_dlm import (
            ColaTextVAEWrapper,
        )

        config = server_args.pipeline_config
        model_path = server_args.model_path

        # Resolve sub-model paths
        dit_path = os.path.join(model_path, config.dit_path)
        vae_path = os.path.join(model_path, config.vae_path)

        # Load tokenizer
        tokenizer_path = os.path.join(model_path, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            from tokenizers import Tokenizer

            logger.info("Loading tokenizer from %s", tokenizer_path)
            tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            from transformers import AutoTokenizer

            logger.info("Loading tokenizer from %s (AutoTokenizer)", model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

        # Load DiT model and wrap
        from cola_dlm.modeling_cola_dit import ColaDiTModel

        logger.info("Loading Cola-DLM DiT from %s", dit_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dit_raw = ColaDiTModel.from_pretrained(dit_path).to(device).eval()

        dit = ColaDiTWrapper(config.dit_config)
        dit.load_model(dit_raw)

        # Load VAE model and wrap
        from cola_dlm.modeling_cola_vae import ColaTextVAEModel

        logger.info("Loading Cola-DLM VAE from %s", vae_path)
        vae_raw = ColaTextVAEModel.from_pretrained(vae_path).to(device).eval()

        vae = ColaTextVAEWrapper(config.vae_config)
        vae.load_model(vae_raw)

        logger.info("All Cola-DLM components loaded successfully")

        return {
            "dit": dit,
            "vae": vae,
            "tokenizer": tokenizer,
        }

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Create the three pipeline stages."""
        dit = self.get_module("dit")
        vae = self.get_module("vae")
        tokenizer = self.get_module("tokenizer")
        config = server_args.pipeline_config

        # Stage 1: Tokenization + VAE Encode
        self.add_stage(
            stage_name="cola_tokenization",
            stage=ColaTokenizationStage(
                vae=vae,
                tokenizer=tokenizer,
                pipeline_config=config,
            ),
        )

        # Stage 2: Block-wise Denoising
        self.add_stage(
            stage_name="cola_denoising",
            stage=ColaBlockDenoisingStage(
                dit=dit,
                vae=vae,
                pipeline_config=config,
            ),
        )

        # Stage 3: Text Decoding
        self.add_stage(
            stage_name="cola_text_decoding",
            stage=ColaTextDecodingStage(
                tokenizer=tokenizer,
                pipeline_config=config,
            ),
        )


EntryClass = [ColaDLMPipeline]
