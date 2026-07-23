"""Multimodal processor registration for MinerU external architecture."""

import logging

from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

logger = logging.getLogger(__name__)


class MinerUDiffusionForConditionalGeneration:
    """Name-only placeholder used for processor registration."""


class MinerUDiffusionProcessor(QwenVLImageProcessor):
    """Reuse Qwen-VL image processor for MinerU's Qwen2-VL-based visual inputs."""

    models = [MinerUDiffusionForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        if not server_args.disable_fast_image_processor:
            raise ValueError(
                "MinerU external processor requires --disable-fast-image-processor. "
                "Please set it in launch args to avoid tokenizer kwargs incompatibility."
            )

        if self.model_type == "mineru_diffusion":
            logger.info(
                "Map MinerU processor model_type from `%s` to `qwen2_vl` for "
                "mRoPE index compatibility.",
                self.model_type,
            )
            self.model_type = "qwen2_vl"
