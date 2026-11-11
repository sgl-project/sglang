"""Hugging Face-level processor for Jina-OCR-v1.

Jina's processor_config.json only carries its own remote-code fields (base_size,
crop_mode, image_size), while DeepseekOCRProcessor requires the three geometry
fields defaulted below. They are DeepSeek-OCR's values, which Jina's own processing
code also uses (1024 base view, 16 px patches, 4x downsample).
"""

from typing import Tuple

from transformers import LlamaTokenizerFast, PretrainedConfig

from sglang.srt.configs.deepseek_ocr import DeepseekOCRProcessor
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)


class JinaOCRHFProcessor(DeepseekOCRProcessor):
    def __init__(
        self,
        tokenizer: LlamaTokenizerFast,
        candidate_resolutions: Tuple[Tuple[int, int]] = ((1024, 1024),),
        patch_size: int = 16,
        downsample_ratio: int = 4,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            candidate_resolutions=candidate_resolutions,
            patch_size=patch_size,
            downsample_ratio=downsample_ratio,
            **kwargs,
        )


@register_customized_processor(processor_class=JinaOCRHFProcessor)
class JinaOCRConfig(PretrainedConfig):
    model_type = "jina-ocr"


def is_jina_ocr_config(config) -> bool:
    return hasattr(config, "mtp_num_heads")
