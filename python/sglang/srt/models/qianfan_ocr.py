# Adapted from https://huggingface.co/baidu/Qianfan-OCR

from typing import Optional

from transformers import PretrainedConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.internvl import InternVLChatModel


def normalize_qianfan_ocr_config(config: PretrainedConfig) -> None:
    """Spell ``QianfanOCRConfig`` the way the shared InternVL path reads it.

    Qianfan-OCR ships InternVL2.5 weights -- an InternViT tower, the ``mlp1``
    pixel-shuffle projector and a Qwen3 backbone -- so only the transformers
    config field spellings differ. Runs in both the scheduler (model) and the
    tokenizer (processor) process, since each reads ``hf_config`` on its own.
    """
    if not hasattr(config, "llm_config"):
        config.llm_config = config.text_config

    vision_config = config.vision_config
    # QianfanOCRVisionConfig stores these as 2-tuples; InternViT does integer
    # floor-division on them when sizing the patch grid.
    for name in ("image_size", "patch_size"):
        value = getattr(vision_config, name)
        if isinstance(value, (list, tuple)):
            setattr(vision_config, name, value[0])

    # InternViT reads both unconditionally, and QianfanOCRVisionConfig omits
    # them. Inference-safe defaults: dropout is a no-op in eval mode, and
    # initializer_factor only seeds ls1/ls2, which the checkpoint overwrites.
    if not hasattr(vision_config, "dropout"):
        vision_config.dropout = 0.0
    if not hasattr(vision_config, "initializer_factor"):
        vision_config.initializer_factor = 1.0


class QianfanOCRForConditionalGeneration(InternVLChatModel):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        use_flash_attn=True,
    ) -> None:
        normalize_qianfan_ocr_config(config)
        super().__init__(
            config=config,
            quant_config=quant_config,
            use_flash_attn=use_flash_attn,
        )


EntryClass = QianfanOCRForConditionalGeneration
