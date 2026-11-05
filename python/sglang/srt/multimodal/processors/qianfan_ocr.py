from sglang.srt.models.qianfan_ocr import (
    QianfanOCRForConditionalGeneration,
    normalize_qianfan_ocr_config,
)
from sglang.srt.multimodal.processors.internvl import InternVLProcessor


class QianfanOCRProcessor(InternVLProcessor):
    """Qianfan-OCR uses InternVL2.5 dynamic tiling and the same image tokens
    (``<img>`` / ``<IMG_CONTEXT>`` / ``</img>``), so only the config needs
    normalizing before the InternVL processor reads it."""

    models = [QianfanOCRForConditionalGeneration]

    def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
        normalize_qianfan_ocr_config(hf_config)
        super().__init__(hf_config, server_args, _image_processor, *args, **kwargs)
