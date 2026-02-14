from sglang.srt.models.qwen2_vl import (
    Qwen2VLForConditionalGeneration as OriginalQwen2VLForConditionalGeneration,
)
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor


class Qwen2VLForConditionalGeneration(OriginalQwen2VLForConditionalGeneration):
    def __init__(self, config, quant_config, prefix: str = "") -> None:
        super().__init__(config, quant_config, prefix)
        print("init custom model:", self.__class__.__name__)


class CustomProcessor(QwenVLImageProcessor):
    models = [Qwen2VLForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        print("init custom processor:", self.__class__.__name__)


EntryClass = Qwen2VLForConditionalGeneration
