from typing import Optional

from transformers import AutoProcessor, Qwen2_5_VLProcessor
from transformers.image_processing_utils import BaseImageProcessor
from transformers.models.qwen2 import Qwen2Config

from sglang.srt.configs.dots_vlm import DotsVisionConfig


class DotsOCRConfig(Qwen2Config):
    model_type = "dots_ocr"

    def __init__(
        self,
        image_token_id=151665,
        video_token_id=151656,
        vision_config: Optional[dict] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_config = DotsVisionConfig(**(vision_config or {}))

    def save_pretrained(self, save_directory, **kwargs):
        self._auto_class = None
        super().save_pretrained(save_directory, **kwargs)


class DummyVideoProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __call__(self, *args, **kwargs):
        return None


class DotsVLProcessor(Qwen2_5_VLProcessor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs
    ):
        if video_processor is None:
            video_processor = DummyVideoProcessor()
        super().__init__(
            image_processor, tokenizer, video_processor, chat_template=chat_template
        )
        self.image_token = (
            "<|imgpad|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None) is not None
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )


AutoProcessor.register(DotsOCRConfig, DotsVLProcessor)
