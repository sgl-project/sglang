import re
from typing import List, Union

from sglang.srt.models.eagle2_5_vl import Eagle2_5_VLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.utils import logger


class Eagle2_5_VLProcessor(BaseMultimodalProcessor):
    models = [Eagle2_5_VLForConditionalGeneration]

    IMG_START = "<img>"
    IMG_END = "</img>"
    IMG_PLACEHOLDER = "image"
    VIDEO_PLACEHOLDER = "video"
    IMG_CONTEXT = "<IMG_CONTEXT>"

    # HF chat_template placeholders
    HF_IMAGE_PLACEHOLDER_RE = re.compile(r"<image-\d+>")
    HF_VIDEO_PLACEHOLDER_RE = re.compile(r"<video-\d+>")

    # add_vision_id might add "<image 1>" (not actual placeholder)
    HF_IMAGE_VISION_ID_RE = re.compile(r"<image\s+\d+>")
    HF_VIDEO_VISION_ID_RE = re.compile(r"<video\s+\d+>")

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.config = hf_config
        self.hf_config = hf_config

        tokenizer = (
            self._processor.tokenizer
            if hasattr(self._processor, "tokenizer")
            else self._processor
        )
        self.tokenizer = tokenizer

        img_ctx_id = tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT)
        if img_ctx_id is None or img_ctx_id < 0:
            raise RuntimeError("Token <IMG_CONTEXT> not found in tokenizer vocab.")

        self.image_token_id = img_ctx_id
        self.video_token_id = img_ctx_id

        self.img_start_token_id = tokenizer.convert_tokens_to_ids(self.IMG_START)
        self.img_end_token_id = tokenizer.convert_tokens_to_ids(self.IMG_END)

        # canonical placeholder text for SGLang
        self._img_placeholder_text = (
            f"{self.IMG_START}{self.IMG_PLACEHOLDER}{self.IMG_END}"
        )
        self._video_placeholder_text = (
            f"{self.IMG_START}{self.VIDEO_PLACEHOLDER}{self.IMG_END}"
        )

        # IMPORTANT: combined regex should match both formats,
        # but we will REWRITE HF format into canonical format before load_mm_data
        self._img_placeholder_re = re.compile(r"(?:<img>\s*image\s*</img>|<image-\d+>)")
        self._video_placeholder_re = re.compile(
            r"(?:<img>\s*video\s*</img>|<video-\d+>)"
        )

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self._img_placeholder_text,
            image_token_id=self.image_token_id,
            image_token_regex=self._img_placeholder_re,
            video_token=self._video_placeholder_text,
            video_token_id=self.video_token_id,
            video_token_regex=self._video_placeholder_re,
        ).build(_processor)

    def _rewrite_mm_placeholders(self, input_text: str) -> str:
        if not isinstance(input_text, str):
            return input_text

        # remove vision-id tags if present
        input_text = self.HF_IMAGE_VISION_ID_RE.sub("", input_text)
        input_text = self.HF_VIDEO_VISION_ID_RE.sub("", input_text)

        # rewrite <image-1> -> <img>image</img>
        input_text = self.HF_IMAGE_PLACEHOLDER_RE.sub(
            self._img_placeholder_text, input_text
        )
        input_text = self.HF_VIDEO_PLACEHOLDER_RE.sub(
            self._video_placeholder_text, input_text
        )

        return input_text

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):

        input_text = self._rewrite_mm_placeholders(input_text)

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            audio_data=request_obj.audio_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        # ret should NOT be None if raw images were processed
        if ret is None:
            logger.warning(
                "Processor ret is None. This usually means raw images were not processed by HF processor. "
                "Check that image_data is non-empty and load_mm_data aligned tokens with data."
            )

        if (
            (image_data or request_obj.video_data)
            and mm_items
            and all((not it.offsets) for it in mm_items)
        ):
            keys = list(ret.keys()) if isinstance(ret, dict) else None
            logger.warning(
                "Multimodal inputs exist but all mm_items.offsets are empty. "
                "Usually means processor didn't insert <IMG_CONTEXT> into input_ids. "
                "processor_ret_keys=%s, image_token_id=%s",
                keys,
                self.image_token_id,
            )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_start_id": self.img_start_token_id,
            "im_end_id": self.img_end_token_id,
            "im_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
        }
