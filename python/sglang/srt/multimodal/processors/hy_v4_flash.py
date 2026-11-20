from sglang.srt.models.hunyuan_v4_flash import HYV4FlashForConditionalGeneration
from sglang.srt.multimodal.processors.hy_v4_vl import HYV4VLMultimodalProcessor


class HYV4FlashMultimodalProcessor(HYV4VLMultimodalProcessor):
    models = [HYV4FlashForConditionalGeneration]
    # String-content media markers are qualified only against the VL chat template.
    string_content_media_placeholders = None
