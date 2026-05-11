# Models
from sglang.srt.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration


class InternS2PreviewForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    """InternS2Preview Vision-Language Model."""


EntryClass = [InternS2PreviewForConditionalGeneration]
