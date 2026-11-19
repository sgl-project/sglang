from sglang.srt.models.inkling2 import Inkling2ForConditionalGeneration
from sglang.srt.multimodal.inkling2.processor import (
    Inkling2MultimodalProcessor as _Inkling2Processor,
)


class Inkling2MultimodalProcessor(_Inkling2Processor):
    models = [Inkling2ForConditionalGeneration]
