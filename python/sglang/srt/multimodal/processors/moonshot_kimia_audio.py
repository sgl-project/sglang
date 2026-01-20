from sglang.srt.models.moonshot_kimia import MoonshotKimiaForCausalLM
from sglang.srt.multimodal.processors.qwen_audio import (
    Qwen2AudioMultimodalProcessor,
)

__all__ = ["MoonshotKimiaAudioMultimodalProcessor"]


class MoonshotKimiaAudioMultimodalProcessor(Qwen2AudioMultimodalProcessor):
    models = [MoonshotKimiaForCausalLM]
