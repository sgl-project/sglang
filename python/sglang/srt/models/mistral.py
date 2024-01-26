"""Inference-only Mistral model."""
from sglang.srt.models.llama2 import LlamaForCausalLM


class MistralForCausalLM(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


EntryClass = MistralForCausalLM
