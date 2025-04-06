# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only Mistral model."""

from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

from sglang.srt.models.llama import LlamaForCausalLM


class MistralForCausalLM(LlamaForCausalLM):
    pass


class Mistral3ForConditionalGeneration:
    MULTIMODAL_PROJECTOR_TYPE = Mistral3MultiModalProjector

    def __init__(self, **kwargs):
        # lazy load inner class
        # to bypass circular import
        from sglang.srt.models.llava import LlavaForConditionalGeneration

        self.inner = LlavaForConditionalGeneration(**kwargs)
        self.inner.multi_modal_projector = self.MULTIMODAL_PROJECTOR_TYPE(
            kwargs["config"]
        )

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __hasattr__(self, name):
        return hasattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


EntryClass = [MistralForCausalLM, Mistral3ForConditionalGeneration]
