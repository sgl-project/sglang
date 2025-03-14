# Copyright 2025 SGLang Team
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
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from transformers import AutoModel, PreTrainedModel
from transformers.utils.generic import ModelOutput

from sglang.srt.configs import Gemma3Config, ShieldGemma2Config
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.gemma3_mm import Gemma3ForConditionalGeneration
from sglang.srt.utils import add_prefix


@dataclass
class ImageClassifierOutputWithNoAttention(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class ShieldGemma2ImageClassifierOutputWithNoAttention(
    ImageClassifierOutputWithNoAttention
):
    """ShieldGemma2 classifies images as violative or not relative to a specific policy
    Args:
    """

    embeddings: torch.Tensor = None


class ShieldGemma2ForImageClassification(PreTrainedModel):
    config_class = ShieldGemma2Config

    def __init__(
        self,
        config: ShieldGemma2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config
        self.yes_token_index = getattr(config, "yes_token_index", 10_784)
        self.no_token_index = getattr(config, "no_token_index", 3771)
        gemma3_config = Gemma3Config(
            text_config=config.text_config,
            vision_config=config.vision_config,
            mm_tokens_per_image=config.mm_tokens_per_image,
            boi_token_index=config.boi_token_index,
            eoi_token_index=config.eoi_token_index,
            image_token_index=config.image_token_index,
            initializer_range=config.initializer_range,
        )
        self.model = Gemma3ForConditionalGeneration(
            config=gemma3_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.language_model.get_decoder()

    def tie_weights(self):
        return self.model.language_model.tie_weights()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> ShieldGemma2ImageClassifierOutputWithNoAttention:
        """
        Predicts the binary probability that the image violates the specified policy.

        Args:

        Returns:
        """
        assert (
            get_embedding
        ), "ShieldGemma2ForImageClassification is only used for embedding. Please add --is-embedding when you launch the server."

        out: LogitsProcessorOutput = self.model(
            input_ids, positions, forward_batch, input_embeds
        )
        logits = out.next_token_logits
        print(f"logits: {logits}")
        selected_logits = logits[-1, [self.yes_token_index, self.no_token_index]]
        probabilities = torch.softmax(selected_logits, dim=-1)

        return ShieldGemma2ImageClassifierOutputWithNoAttention(
            logits=selected_logits,
            embeddings=probabilities,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        for name, loaded_weight in weights:
            Gemma3ForConditionalGeneration.load_weights(self, [(name, loaded_weight)])


EntryClass = ShieldGemma2ForImageClassification
AutoModel.register(
    config_class=ShieldGemma2Config,
    model_class=ShieldGemma2ForImageClassification,
    exist_ok=True,
)
