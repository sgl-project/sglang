"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from functools import partial
from typing import Iterable, Optional, Tuple, Union

import torch

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from sglang.srt.utils import add_prefix


class FusedResidualIdentity(torch.nn.Identity):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return (hidden_states, None)
        else:
            return (hidden_states + residual, None)


class Qwen3ForCausalLMEagle(Qwen3ForCausalLM):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):

        super().__init__(config, quant_config, prefix)

        # EAGLE-2 overwrites draft model embedding and lm_head from target layers.
        # See the calling of `set_embed_and_head` in EAGLE worker and SGL Qwen3ForCausalLM.
        # Under tie_word_embeddings, this could be deleted twice by the SGLang, so we make a placeholder here.
        if getattr(self.config, "tie_word_embeddings", True):
            assert id(self.lm_head.weight) == id(self.model.embed_tokens.weight)
            self.lm_head = torch.nn.Linear(1, 1)

        # Skip the first layer's input_layernorm:
        # https://github.com/SafeAILab/EAGLE/blob/35c78f6cdc19a73e05cf5c330b4c358dad970c6a/eagle/model/cnets.py#L427
        if getattr(self.config, "skip_first_input_layernorm", True):
            self.model.layers[0].input_layernorm = torch.nn.Identity()
            self.model.layers[0].layer_communicator.input_layernorm = (
                torch.nn.Identity()
            )

        # Eagle-2 does not normalize model output by default
        if getattr(self.config, "skip_output_norm", True):
            self.model.norm = FusedResidualIdentity()

        # Add the fully connected layer:
        # https://github.com/SafeAILab/EAGLE/blob/35c78f6cdc19a73e05cf5c330b4c358dad970c6a/eagle/model/cnets.py#L511
        self.model.eagle_fc = torch.nn.Linear(
            2 * self.config.hidden_size, self.config.hidden_size
        )

        # wrap Qwen3 forward to be pass through the eagle_fc layer:
        self.model.forward = partial(
            self.wrap_forward, self.model, origin_forward=self.model.forward
        )

    @staticmethod
    def wrap_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs
    ) -> Union[torch.Tensor, PPProxyTensors]:

        origin_forward = kwargs.pop("origin_forward", None)
        assert origin_forward is not None

        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds

            prev_states = forward_batch.spec_info.hidden_states
            hidden_states = self.eagle_fc(
                torch.cat((hidden_states, prev_states), dim=-1)
            )
        else:
            assert "pp_proxy_tensors" in kwargs

        return origin_forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=hidden_states,
            **kwargs
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = [(add_prefix(k, "model"), v) for k, v in weights]
        print([w[0] for w in weights])
        print([k for k, v in self.named_parameters()])
        super().load_weights(weights)


EntryClass = [Qwen3ForCausalLMEagle]
