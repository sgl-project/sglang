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

# Adapted from
# https://github.com/vllm-project/vllm/blob/a1a2aaadb9122f05667140e39cf67e5736c8b6d6/vllm/model_executor/models/transformers.py
"""Wrapper around `transformers` models"""
import logging
import re
from typing import Iterable, Literal, Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sglang.srt.distributed import divide, get_tensor_model_parallel_world_size
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)


def maybe_prefix(prefix: str, name: str) -> str:
    """Add a prefix to a name if the prefix is non-empty.

    Args:
        prefix: The prefix to add. If empty, no prefix will be added.
        name: The name to potentially prefix.

    Returns:
        The string "prefix.name" if prefix was non-empty, otherwise just "name".
    """
    return name if not prefix else f"{prefix}.{name}"


def sglang_flash_attention_forward(
    # Transformers args
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    # sglang kwargs
    forward_batch: ForwardBatch,
    # Transformers kwargs
    scaling: float = None,
    attention_instances: list[RadixAttention] = None,
    **kwargs,
):
    self_attn: RadixAttention = attention_instances[module.layer_idx]
    if scaling is not None:
        self_attn.scaling = float(scaling)
    hidden = query.shape[-2]
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    query, key, value = (x.reshape(hidden, -1) for x in (query, key, value))
    return self_attn.forward(query, key, value, forward_batch=forward_batch), None


ALL_ATTENTION_FUNCTIONS["sglang"] = sglang_flash_attention_forward


class HFColumnParallelLinear(ColumnParallelLinear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)[0]


class HFRowParallelLinear(RowParallelLinear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)[0]


def replace_linear_class(
    linear: nn.Linear,
    style: Literal["colwise", "rowwise"],
    quant_config: QuantizationConfig,
) -> Union[ColumnParallelLinear, RowParallelLinear]:
    """
    Replace nn.Linear with one of vLLM's tensor parallel linear classes.

    Args:
        linear (nn.Linear): `nn.Linear` to be replaced.
        style (str): Tensor parallel style of the new linear, e.g. "colwise".
        quant_config (QuantConfig): Quantization config for the new linear.
    Returns:
        Union[ColumnParallelLinear, RowParallelLinear]: The new linear.
    """

    if not isinstance(style, str):
        raise ValueError(f"Unsupported parallel style type {type(style)}, expected str")

    sglang_linear_cls = {
        "colwise": ColumnParallelLinear,
        "rowwise": RowParallelLinear,
    }.get(style, ReplicatedLinear)

    class HFCompatibleLinear(sglang_linear_cls):
        """
        Wrapper class that removes `output_bias` from returned output.
        """

        @property
        def parent_cls(self) -> type:
            return sglang_linear_cls

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return super().forward(input)[0]

    return HFCompatibleLinear(
        input_size=linear.in_features,
        output_size=linear.out_features,
        bias=linear.bias is not None,
        quant_config=quant_config,
    )


class TransformersForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        logger.info("Using Transformers backend.")

        self.quant_config = quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size

        # model is loaded under set_default_torch_dtype(model_config.dtype)
        self.model: PreTrainedModel = AutoModel.from_config(
            self.config,
            torch_dtype=torch.get_default_dtype(),
            attn_implementation="sglang",
            trust_remote_code=True,
        )

        # Attention modifications (assumes 1 attention op per hidden layer)
        tp_size = get_tensor_model_parallel_world_size()

        # MLP modifications
        self.tensor_parallel(tp_size)

        head_dim = (
            (config.hidden_size // config.num_attention_heads)
            if not hasattr(config, "head_dim")
            else config.head_dim
        )
        self.attention_instances = [
            RadixAttention(
                num_heads=divide(config.num_attention_heads, tp_size),
                head_dim=head_dim,
                # NOTE: We use Llama scale as default, if it's set by
                # Transformers, it's updated in sglang_flash_attention_forward
                scaling=head_dim**-0.5,
                num_kv_heads=divide(config.num_key_value_heads, tp_size),
                layer_id=i,
                quant_config=self.quant_config,
                prefix=f"{i}.attn",
            )
            for i in range(config.num_hidden_layers)
        ]

        # Model modifications
        self.replace_vocab_embed_class(self.model)

        # ForCausalLM modifications
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.get_input_embeddings().weight

        self.logits_processor = LogitsProcessor(config)

    def log_replacement(self, name: str, old_module: nn.Module, new_module: nn.Module):
        logger.debug("%s: %s -> %s", name, old_module, new_module)

    def tensor_parallel(self, tp_size: int):
        """
        Apply the model's tensor parallelization plan.
        Currently only supports linear layers.
        """
        if not self.model.supports_tp_plan:
            if tp_size <= 1:
                return

            raise ValueError(
                f"{type(self.model)} does not support tensor parallel yet!"
            )

        tp_plan = self.model._tp_plan

        def _tensor_parallel(module: nn.Module, prefix: str = ""):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)
                for pattern, style in tp_plan.items():
                    if re.match(pattern, qual_name) and isinstance(
                        child_module, nn.Linear
                    ):
                        new_module = replace_linear_class(
                            child_module, style, self.quant_config
                        )
                        setattr(module, child_name, new_module)
                        self.log_replacement(qual_name, child_module, new_module)
                else:
                    _tensor_parallel(child_module, prefix=qual_name)

        _tensor_parallel(self.model)

    def replace_vocab_embed_class(self, module: nn.Module):
        # Use native set input embeddings
        new_module = VocabParallelEmbedding(
            self.vocab_size,
            self.config.hidden_size,
            org_num_embeddings=self.config.vocab_size,
            quant_config=None,
        )
        self.log_replacement(
            "input embedding", self.model.get_input_embeddings(), new_module
        )
        self.model.set_input_embeddings(new_module)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        assert get_embedding is False, "embedding is not supported yet"
        aux_hidden_states = None
        hidden_states = self.model(
            input_ids[None, ...],
            use_cache=False,
            position_ids=positions[None, ...],
            forward_batch=forward_batch,
            attention_instances=self.attention_instances,
            return_dict=False,
        )[0][
            0, ...
        ]  # we remove batch dimension for now

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name not in params_dict:
                name = f"{self.model.base_model_prefix}.{name}"
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [TransformersForCausalLM]
