# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/teleflm.py

from typing import List, Optional, Tuple, Union

import torch
from transformers import LlamaConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.llama import LlamaForCausalLM, LlamaModel


class TeleFLMModel(LlamaModel):
    """
    This implementation is based on the µScaling paper presented at
    the ICLR 2025 Workshop:
    NanoLM: An Affordable LLM Study Benchmark \
    via Accurate Loss Prediction across Scales
    by Yiqun Yao et al.
    Available at: https://openreview.net/forum?id=IwaPYg1SCA
    arXiv preprint: https://arxiv.org/abs/2304.06875
    """

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.use_mup = getattr(self.config, "use_mup", False)
        if self.use_mup:
            self.input_mult = self.config.input_mult

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:
        if self.pp_group.is_first_rank and input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
            if self.use_mup:
                input_embeds = input_embeds * self.input_mult

        return super().forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )


class TeleFLMForCausalLM(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.use_mup = getattr(self.config, "use_mup", False)
        if self.use_mup:
            self.mup_scale_factor = self.config.mup_scale_factor
            self.output_mult = self.config.output_mult / self.mup_scale_factor
            self.logits_processor.logit_scale = self.output_mult

    def _init_model(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return TeleFLMModel(config, quant_config=quant_config, prefix=prefix)


EntryClass = TeleFLMForCausalLM
