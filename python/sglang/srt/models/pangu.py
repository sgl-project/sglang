# Copyright 2024 SGLang Team
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

"""Native SGLang implementation for OpenPangu Embedded models.

The architecture matches the OpenPangu Embedded 7B model released at
https://huggingface.co/FreedomIntelligence/openPangu-Embedded-7B-V1.1.

The model is architecturally a decoder-only transformer that is largely
compatible with LLaMA style blocks (RMSNorm + rotary attention +
SwiGLU MLP).  The key differences compared to some LLaMA variants are:

* Attention and projection layers keep bias terms enabled (`config.bias=True`).
* RoPE uses the large base value shipped with the official checkpoints
  (`rope_theta=16M`).

SGLang's LLaMA implementation already supports both behaviours, so we reuse
the optimized kernels and infrastructure and expose an explicit
`PanguEmbeddedForCausalLM` entry point that plugs into the model registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.models.llama import LlamaForCausalLM, LlamaModel

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from sglang.srt.layers.quantization.base_config import QuantizationConfig


# Thin alias; we reuse the optimized LLaMA block composition directly.
PanguEmbeddedModel = LlamaModel


class PanguEmbeddedForCausalLM(LlamaForCausalLM):
    """Pangu Embedded decoder-only language model.

    We override `_init_model` to construct `PanguEmbeddedModel`, but otherwise
    inherit the high-performance runtime from the LLaMA implementation.  The
    weight mapping logic in :class:`LlamaForCausalLM` is compatible with the
    official checkpoint structure (q/k/v projections + SwiGLU MLP), so no
    additional remapping is required.
    """

    def _init_model(
        self,
        config: "PretrainedConfig",
        quant_config: Optional["QuantizationConfig"] = None,
        prefix: str = "",
    ):
        return PanguEmbeddedModel(config, quant_config=quant_config, prefix=prefix)


EntryClass = [PanguEmbeddedForCausalLM]
