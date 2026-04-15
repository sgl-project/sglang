# Copyright 2025-2026 SGLang Team
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
"""Configuration for GLM-4.7-Flash (glm4_moe_lite) with MLA attention."""

from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig


class Glm4MoeLiteConfig(Glm4MoeConfig):
    """
    Configuration for GLM-4.7-Flash (architecture: Glm4MoeLiteForCausalLM).

    Extends Glm4MoeConfig with Multi-head Latent Attention (MLA) parameters:
    q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim.

    Transformers does not natively register "glm4_moe_lite" as a model type.
    Registering this class with AutoConfig allows SGLang to load the model config
    during server argument parsing without requiring trust_remote_code custom files.
    """

    model_type = "glm4_moe_lite"

    def __init__(
        self,
        q_lora_rank: int = 768,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        **kwargs,
    ):
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        super().__init__(**kwargs)
