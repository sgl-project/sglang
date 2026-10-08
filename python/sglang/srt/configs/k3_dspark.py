# SPDX-License-Identifier: Apache-2.0
"""Config for the Inferact Kimi-K3 DSpark draft (``K3DSparkModel``).

The checkpoint is a 5-layer absorbed MLA stack, not the GQA
``DSparkDraftModel``. ``model_type`` is ``k3_dspark``, which transformers does
not know, so this class is registered with ``AutoConfig``.
"""

from transformers import PretrainedConfig


class K3DSparkConfig(PretrainedConfig):
    model_type = "k3_dspark"

    def __init__(
        self,
        hidden_size: int = 7168,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        mla_use_nope: bool = False,
        mla_use_output_gate: bool = False,
        vocab_size: int = 163840,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 1048576,
        rope_theta: float = 50000.0,
        num_target_layers: int = 5,
        target_hidden_size: int = 7168,
        target_num_hidden_layers: int = 93,
        target_layer_ids=None,
        mask_token_id=None,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        enable_confidence_head: bool = True,
        confidence_head_with_markov: bool = True,
        hidden_act: str = "silu",
        n_routed_experts=None,
        block_size=None,
        rope_parameters=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.num_target_layers = num_target_layers
        self.target_hidden_size = target_hidden_size
        self.target_num_hidden_layers = target_num_hidden_layers
        self.target_layer_ids = target_layer_ids
        self.mask_token_id = mask_token_id
        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type
        self.enable_confidence_head = enable_confidence_head
        self.confidence_head_with_markov = confidence_head_with_markov
        self.hidden_act = hidden_act
        self.n_routed_experts = n_routed_experts
        self.block_size = block_size
        self.rope_parameters = rope_parameters
