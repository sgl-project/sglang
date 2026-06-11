# SPDX-License-Identifier: Apache-2.0
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
"""Configuration class for Zyphra ZAYA1 series models."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from transformers.configuration_utils import PretrainedConfig

if TYPE_CHECKING:
    from sglang.srt.configs.mamba_utils import Mamba2CacheParams


class ZayaConfig(PretrainedConfig):
    """HuggingFace configuration for ZAYA1 hybrid (CCA attention + MoE) models.

    Mirrors the field set used by Zyphra/ZAYA1-base/config.json. Most fields
    are surfaced as constructor arguments so the same class can be instantiated
    either from a published checkpoint via ``AutoConfig.from_pretrained`` or
    programmatically in unit tests.
    """

    model_type = "zaya"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        cca: bool = True,
        num_query_groups: int = 2,
        use_cache: bool = True,
        attention_bias: bool = False,
        lm_head_bias: bool = False,
        vocab_size: int = 262272,
        hidden_size: int = 2048,
        ffn_hidden_size: int = 4096,
        num_hidden_layers: int = 80,
        num_experts: int = 16,
        num_attention_heads: int = 8,
        head_dim: int = 128,
        activation_func: str = "swiglu",
        max_position_embeddings: int = 32768,
        norm_epsilon: float = 1e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = True,
        rope_theta: float = 1_000_000.0,
        attention_dropout: float = 0.0,
        moe_router_topk: int = 1,
        normalization: str = "RMSNorm",
        zaya_mlp_expansion=256,
        zaya_use_mod: bool = True,
        zaya_high_prec: bool = True,
        zaya_use_eda: bool = True,
        add_bias_linear: bool = False,
        gated_linear_unit: bool = True,
        scale_residual_merge: bool = True,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = True,
        apply_rope_fusion: bool = True,
        bias_activation_fusion: bool = True,
        activation_func_fp8_input_store: bool = False,
        sliding_window=None,
        rope_scaling=None,
        rope_parameters=None,
        partial_rotary_factor: float = 0.5,
        num_key_value_heads: int = 2,
        clamp_temp: bool = False,
        cca_time0: int = 2,
        cca_time1: int = 2,
        swa_layers=None,
        swa_rotary_base=None,
        zaya_layers=None,
        cca_num_q_heads=None,
        num_query_groups_list=None,
        ffn_hidden_size_list=None,
        kv_channels=None,
        _attn_implementation: str = "eager",
        **kwargs,
    ):
        self.cca = cca
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.lm_head_bias = lm_head_bias
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        # ZAYA1-base ships a ``zaya_layers`` list whose entries are either the
        # literal string ``"a"`` (attention layer) or an integer (number of
        # experts in a MoE layer). When present it is the source of truth for
        # both the total layer count and the per-layer placement. The HF
        # config also carries a scalar ``num_hidden_layers`` that can disagree
        # with ``len(zaya_layers)`` for historical reasons, so the list takes
        # precedence whenever it is non-empty.
        self.zaya_layers = list(zaya_layers) if zaya_layers else None
        if self.zaya_layers:
            self.num_hidden_layers = len(self.zaya_layers)
        else:
            self.num_hidden_layers = num_hidden_layers

        # When the per-layer lists are present, derive each active scalar
        # field from the first non-zero entry of the corresponding list.
        # This matches ZAYA1-base in practice: every attention layer shares
        # the same ``cca_num_q_heads`` (e.g. 8) and ``num_query_groups``
        # (e.g. 2), and every MoE layer shares the same ``ffn_hidden_size``
        # (e.g. 4096) and ``zaya_mlp_expansion`` (e.g. 256). When no list is
        # provided, the constructor argument is used unchanged.
        self.cca_num_q_heads_list = list(cca_num_q_heads) if cca_num_q_heads else None
        self.num_query_groups_list = (
            list(num_query_groups_list) if num_query_groups_list else None
        )
        self.ffn_hidden_size_list = (
            list(ffn_hidden_size_list) if ffn_hidden_size_list else None
        )
        if isinstance(zaya_mlp_expansion, (list, tuple)):
            self.zaya_mlp_expansion_list = list(zaya_mlp_expansion)
            zaya_mlp_expansion_scalar = next(
                (v for v in self.zaya_mlp_expansion_list if v), 256
            )
        else:
            self.zaya_mlp_expansion_list = None
            zaya_mlp_expansion_scalar = int(zaya_mlp_expansion)

        if self.cca_num_q_heads_list:
            self.num_attention_heads = next(
                (v for v in self.cca_num_q_heads_list if v), num_attention_heads
            )
        else:
            self.num_attention_heads = num_attention_heads

        if self.num_query_groups_list:
            self.num_query_groups = next(
                (v for v in self.num_query_groups_list if v), num_query_groups
            )
        else:
            self.num_query_groups = num_query_groups

        if self.ffn_hidden_size_list:
            self.ffn_hidden_size = next(
                (v for v in self.ffn_hidden_size_list if v), ffn_hidden_size
            )
        else:
            self.ffn_hidden_size = ffn_hidden_size

        self.zaya_mlp_expansion = zaya_mlp_expansion_scalar

        # The HF config exposes the per-head dim as ``kv_channels``; accept
        # either spelling and keep both attributes in sync for downstream code.
        if head_dim is None and kv_channels is not None:
            head_dim = int(kv_channels)
        self.head_dim = head_dim
        self.kv_channels = kv_channels if kv_channels is not None else head_dim
        assert self.head_dim is not None, "head_dim is required for ZayaConfig"
        assert (
            self.num_query_groups == num_key_value_heads
        ), "num_query_groups must equal num_key_value_heads for ZAYA1 checkpoints"
        self.num_key_value_heads = num_key_value_heads
        self.activation_func = activation_func
        self.max_position_embeddings = max_position_embeddings
        self.norm_epsilon = norm_epsilon
        self.normalization = normalization
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_dropout = attention_dropout
        self.moe_router_topk = moe_router_topk
        self.zaya_use_mod = zaya_use_mod
        self.zaya_high_prec = zaya_high_prec
        self.zaya_use_eda = zaya_use_eda
        self.add_bias_linear = add_bias_linear
        self.gated_linear_unit = gated_linear_unit
        self.scale_residual_merge = scale_residual_merge
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.apply_rope_fusion = apply_rope_fusion
        self.bias_activation_fusion = bias_activation_fusion
        self.activation_func_fp8_input_store = activation_func_fp8_input_store
        self.sliding_window = sliding_window
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        if isinstance(rope_parameters, dict):
            rope_parameters_dict = dict(rope_parameters)
        elif isinstance(rope_scaling, dict):
            rope_parameters_dict = dict(rope_scaling)
        else:
            rope_parameters_dict = {"rope_type": "default"}
        if "type" in rope_parameters_dict:
            rope_parameters_dict.setdefault(
                "rope_type", rope_parameters_dict.pop("type")
            )
        rope_parameters_dict.setdefault("rope_theta", rope_theta)
        rope_parameters_dict.setdefault("partial_rotary_factor", partial_rotary_factor)
        self.rope_parameters = rope_parameters_dict

        self.clamp_temp = clamp_temp
        self.cca_time0 = cca_time0
        self.cca_time1 = cca_time1
        self.swa_layers = swa_layers
        self.swa_rotary_base = swa_rotary_base
        self._attn_implementation = _attn_implementation

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            **kwargs,
        )

    # -- Hybrid model interface (HybridReqToTokenPool / MambaPool) ----------

    @property
    def full_attention_layer_ids(self) -> List[int]:
        if self.zaya_layers:
            return [i for i, lt in enumerate(self.zaya_layers) if lt == "a"]
        return [i for i in range(self.num_hidden_layers) if i % 2 == 0]

    @property
    def linear_layer_ids(self) -> List[int]:
        return self.full_attention_layer_ids

    @property
    def mamba_chunk_size(self) -> int:
        return 1

    @property
    def mamba2_cache_params(self) -> Optional[Mamba2CacheParams]:
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateShape,
            mamba2_state_dtype,
        )

        attn_layer_ids = self.linear_layer_ids
        if not attn_layer_ids:
            return None

        # ``conv[0]`` (conv_qk left padding) is sized per TP rank because CCA
        # is head-parallel. ``conv[1]`` (prev_hs) carries the full hidden_state
        # and feeds the replicated val_proj1 / val_proj2, so it stays at full
        # ``hidden_size`` on every rank.
        #
        # Use the *global* TP world size -- the same accessor that
        # ``ZayaAttention`` / ``CCA`` use to split heads and over which
        # ``o_proj`` all-reduces -- so the cache shape and the per-rank
        # ``in_out_ch`` stay in lockstep. ZAYA1 asserts the attention-TP group
        # equals the global TP group (DP attention is unsupported), so the two
        # are always identical in practice.
        try:
            from sglang.srt.distributed import (
                get_tensor_model_parallel_world_size,
            )

            tp_size = get_tensor_model_parallel_world_size()
        except (AssertionError, RuntimeError):
            tp_size = 1

        in_out_ch_full = (
            self.num_attention_heads + self.num_key_value_heads
        ) * self.head_dim
        assert in_out_ch_full % tp_size == 0, (
            f"CCA channels ({in_out_ch_full}) must be divisible by TP size "
            f"({tp_size}); both num_attention_heads and num_query_groups must "
            "be divisible by tp_size for ZAYA1 head-parallel attention."
        )
        in_out_ch_per_rank = in_out_ch_full // tp_size
        total_padding = (self.cca_time0 - 1) + (self.cca_time1 - 1)

        shape = Mamba2StateShape(
            conv=[
                (in_out_ch_per_rank, total_padding),
                (self.hidden_size, 1),
            ],
            temporal=(1, 1, 0),
            intermediate_size=in_out_ch_per_rank,
            conv_dim=in_out_ch_per_rank,
            ssm_state_size=0,
            num_heads=1,
            head_dim=1,
            state_size=0,
            conv_kernel=total_padding + 1,
        )

        return Mamba2CacheParams(
            shape=shape,
            layers=attn_layer_ids,
            dtype=mamba2_state_dtype(self),
        )


def register_zaya_config() -> None:
    """Register :class:`ZayaConfig` with HuggingFace ``AutoConfig``.

    Safe to call multiple times. ``AutoConfig.register`` raises ``ValueError``
    on duplicate registration, which is suppressed so importing this module
    stays idempotent.
    """
    try:
        from transformers import AutoConfig

        AutoConfig.register(ZayaConfig.model_type, ZayaConfig)
    except (ValueError, ImportError):
        # Either the installed ``transformers`` does not expose
        # ``AutoConfig.register``, or the "zaya" model type is already
        # registered – nothing to do in either case.
        pass


register_zaya_config()
