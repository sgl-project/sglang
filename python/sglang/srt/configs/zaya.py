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

from sglang.srt.runtime_context import get_parallel

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

        # ``zaya_layers`` entries are either the literal ``"a"`` (attention) or
        # an integer (expert count of a MoE layer). The scalar
        # ``num_hidden_layers`` can disagree with ``len(zaya_layers)`` for
        # historical reasons, so a non-empty list takes precedence.
        self.zaya_layers = list(zaya_layers) if zaya_layers else None
        if self.zaya_layers:
            self.num_hidden_layers = len(self.zaya_layers)
        else:
            self.num_hidden_layers = num_hidden_layers

        # Derive each active scalar from the first non-zero entry of the
        # corresponding per-layer list; ZAYA1 shares one value across all
        # attention layers and one across all MoE layers. Without a list the
        # constructor argument is used unchanged.
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

        # The *inclusive* window, matching the HF ``sliding_window`` convention
        # ``ModelConfig`` reads. The attention backends take the exclusive
        # ``window - 1`` via ``get_attention_sliding_window_size`` instead.
        window = self.swa_window_size
        self.sliding_window_size = window

        # Opt in to the hybrid-SWA KV pool when the checkpoint interleaves
        # sliding-window layers. ``ModelConfig.is_hybrid_swa_model`` honours this
        # flag (paired with ``hybrid_layer_pattern``) as a generic escape from
        # its architecture allowlist, so ZAYA1 needs no entry there; base
        # checkpoints omit ``swa_layers`` and stay on the single-pool path.
        #
        # SWA-KV and per-request linear state compose with no new pool type: the
        # KV side takes SWAKVPool + SWATokenToKVPoolAllocator while the CCA conv
        # state rides on HybridReqToTokenPool.mamba_pool, as Inkling does.
        self.is_hybrid_swa = window is not None

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

    # -- CCA v2 lag state (conv[1]) ------------------------------------------

    @property
    def cca_cache_projected_v2(self) -> bool:
        """Whether conv[1] caches ``W_v2 . hs`` instead of the raw ``hs``.

        CCA's second state entry exists only to feed ``val_proj2`` with the
        previous token's hidden state. That projection is linear, so caching its
        *output* is the same function as caching its input and re-projecting, and
        the output is ``latent_k_dim / 2`` wide instead of ``hidden_size``. Two
        conditions gate it:

        * ``attention_bias`` off. MambaPool zeroes a freshly allocated slot and
          the first ``val_proj2`` input is defined to be zero, which only
          ``W . 0 == 0`` reproduces.
        * ``num_query_groups`` even, so ``val_proj1`` / ``val_proj2`` split the K
          heads on a head boundary; an odd count makes the per-rank slicing
          channel- rather than head-aligned.

        ``CCA.__init__`` derives the same predicate from its constructor
        arguments; both must agree or the pool entry and the value written into
        it disagree in width, which raises on the first prefill.
        """
        return (not bool(getattr(self, "attention_bias", False))) and (
            self.num_query_groups % 2 == 0
        )

    @property
    def cca_v2_state_dim(self) -> int:
        """Feature width of the CCA conv[1] pool entry."""
        if self.cca_cache_projected_v2:
            return (self.num_query_groups * self.head_dim) // 2
        return self.hidden_size

    # -- Sliding-window attention (ZAYA1-74B) -------------------------------

    def sliding_window_for_layer(self, layer_id: int) -> int:
        """Sliding-window size for ``layer_id`` (0 == full attention).

        ``swa_layers`` is aligned with the global layer index: the window size
        for a sliding-window attention layer, 0 for a full-attention or MoE one.
        Base checkpoints omit it, so every attention layer is full attention.
        """
        if not self.swa_layers:
            return 0
        return int(self.swa_layers[layer_id])

    @property
    def swa_window_size(self) -> Optional[int]:
        """The single sliding-window size shared by every SWA layer, or None.

        The runtime tracks one global sliding-window size for the attention
        backend, so all SWA layers must share the same window. Checkpoints
        without ``swa_layers`` (or with all-zero entries) report None.
        """
        if not self.swa_layers:
            return None
        windows = {int(w) for w in self.swa_layers if int(w) > 0}
        if not windows:
            return None
        assert len(windows) == 1, (
            "ZAYA1 expects a single sliding-window size across all SWA layers, "
            f"got {sorted(windows)}"
        )
        return next(iter(windows))

    def get_attention_sliding_window_size(self) -> Optional[int]:
        """Global window size handed to the attention backend, or None.

        Returns ``window - 1`` so the backend applies an inclusive
        ``[i-w+1, i]`` window -- the exclusive convention shared across SGLang's
        attention backends.
        """
        window = self.swa_window_size
        return (window - 1) if window is not None else None

    @property
    def swa_attention_layer_ids(self) -> List[int]:
        """Attention layers that use the sliding window (empty when no SWA)."""
        if self.swa_window_size is None:
            return []
        return [i for i in self.full_attention_layer_ids if self.swa_layers[i]]

    @property
    def hybrid_layer_pattern(self) -> Optional[List[int]]:
        """Per-layer KV class: 1 = sliding attention, 0 = full attention, -1 = none.

        ``ModelConfig.get_hybrid_layer_ids`` consumes this generic opt-in (paired
        with ``is_hybrid_swa``) so ZAYA1 needs no entry in the hybrid-SWA
        architecture allowlist. It derives ``swa_attention_layer_ids`` from the
        ``== 1`` entries and ``full_attention_layer_ids`` from the ``== 0`` ones,
        so any other value -- here -1 -- is excluded from both lists.

        ZAYA1's odd layers are MoE and hold no KV at all, so they MUST be -1 and
        not 0. Those lists do not merely index the pools, they *size* them:
        reporting MoE layers as full-attention made ``SWAKVPool``'s full sub-pool
        90 layers wide instead of 30 on the 74B, tripling its per-token cost.
        """
        if self.swa_window_size is None:
            return None
        attention_layers = set(self.full_attention_layer_ids)
        return [
            (
                (1 if self.sliding_window_for_layer(i) else 0)
                if i in attention_layers
                else -1
            )
            for i in range(self.num_hidden_layers)
        ]

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

        # ``conv[0]`` (conv_qk left padding) is sized per TP rank because CCA is
        # head-parallel. ``conv[1]`` is the one-token ``val_proj2`` lag, holding
        # the *projected* value rather than the raw hidden state (see
        # ``cca_cache_projected_v2``). ``val_proj2`` is replicated, so that entry
        # stays the same width on every rank -- a rank whose K heads all come
        # from ``val_proj1`` simply leaves it untouched -- which is what keeps
        # ``max_mamba_cache_size``, and so the replicated scheduler's slot
        # accounting, identical across the attention-TP group.
        #
        # Use the *attention* TP world size: the same accessor ``ZayaAttention``
        # / ``CCA`` use to split heads, so the cache shape and the per-rank
        # ``in_out_ch`` stay in lockstep under plain TP and DP attention alike.
        try:
            tp_size = get_parallel().attn_tp_size
        except (AssertionError, RuntimeError, ValueError):
            tp_size = 1

        in_out_ch_full = (
            self.num_attention_heads + self.num_key_value_heads
        ) * self.head_dim
        assert in_out_ch_full % tp_size == 0, (
            f"CCA channels ({in_out_ch_full}) must be divisible by attention "
            f"TP size ({tp_size}); both num_attention_heads and num_query_groups "
            "must be divisible by attention tp_size for ZAYA1 head-parallel "
            "attention."
        )
        in_out_ch_per_rank = in_out_ch_full // tp_size
        total_padding = (self.cca_time0 - 1) + (self.cca_time1 - 1)

        shape = Mamba2StateShape(
            conv=[
                (in_out_ch_per_rank, total_padding),
                (self.cca_v2_state_dim, 1),
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
