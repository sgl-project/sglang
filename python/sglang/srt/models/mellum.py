# Copyright 2023-2026 SGLang Team
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

"""
Inference-only Mellum (JetBrains/Mellum2-12B-A2.5B-Base) Qwen3-MoE variant with
interleaved sliding-window/full attention, per-layer-type RoPE and mixed dense/MoE
MLP layers.
"""

import logging
from typing import Any, Dict, Optional, Tuple, cast

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding, get_rope
from sglang.srt.layers.rotary_embedding.yarn import (
    yarn_find_correction_range,
    yarn_get_mscale,
)
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from sglang.srt.models.utils import (
    apply_qk_norm,
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.jit_kernel.fused_qknorm_rope import (
        can_use_fused_qk_norm_rope,
        fused_qk_norm_rope,
    )

logger = logging.getLogger(__name__)


def _compute_yarn_from_rope_params(
    rope_params: Dict[str, Any],
    head_dim: int,
    max_position_embeddings: int,
) -> Dict[str, float]:
    _default = {"factor": 1.0, "low": 0, "high": 0, "attention_factor": 1.0}
    if rope_params is None:
        return _default

    rope_type = rope_params.get("rope_type") or "default"
    if rope_type == "default":
        return _default

    base = rope_params.get("rope_theta", 10000)
    dim = head_dim
    factor = rope_params.get("factor", 1.0)
    attention_factor = rope_params.get("attention_factor")
    mscale = rope_params.get("mscale")
    mscale_all_dim = rope_params.get("mscale_all_dim")

    original_max_position_embeddings = rope_params.get(
        "original_max_position_embeddings", max_position_embeddings
    )
    if "original_max_position_embeddings" in rope_params:
        factor = max_position_embeddings / original_max_position_embeddings

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(
                yarn_get_mscale(factor, mscale)
                / yarn_get_mscale(factor, mscale_all_dim)
            )
        else:
            attention_factor = yarn_get_mscale(factor)

    beta_fast = rope_params.get("beta_fast") or 32
    beta_slow = rope_params.get("beta_slow") or 1

    truncate = rope_params.get("truncate", True)
    low, high = yarn_find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate
    )
    return {
        "factor": factor,
        "low": low,
        "high": high,
        "attention_factor": attention_factor,
    }


def get_attention_sliding_window_size(config: PretrainedConfig) -> Optional[int]:
    sw = getattr(config, "sliding_window", None)
    if sw is not None:
        return sw - 1
    return None


class MellumAttention(Qwen3MoeAttention):
    """
    Qwen3MoeAttention with per-layer sliding window and per-layer-type RoPE.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        start_layer: int = 0,
        rope_params: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        config: Optional[PretrainedConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sliding_window_size: int = -1,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        # Skip Qwen3MoeAttention.__init__ so that we can pass the correct
        # correct per-layer rope_theta / rope_scaling and sliding window.
        nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.start_layer = start_layer

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.config = config
        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        rope_params = rope_params or {}
        self.rope_theta = rope_params.get("rope_theta", 10000.0)
        rope_type = rope_params.get("rope_type") or "default"
        rope_scaling = rope_params if rope_type != "default" else None

        from sglang.srt.distributed import get_tensor_model_parallel_rank

        self.tp_rank = get_tensor_model_parallel_rank()

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.compatible_with_fused_kv_buffer = not isinstance(
            self.rotary_emb, MRotaryEmbedding
        )
        self.compatible_with_fused_qk_norm_rope = not isinstance(
            self.rotary_emb, MRotaryEmbedding
        ) and self.head_dim in (64, 128, 256)

        # TODO: Precompute YaRN parameters in Qwen3MoeAttention for all
        # inheriting models instead of recomputing them in every forward pass.
        self._yarn_params = _compute_yarn_from_rope_params(
            rope_params, self.head_dim, max_position_embeddings
        )
        _yarn_factor = self._yarn_params["factor"]

        self.use_fused_qk_norm_rope = (
            get_global_server_args().enable_fused_qk_norm_rope
            and self.compatible_with_fused_qk_norm_rope
            and _is_cuda
            and can_use_fused_qk_norm_rope(
                self.head_dim,
                self.rotary_emb.is_neox_style,
                torch.bfloat16,
                _yarn_factor != 1.0,
            )
        )
        self._used_fused_qk_norm_rope_last_call = False

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            prefix=add_prefix("attn", prefix),
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.alt_stream = alt_stream

    def apply_qk_norm_rope(self, qkv, positions, forward_batch):
        # Overridden to use pre-computed per-layer YaRN params.
        use_fused = self.use_fused_qk_norm_rope and qkv.dtype == torch.bfloat16
        if use_fused:
            theta = self.rope_theta
            fused_qk_norm_rope(
                qkv,
                self.num_heads,
                self.num_kv_heads,
                self.num_kv_heads,
                self.head_dim,
                self.q_norm.variance_epsilon,
                self.q_norm.weight,
                self.k_norm.weight,
                theta,
                self.rotary_emb.is_neox_style,
                positions,
                self._yarn_params["factor"],
                self._yarn_params["low"],
                self._yarn_params["high"],
                self._yarn_params["attention_factor"],
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            self._used_fused_qk_norm_rope_last_call = True
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = apply_qk_norm(
                q=q,
                k=k,
                q_norm=self.q_norm,
                k_norm=self.k_norm,
                head_dim=self.head_dim,
                alt_stream=self.alt_stream,
            )
            q, k = self.rotary_emb(
                positions,
                q,
                k,
                fused_set_kv_buffer_arg=(
                    create_fused_set_kv_buffer_arg(
                        value=v,
                        layer=self.attn,
                        forward_batch=forward_batch,
                    )
                    if enable_fused_set_kv_buffer(forward_batch)
                    and self.compatible_with_fused_kv_buffer
                    else None
                ),
            )
            self._used_fused_qk_norm_rope_last_call = False
        return q, k, v


class MellumDecoderLayer(Qwen3MoeDecoderLayer):
    """
    Qwen3MoeDecoderLayer with per-layer attention type, RoPE and dense/sparse MLP.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        # As with MellumAttention, skip parent __init__ to wire up
        # per-layer attention params, RoPE params, sliding window
        # and mixed dense/MoE MLP.
        nn.Module.__init__(self)

        self.config = config
        cfg = cast(Any, config)
        self.hidden_size = cfg.hidden_size
        self.layer_id = layer_id

        layer_types = cfg.layer_types
        if len(layer_types) != cfg.num_hidden_layers:
            raise ValueError(
                "Expected len(layer_types) == num_hidden_layers, got "
                f"{len(layer_types)} and {cfg.num_hidden_layers}"
            )
        layer_type = layer_types[layer_id]
        if layer_type not in ("sliding_attention", "full_attention"):
            raise ValueError(
                f"Unsupported layer_types[{layer_id}]={layer_type}; "
                "expected 'sliding_attention' or 'full_attention'"
            )

        rope_parameters = cfg.rope_parameters
        rope_params = rope_parameters.get(layer_type)
        if rope_params is None:
            raise ValueError(
                f"Missing rope_parameters[{layer_type}] for Mellum layer {layer_id}"
            )

        # Mellum routes SWA per-layer via layer_types. Preserve the configured
        # window regardless of legacy use_sliding_window post-init side effects.
        if layer_type == "sliding_attention":
            sliding_window_size = get_attention_sliding_window_size(config) or -1
        else:
            sliding_window_size = -1

        max_position_embeddings = cfg.max_position_embeddings
        head_dim = cfg.head_dim
        rms_norm_eps = cfg.rms_norm_eps
        attention_bias = cfg.attention_bias

        self.self_attn = MellumAttention(
            hidden_size=self.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_params=rope_params,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            sliding_window_size=sliding_window_size,
            alt_stream=alt_stream,
        )

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        mlp_layer_types = cfg.mlp_layer_types
        num_experts = cfg.num_experts

        if len(mlp_layer_types) != cfg.num_hidden_layers:
            raise ValueError(
                "Expected len(mlp_layer_types) == num_hidden_layers, got "
                f"{len(mlp_layer_types)} and {cfg.num_hidden_layers}"
            )

        def _is_sparse(lid: int) -> bool:
            if lid < 0 or lid >= cfg.num_hidden_layers:
                return False
            mlp_type = mlp_layer_types[lid]
            if mlp_type not in ("sparse", "dense"):
                raise ValueError(
                    f"Unsupported mlp_layer_types[{lid}]={mlp_type}; "
                    "expected 'sparse' or 'dense'"
                )
            return mlp_type == "sparse"

        self.is_layer_sparse = _is_sparse(layer_id)

        if self.is_layer_sparse:
            if num_experts <= 0:
                raise ValueError(
                    "Sparse MLP requested but num_experts <= 0 in Mellum config"
                )
            self.mlp = Qwen3MoeSparseMoeBlock(
                layer_id=layer_id,
                config=cfg,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                hidden_act=cfg.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )

        is_previous_layer_sparse = _is_sparse(layer_id - 1)
        is_next_layer_sparse = _is_sparse(layer_id + 1)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=cfg.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=rms_norm_eps)

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == cfg.num_hidden_layers - 1),
        )


class MellumModel(Qwen3MoeModel):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=MellumDecoderLayer,
        )


class MellumForCausalLM(Qwen3MoeForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        from sglang.srt.distributed import (
            get_attn_context_model_parallel_rank,
            get_attn_context_model_parallel_world_size,
            get_moe_data_parallel_world_size,
            get_pp_group,
        )
        from sglang.srt.layers.logits_processor import LogitsProcessor

        self.pp_group = get_pp_group()
        cfg = cast(Any, config)
        self.config = cfg
        self.quant_config = quant_config

        if hasattr(cfg, "max_window_layers"):
            logger.warning(
                "Mellum ignores config.max_window_layers and relies on "
                "config.layer_types for per-layer SWA routing."
            )

        self.model = MellumModel(cfg, quant_config, prefix=add_prefix("model", prefix))
        self.use_fused_qk_norm_rope = any(
            getattr(
                getattr(layer, "self_attn", None),
                "use_fused_qk_norm_rope",
                False,
            )
            for layer in self.model.layers
        )
        self.lm_head = ParallelLMHead(
            cfg.vocab_size,
            cfg.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(cfg)
        self.capture_aux_hidden_states = False

        self.attn_cp_size = get_attn_context_model_parallel_world_size()
        self.attn_cp_rank = get_attn_context_model_parallel_rank()
        self.moe_dp_size = get_moe_data_parallel_world_size()

        assert self.attn_cp_size % self.moe_dp_size == 0, (
            f"attn_cp_size ({self.attn_cp_size}) must be divisible by "
            f"moe_dp_size ({self.moe_dp_size})"
        )

    def _prepare_positions(
        self,
        positions: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if self.use_fused_qk_norm_rope:
            return positions.view(-1).to(dtype=torch.int32, device=device).contiguous()
        return positions

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if pp_proxy_tensors is not None:
            positions_device = pp_proxy_tensors["hidden_states"].device
        elif input_embeds is not None:
            positions_device = input_embeds.device
        else:
            positions_device = input_ids.device
        positions = self._prepare_positions(positions, positions_device)
        return super().forward(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],
        input_embeds: Optional[torch.Tensor] = None,
    ):
        positions_device = (
            input_embeds.device if input_embeds is not None else input_ids.device
        )
        positions = self._prepare_positions(positions, positions_device)
        return super().forward_split_prefill(
            input_ids,
            positions,
            forward_batch,
            split_interval,
            input_embeds,
        )

    def get_attention_sliding_window_size(self) -> Optional[int]:
        return get_attention_sliding_window_size(self.config)


EntryClass = MellumForCausalLM
