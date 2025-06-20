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

import copy
from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    ROPE_INIT_FUNCTIONS,
    AutoModel,
    Gemma3nTextConfig,
    PretrainedConfig,
    PreTrainedModel,
)

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import GeluAndMul
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils import add_prefix, make_layers


# Aligned with HF's implementation, using sliding window inclusive with the last token
# SGLang assumes exclusive
def get_attention_sliding_window_size(config):
    return config.sliding_window - 1


class Gemma3nRMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        scale_shift: float = 0.0,
        with_scale: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.scale_shift = scale_shift
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.tensor(1.0), persistent=False)

    def _norm(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Llama does x.to(float16) * w whilst Gemma3n is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x) * (self.weight + self.scale_shift).type_as(x)
        return output.type_as(x)


class Gemma3nTextScaledWordEmbedding(VocabParallelEmbedding):
    """
    This module overrides VocabParallelEmbedding's forward by multiplying with embeddings scale.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        org_num_embeddings: Optional[int] = None,
        embed_scale: float = 1.0,
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            quant_config=quant_config,
            prefix=prefix,
            org_num_embeddings=org_num_embeddings,
        )
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = super().forward(input_ids)
        return embeddings * self.embed_scale


class Gemma3nMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        activation_sparsity: float = 0.0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma3n uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_activation` to "
                "`gelu_pytorch_tanh`."
            )
        # Use proper GELU with tanh approximation as specified
        self.act_fn = GeluAndMul()
        self.activation_sparsity = activation_sparsity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)

        # Split gate and up projections
        gate_proj, up_proj = gate_up.chunk(2, dim=-1)

        # Apply activation sparsity if needed
        if self.activation_sparsity > 0.0 and self.training:
            gate_proj = self._gaussian_topk(gate_proj)

        # Apply GELU activation to gate projection and multiply with up projection
        activated_gate = self.act_fn(gate_proj)
        x = activated_gate * up_proj
        x, _ = self.down_proj(x)
        return x

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        target_sparsity_tensor = torch.tensor(
            self.activation_sparsity, dtype=torch.float32, device=inputs.device
        )
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier = normal_dist.icdf(target_sparsity_tensor)
        std_multiplier = std_multiplier.type(inputs.dtype)
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return F.relu(inputs - cutoff_x)


class Gemma3nLaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(
        self,
        config: Gemma3nTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.linear_left = ColumnParallelLinear(
            config.hidden_size,
            config.laurel_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("linear_left", prefix),
        )
        self.linear_right = RowParallelLinear(
            config.laurel_rank,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("linear_right", prefix),
        )
        self.post_laurel_norm = Gemma3nRMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        laurel_x, _ = self.linear_left(x)
        laurel_x, _ = self.linear_right(laurel_x)
        normed_laurel_x = self.post_laurel_norm(laurel_x)
        return x + normed_laurel_x


class Gemma3nAltUp(nn.Module):
    """Alternating Updates (AltUp)"""

    def __init__(
        self,
        config: Gemma3nTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.correct_output_scale = nn.Parameter(
            torch.zeros(config.hidden_size, dtype=torch.float32)
        )
        self.correction_coefs = ColumnParallelLinear(
            config.altup_num_inputs,
            config.altup_num_inputs,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("correction_coefs", prefix),
        )
        self.prediction_coefs = ColumnParallelLinear(
            config.altup_num_inputs,
            config.altup_num_inputs**2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("prediction_coefs", prefix),
        )
        self.modality_router = ColumnParallelLinear(
            config.hidden_size,
            config.altup_num_inputs,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("modality_router", prefix),
        )

        self.router_norm = Gemma3nRMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )

        self.register_buffer(
            "router_input_scale",
            torch.tensor(config.hidden_size**-1.0),
            persistent=False,
        )

    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
        router_inputs = self.router_norm(x) * self.router_input_scale.type(
            self.router_norm.weight.dtype
        )
        routed, _ = self.modality_router(router_inputs)
        return torch.tanh(routed.float())

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predicts the output of a layer using a trainable map."""
        modalities = self.compute_router_modalities(
            hidden_states[self.config.altup_active_idx]
        )

        if self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight.data.clamp_(
                -self.config.altup_coef_clip, self.config.altup_coef_clip
            )

        all_coefs, _ = self.prediction_coefs(modalities)
        all_coefs = all_coefs.reshape(
            *modalities.shape[:-1],
            self.config.altup_num_inputs,
            self.config.altup_num_inputs,
        ).permute(0, 1, 3, 2)

        # permute hidden_states to [batch_size, num_tokens, hidden_size, altup_num_inputs]
        predictions = torch.matmul(hidden_states.float().permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2)  # undo the permute
        predictions += hidden_states  # add the original input
        return predictions.contiguous().type_as(hidden_states)

    def correct(
        self, predictions: torch.Tensor, activated: torch.Tensor
    ) -> torch.Tensor:
        """Corrects the predictions relative to the activated inputs."""
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.config.altup_active_idx]
        innovation = innovation.repeat(self.config.altup_num_inputs, 1, 1, 1)

        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight.data.clamp_(
                -self.config.altup_coef_clip, self.config.altup_coef_clip
            )

        all_coefs, _ = self.correction_coefs(modalities)
        all_coefs = (all_coefs + 1.0).permute(2, 0, 1).unsqueeze(-1)

        corrected = torch.mul(innovation, all_coefs)
        corrected += predictions
        return corrected.contiguous().type_as(activated)

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        """Scales the provided 3D tensor."""
        return corrected * self.correct_output_scale

    def forward(
        self, hidden_states: torch.Tensor, activated: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts, correct, and optionally scales the output of a layer using trainable maps."""
        predictions = self.predict(hidden_states)
        corrected = self.correct(predictions=predictions, activated=activated)
        output = corrected[self.config.altup_active_idx]
        if self.config.altup_correct_scale:
            output = self.scale_corrected_output(output)
        return corrected, output


class Gemma3nAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        layer_id: int,
        config: Gemma3nTextConfig,
        max_position_embeddings: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        hidden_size = config.hidden_size
        head_dim = getattr(
            config, "head_dim", hidden_size // config.num_attention_heads
        )
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_rescale_scalar / config.query_pre_attn_scalar

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # Determine if layer uses sliding window based on pattern
        self.is_sliding = config.layer_types[layer_id] == "sliding_attention"

        # Check if this is a KV shared layer
        first_kv_shared_layer_idx = (
            config.num_hidden_layers - config.num_kv_shared_layers
        )
        self.is_kv_shared_layer = layer_id >= first_kv_shared_layer_idx

        # Compute the layer index from which shared KV cache values will be retrieved
        if not self.is_kv_shared_layer:
            self.kv_shared_layer_index = None
        elif self.is_sliding:
            self.kv_shared_layer_index = first_kv_shared_layer_idx - 2
        else:
            self.kv_shared_layer_index = first_kv_shared_layer_idx - 1

        # Initialize the rotary embedding
        if self.is_sliding:
            self.rope_theta = config.rope_local_base_freq
            self.rope_scaling = {"rope_type": "default"}
            self.sliding_window = get_attention_sliding_window_size(config)
        else:
            self.rope_theta = config.rope_theta
            self.rope_scaling = config.rope_scaling
            self.sliding_window = None

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=getattr(config, "attn_logit_softcapping", 0.0) or 0.0,
            sliding_window_size=self.sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        # Gemma3n adds normalization for q, k, v
        self.qkv_norm = Gemma3nRMSNorm(
            dim=config.head_dim,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Apply normalization to q, k, v
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = q.transpose(0, 1).unsqueeze(0)
        q = self.qkv_norm(q)

        # Check if we should use shared KV cache
        if self.is_kv_shared_layer and self.kv_shared_layer_index is not None:
            # For KV shared layers, we skip K/V computation and normalization
            # The RadixAttention will handle retrieving shared KV from cache
            k = None
            v = None
        else:
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = k.transpose(0, 1).unsqueeze(0)
            k = self.qkv_norm(k)

            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = v.transpose(0, 1).unsqueeze(0)
            v = self.qkv_norm(v)

        # Apply rotary embeddings
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Reshape for attention
        q = q.permute(0, 2, 1, 3)
        if k is not None:
            k = k.permute(0, 2, 1, 3)
        if v is not None:
            v = v.permute(0, 2, 1, 3)

        attn_output = self.attn(
            q,
            k,
            v,
            forward_batch=forward_batch,
            kv_shared_layer_idx=(
                self.kv_shared_layer_index if self.is_kv_shared_layer else None
            ),
        )

        # Compatible with triton backend which returns [1, s, h, head_dim]
        if attn_output.dim() == 4 and attn_output.shape[0] == 1:
            attn_output = attn_output.squeeze(0)
            attn_output = attn_output.flatten(-2, -1)

        output, _ = self.o_proj(attn_output)
        return output


class Gemma3nRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """Dynamic RoPE frequency update."""
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Gemma3nDecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.attention_type = config.layer_types[layer_id]

        self.self_attn = Gemma3nAttention(
            layer_id=layer_id,
            config=config,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        activation_sparsity = config.activation_sparsity_pattern[layer_id]
        self.mlp = Gemma3nMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            activation_sparsity=activation_sparsity,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = Gemma3nRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=True
        )
        self.post_attention_layernorm = Gemma3nRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=True
        )
        self.pre_feedforward_layernorm = Gemma3nRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=True
        )
        self.post_feedforward_layernorm = Gemma3nRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=True
        )

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input

        self.altup = Gemma3nAltUp(
            config, quant_config, prefix=add_prefix("altup", prefix)
        )
        self.laurel = Gemma3nLaurelBlock(
            config, quant_config, prefix=add_prefix("laurel", prefix)
        )

        self.per_layer_input_gate = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size_per_layer_input,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("per_layer_input_gate", prefix),
        )
        self.per_layer_projection = RowParallelLinear(
            self.hidden_size_per_layer_input,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("per_layer_projection", prefix),
        )
        self.post_per_layer_input_norm = Gemma3nRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=True
        )
        self.is_sliding = self.self_attn.is_sliding

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        per_layer_input: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        predictions = self.altup.predict(hidden_states)
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        attn = self.self_attn(
            positions=positions,
            hidden_states=active_prediction_normed,
            position_embeddings=position_embeddings,
            forward_batch=forward_batch,
            **kwargs,
        )
        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) / torch.sqrt(torch.tensor(2.0))

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        first_prediction = corrected_predictions[self.config.altup_active_idx]
        if self.config.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        # per_layer_input_gate
        first_prediction, _ = self.per_layer_input_gate(first_prediction)
        first_prediction = F.gelu(first_prediction, approximate="tanh")
        first_prediction = torch.multiply(first_prediction, per_layer_input)

        # per_layer_projection
        first_prediction, _ = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)
        corrected_predictions[1:] += first_prediction

        return corrected_predictions


class Gemma3nTextModel(PreTrainedModel):
    def __init__(
        self,
        config: Gemma3nTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        # Gemma3n downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        self.embed_tokens = Gemma3nTextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
            embed_scale=self.config.hidden_size**0.5,
        )

        self.norm = Gemma3nRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )

        self.rotary_emb = Gemma3nRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Local RoPE layer with different theta
        config_local = copy.deepcopy(config)
        config_local.rope_theta = config.rope_local_base_freq
        config_local.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3nRotaryEmbedding(config=config_local)

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Gemma3nDecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )

        # Per-layer input embeddings
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input

        self.embed_tokens_per_layer = Gemma3nTextScaledWordEmbedding(
            config.vocab_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens_per_layer", prefix),
            embed_scale=config.hidden_size_per_layer_input**0.5,
        )

        self.per_layer_model_projection = ColumnParallelLinear(
            self.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("per_layer_model_projection", prefix),
        )

        self.per_layer_projection_norm = Gemma3nRMSNorm(
            dim=config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )

        self.altup_projections = make_layers(
            self.config.altup_num_inputs - 1,
            lambda idx, prefix: ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("altup_projections", prefix),
        )

        self.altup_unembed_projections = make_layers(
            self.config.altup_num_inputs - 1,
            lambda idx, prefix: ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("altup_unembed_projections", prefix),
        )

        self.register_buffer(
            "per_layer_projection_scale",
            torch.tensor(self.hidden_size**-0.5),
            persistent=False,
        )
        self.register_buffer(
            "per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False
        )

        self.post_init()

    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        per_layer_inputs_mask = torch.logical_and(
            input_ids >= 0, input_ids < self.vocab_size
        )
        tokens = torch.where(
            per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids)
        )
        embeddings = self.embed_tokens_per_layer(tokens)
        return embeddings.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_layer_projection, _ = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection *= self.per_layer_projection_scale.type(
            inputs_embeds.dtype
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        if per_layer_projection.shape != per_layer_inputs.shape:
            # per-layer inputs are sometimes padded with zeros, slice the relevant embeddings
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]

        return (
            per_layer_projection + per_layer_inputs
        ) * self.per_layer_input_scale.type(inputs_embeds.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        if per_layer_inputs is None and input_ids is not None:
            per_layer_inputs = self.get_per_layer_inputs(input_ids)

        per_layer_inputs = self.project_per_layer_inputs(
            hidden_states, per_layer_inputs
        )

        if positions.dim() == 1:
            positions = positions.unsqueeze(0)

        # Initialize RoPE embeddings
        position_embeddings_global = self.rotary_emb(hidden_states, positions)
        position_embeddings_local = self.rotary_emb_local(hidden_states, positions)

        # Expand hidden_states to support per-layer inputs
        target_magnitude = torch.mean(hidden_states**2, dim=-1, keepdim=True) ** 0.5
        epsilon_tensor = torch.tensor(torch.finfo(hidden_states.dtype).min)

        hidden_states_list = [hidden_states] * self.config.altup_num_inputs

        for i in range(1, self.config.altup_num_inputs):
            altup_proj, _ = self.altup_projections[i - 1](hidden_states_list[i])
            hidden_states_list[i] = altup_proj.type(hidden_states.dtype)
            new_magnitude = (
                torch.mean(hidden_states_list[i] ** 2, dim=-1, keepdim=True) ** 0.5
            )
            hidden_states_list[i] *= target_magnitude / torch.maximum(
                new_magnitude, epsilon_tensor
            )

        hidden_states = torch.stack(
            hidden_states_list, dim=0
        )  # [num_altup_inputs, batch, seq_len, hidden_size]

        for layer_idx, layer in enumerate(self.layers):
            per_layer_input = per_layer_inputs[:, :, layer_idx, :]
            hidden_states = layer(
                positions=positions,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                per_layer_input=per_layer_input,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                **kwargs,
            )

        # Per-layer inputs to single output
        target_magnitude = (
            torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        )
        for i in range(1, self.config.altup_num_inputs):
            altup_unemb_proj, _ = self.altup_unembed_projections[i - 1](
                hidden_states[i]
            )
            hidden_states[i] = altup_unemb_proj.type(hidden_states[0].dtype)
            new_magnitude = (
                torch.mean(hidden_states[i] ** 2, dim=-1, keepdim=True) ** 0.5
            )
            hidden_states[i] *= target_magnitude / torch.maximum(
                new_magnitude, epsilon_tensor
            )

        hidden_states = torch.mean(hidden_states, dim=0)
        hidden_states = self.norm(hidden_states)

        return hidden_states


class Gemma3nForCausalLM(PreTrainedModel):
    config_class = Gemma3nTextConfig

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config_class = Gemma3nTextConfig
    base_model_prefix = "language_model"

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        ".q_proj": (".qkv_proj", 0),
        ".k_proj": (".qkv_proj", 1),
        ".v_proj": (".qkv_proj", 2),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    packed_modules_mapping = {
        ".qkv_proj": [
            ".q_proj",
            ".k_proj",
            ".v_proj",
        ],
        ".gate_up_proj": [
            ".gate_proj",
            ".up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        ".qkv_proj",
        ".o_proj",
        ".gate_up_proj",
        ".down_proj",
    ]
    # Gemma does not apply LoRA to the embedding layer
    embedding_modules = {}
    embedding_padding_modules = []
    supports_lora = True

    def __init__(
        self,
        config: Gemma3nTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config
        self.model = Gemma3nTextModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config)

    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LogitsProcessor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            per_layer_inputs,
            **kwargs,
        )

        return self.logits_processor(
            input_ids, hidden_states, self.model.embed_tokens, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            name = name.replace("model.language_model.", "model.")
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    # Skip loading weights that are not in the model
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # lm_head is not used in vllm as it is tied with embed_token
                if "lm_head.weight" in name:
                    continue
                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if name not in params_dict:
                    # Skip loading weights that are not in the model
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


EntryClass = Gemma3nForCausalLM
AutoModel.register(Gemma3nTextConfig, Gemma3nForCausalLM, exist_ok=True)
