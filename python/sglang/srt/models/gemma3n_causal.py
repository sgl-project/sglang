from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, Gemma3nTextConfig, PretrainedConfig, PreTrainedModel

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import GeluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.gemma3_causal import Gemma3TextScaledWordEmbedding
from sglang.srt.utils import add_prefix, make_layers


# Aligned with HF's implementation, using sliding window inclusive with the last token
# SGLang assumes exclusive
def get_attention_sliding_window_size(config):
    return config.sliding_window - 1


class Gemma3nRMSNorm(RMSNorm):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        with_scale: bool = True,
    ) -> None:
        super().__init__(dim, eps=eps)
        if not with_scale:
            del self.weight
            self.register_buffer(
                "weight",
                torch.ones(dim, dtype=torch.get_default_dtype()),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.contiguous().reshape(-1, original_shape[-1])
        x_2d = super().forward(x_2d)
        x = x_2d.reshape(original_shape)
        return x


class Gemma3nTextScaledWordEmbedding(Gemma3TextScaledWordEmbedding):
    pass


class Gemma3nTextMLP(nn.Module):
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
        self.register_buffer(
            "target_sparsity_tensor",
            torch.tensor(self.activation_sparsity, dtype=torch.float32),
            persistent=False,
        )  # moved from _gaussian_topk for cuda graph

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)

        # Split gate and up projections
        gate_proj, up_proj = gate_up.chunk(2, dim=-1)

        # Apply activation sparsity if needed
        if self.activation_sparsity > 0.0:
            gate_proj = self._gaussian_topk(gate_proj)

        gate_up = torch.cat([gate_proj, up_proj], dim=-1)

        # Apply GELU activation to gate projection and multiply with up projection
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier = normal_dist.icdf(self.target_sparsity_tensor)
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [num_tokens, hidden_size]
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
        )

        self.register_buffer(
            "router_input_scale",
            torch.tensor(config.hidden_size**-1.0),
            persistent=False,
        )

    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
        # x  : [num_tokens, hidden_size]
        router_inputs = self.router_norm(x) * self.router_input_scale.to(
            self.router_norm.weight.dtype
        )
        # router_inputs : [num_tokens, hidden_size]
        routed, _ = self.modality_router(router_inputs)

        # routed : [num_tokens, altup_num_inputs]
        return torch.tanh(routed.float()).type_as(routed)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predicts the output of a layer using a trainable map.
        hidden_states: [num_altup_inputs, num_tokens, hidden_size]
        """
        modalities = self.compute_router_modalities(
            hidden_states[self.config.altup_active_idx]
        )  # (n_tokens, altup_num_inputs)
        # TODO: CHECK DO WE NEED THIS: self.prediction_coefs.float()  # Force computation in float32, in-place operation

        if self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight.data.clamp_(
                -self.config.altup_coef_clip, self.config.altup_coef_clip
            )

        all_coefs, _ = self.prediction_coefs(
            modalities
        )  # (n_tokens, altup_num_inputs) -> (n_tokens, altup_num_inputs**2)

        all_coefs = all_coefs.reshape(
            *modalities.shape[:-1],
            self.config.altup_num_inputs,
            self.config.altup_num_inputs,
        ).permute(0, 2, 1)

        # permute hidden_states from [num_altup_inputs, num_tokens, hidden_size] to [num_tokens, hidden_size, altup_num_inputs]
        predictions = torch.matmul(hidden_states.permute(1, 2, 0), all_coefs)
        predictions = predictions.permute(2, 0, 1)  # undo the permute
        predictions += hidden_states  # add the original input
        return predictions.contiguous().type_as(
            hidden_states
        )  # [num_altup_inputs, num_tokens, hidden_size]

    def correct(
        self, predictions: torch.Tensor, activated: torch.Tensor
    ) -> torch.Tensor:
        """Corrects the predictions relative to the activated inputs."""
        # prediction : [num_altup_inputs, num_tokens, hidden_size]
        # activated  : [num_tokens, hidden_size]
        modalities = self.compute_router_modalities(
            activated
        )  # [num_tokens, altup_num_inputs]
        innovation = (
            activated - predictions[self.config.altup_active_idx]
        )  # [num_tokens, hidden_size]
        innovation = innovation.repeat(
            self.config.altup_num_inputs, 1, 1
        )  # (self.config.altup_num_inputs, num_tokens, hidden_size)

        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight.data.clamp_(
                -self.config.altup_coef_clip, self.config.altup_coef_clip
            )

        all_coefs, _ = self.correction_coefs(
            modalities
        )  # [num_tokens, altup_num_inputs]
        all_coefs = (all_coefs + 1.0).permute(1, 0).unsqueeze(-1)
        # # [num_tokens, altup_num_inputs, 1]

        corrected = torch.mul(innovation, all_coefs)
        corrected += predictions
        return corrected.contiguous().type_as(activated)

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        """Scales the provided 3D tensor."""
        return corrected * self.correct_output_scale.to(corrected.dtype)

    def forward(
        self, hidden_states: torch.Tensor, activated: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts, correct, and optionally scales the output of a layer using trainable maps.

        hidden_states: [num_altup_inputs, num_tokens, hidden_size]
        """

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
        # self.scaling = config.query_rescale_scalar / config.query_pre_attn_scalar
        self.scaling = 1.0

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

        if self.is_sliding:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_local_base_freq,
                rope_scaling={"rope_type": "default"},
            )
        else:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                rope_scaling=config.rope_scaling,
            )

        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=(
                layer_id if not self.is_kv_shared_layer else self.kv_shared_layer_index
            ),
            logit_cap=0.0,
            sliding_window_size=self.sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        # Gemma3n adds normalization for q, k, v
        self.q_norm = Gemma3nRMSNorm(
            dim=config.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Gemma3nRMSNorm(
            dim=config.head_dim,
            eps=config.rms_norm_eps,
        )
        self.v_norm = Gemma3nRMSNorm(
            dim=config.head_dim,
            eps=config.rms_norm_eps,
            with_scale=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: Tuple[torch.Tensor, torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:

        qkv, _ = self.qkv_proj(hidden_states)
        # TODO: for first 20 layers, we use QKVParallelLinear
        #       for others, we only calc Q.
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Apply normalization to q, k, v
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)

        # Check if we should use shared KV cache
        if self.is_kv_shared_layer and self.kv_shared_layer_index is not None:
            # For KV shared layers, we skip K/V computation and normalization
            # The RadixAttention will handle retrieving shared KV from cache
            k = None
            v = None
        else:
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = self.k_norm(k)

            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(v)

        # Flatten back for rotary embedding
        q = q.flatten(-2, -1)

        # Apply rotary embedding
        if k is not None:
            k = k.flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)
            # Reshape k back to head format for attention
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        else:
            # For shared KV layers, create a dummy key for rotary embedding and discard it
            dummy_k = torch.zeros_like(
                q[:, : self.kv_size]
            )  # Create dummy key with same shape as needed
            q, _ = self.rotary_emb(positions, q, dummy_k)

        # Reshape q back to head format for attention
        q = q.unflatten(-1, (self.num_heads, self.head_dim))

        attn_output = self.attn(
            q,
            k,
            v,
            forward_batch=forward_batch,
            save_kv_cache=not self.is_kv_shared_layer,
        )

        output, _ = self.o_proj(attn_output)
        return output


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
        self.config = config

        self.self_attn = Gemma3nAttention(
            layer_id=layer_id,
            config=config,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        intermediate_size = config.intermediate_size[layer_id]
        activation_sparsity = config.activation_sparsity_pattern[layer_id]
        self.mlp = Gemma3nTextMLP(
            hidden_size=self.hidden_size,
            intermediate_size=intermediate_size,
            hidden_activation=config.hidden_activation,
            activation_sparsity=activation_sparsity,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = Gemma3nRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3nRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3nRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3nRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
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
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.is_sliding = self.self_attn.is_sliding

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        predictions = self.altup.predict(
            hidden_states
        )  # [num_altup_inputs, num_tokens, hidden_size]
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(
            active_prediction_normed
        )  # laurel_output: [num_tokens, hidden_size]
        # active_prediction: [num_tokens, hidden_size]

        attn = self.self_attn(
            positions=positions,
            hidden_states=active_prediction_normed,
            forward_batch=forward_batch,
            **kwargs,
        )
        attn = self.post_attention_layernorm(attn)  # [num_tokens, hidden_size]

        attn_gated = active_prediction + attn  # [num_tokens, hidden_size]
        attn_laurel = (attn_gated + laurel_output) / torch.sqrt(torch.tensor(2.0))

        attn_norm = self.pre_feedforward_layernorm(
            attn_laurel
        )  # [num_tokens, hidden_size]
        attn_ffw = self.mlp(attn_norm)  # [num_tokens, hidden_size]
        attn_ffw_norm = self.post_feedforward_layernorm(
            attn_ffw
        )  # [num_tokens, hidden_size]
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm  # [num_tokens, hidden_size]
        corrected_predictions = self.altup.correct(
            predictions, attn_ffw_laurel_gated
        )  # prediction : [num_altup_inputs, num_tokens, hidden_size]
        # attn_ffw_laurel_gated: [num_tokens, hidden_size]
        first_prediction = corrected_predictions[self.config.altup_active_idx]

        if self.config.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        # per_layer_input_gate
        first_prediction = first_prediction.to(self.per_layer_input_gate.weight.dtype)
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
        self.padding_idx = config.pad_token_id

        # Gemma3n downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        self.embed_tokens = Gemma3nTextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,
        )

        self.norm = Gemma3nRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

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
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            self.padding_idx,
            embed_scale=self.config.hidden_size_per_layer_input**0.5,
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

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        embeddings = self.embed_tokens_per_layer(input_ids)
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
        if (input_ids is None) ^ (input_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if input_ids is not None:
            input_embeds = self.embed_tokens(input_ids)
            per_layer_inputs = self.get_per_layer_inputs(input_ids)

        per_layer_inputs = self.project_per_layer_inputs(input_embeds, per_layer_inputs)

        if positions.dim() == 1:
            positions = positions.unsqueeze(0)

        # Expand hidden_states to support per-layer inputs
        target_magnitude = torch.mean(input_embeds**2, dim=-1, keepdim=True) ** 0.5
        epsilon_tensor = torch.tensor(torch.finfo(input_embeds.dtype).min)

        # embed positions
        hidden_states_0 = input_embeds
        temp_hidden_states = [hidden_states_0]

        for i in range(1, self.config.altup_num_inputs):
            altup_proj, _ = self.altup_projections[i - 1](hidden_states_0)
            current_hidden_state = altup_proj.type(hidden_states_0.dtype)
            new_magnitude = (
                torch.mean(current_hidden_state**2, dim=-1, keepdim=True) ** 0.5
            )
            current_hidden_state = current_hidden_state * (
                target_magnitude / torch.maximum(new_magnitude, epsilon_tensor)
            )
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(
            temp_hidden_states, dim=0
        )  # [num_altup_inputs, n_tokens, hidden_size]

        for layer_idx, layer in enumerate(self.layers):
            per_layer_input = per_layer_inputs[:, layer_idx, :]
            hidden_states = layer(
                positions=positions,
                per_layer_input=per_layer_input,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                **kwargs,
            )

        # Per-layer inputs to single output
        target_magnitude = (
            torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        )

        temp_hidden_states = [hidden_states[0]]

        for i in range(1, self.config.altup_num_inputs):
            # altup_unembed_projections adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_unemb_proj, _ = self.altup_unembed_projections[i - 1](
                hidden_states[i]
            )
            current_hidden_state = altup_unemb_proj.type(hidden_states_0.dtype)
            new_magnitude = (
                torch.mean(current_hidden_state**2, dim=-1, keepdim=True) ** 0.5
            )
            current_hidden_state = current_hidden_state * (
                target_magnitude / torch.maximum(new_magnitude, epsilon_tensor)
            )
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states)
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
