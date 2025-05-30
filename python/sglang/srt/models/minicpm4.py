# Source: https://github.com/huggingface/transformers/blob/v4.31-release/src/transformers/models/llama/modeling_llama.py
# Modifications are denoted by the symbol: [MODIFIED]


import math
from typing import Optional, Tuple

import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import LlamaConfig, PretrainedConfig

# [MODIFIED] Import from transformer library
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import LlamaModel

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Create a causal mask for bi-directional self-attention.

    Args:
        input_ids_shape (torch.Size): The shape of input_ids tensor, typically (batch_size, tgt_len).
        dtype (torch.dtype): The data type of the mask.
        device (torch.device): The device on which the mask will be placed.
        past_key_values_length (int, optional): The length of past key values. Default is 0.

    Returns:
        torch.Tensor: The causal mask tensor.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expand attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.

    Args:
        mask (torch.Tensor): The attention mask tensor of shape `[bsz, seq_len]`.
        dtype (torch.dtype): The data type of the mask.
        tgt_len (Optional[int], optional): The target sequence length. If None, it defaults to the source sequence length.

    Returns:
        torch.Tensor: The expanded mask tensor.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


import torch
import torch.nn as nn


class LlamaRMSNorm(nn.Module):
    """
    LlamaRMSNorm is equivalent to T5LayerNorm.

    Args:
        hidden_size (int): The size of the hidden states.
        eps (float, optional): A small value to prevent division by zero. Default is 1e-6.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Apply LlamaRMSNorm to the input hidden states.

        Args:
            hidden_states (torch.Tensor): Input hidden states.

        Returns:
            torch.Tensor: The normalized and scaled hidden states.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    """
    Llama Rotary Positional Embedding Module.

    Args:
        dim (int): The dimension of the embedding.
        max_position_embeddings (int, optional): The maximum position for embeddings. Default is 2048.
        base (int, optional): The base value for rotational encoding. Default is 10000.
        device (str, optional): The device on which the computation will be performed. Default is None.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """
        Set the cosine and sine cache for positional embeddings.

        Args:
            seq_len (int): The sequence length.
            device (str): The device on which the cache tensors will be stored.
            dtype: The data type of the cache tensors.
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x, seq_len=None):
        """
        Forward pass of the LlamaRotaryEmbedding module.

        Args:
            x (torch.Tensor): Input tensor of shape [bs, num_attention_heads, seq_len, head_size].
            seq_len (int): The sequence length. If greater than the cached length, the cache will be updated.

        Returns:
            tuple: A tuple containing two tensors, the cosine and sine embeddings, both of shape [1, 1, seq_len, dim].
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """
    LlamaRotaryEmbedding extended with linear scaling.

    This class adds linear scaling to LlamaRotaryEmbedding. Credits to the Reddit user /u/kaiokendev.

    Args:
        dim (int): The dimension of the embedding.
        max_position_embeddings (int, optional): The maximum number of position embeddings. Default is 2048.
        base (int, optional): The base value for the rotational embeddings. Default is 10000.
        device (str or torch.device, optional): The device where the embeddings should be stored. Default is None.
        scaling_factor (float, optional): The scaling factor for the embeddings. Default is 1.0.
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """
        Set the cosine and sine cache for the rotary embeddings.

        Args:
            seq_len (int): The sequence length.
            device (str or torch.device): The device where the cache should be stored.
            dtype: The data type for the cache.
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """
    LlamaRotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla.
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        """
        Initialize the LlamaDynamicNTKScalingRotaryEmbedding.

        Args:
            dim (int): The dimensionality of the embedding.
            max_position_embeddings (int, optional): Maximum number of position embeddings. Default is 2048.
            base (int, optional): Base value for scaling calculations. Default is 10000.
            device: The device to place tensors on. If None, uses the default device.
            scaling_factor (float, optional): Scaling factor for NTK scaling. Default is 1.0.
        """
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """
        Set the cached values for cosine and sine.

        Args:
            seq_len (int): The sequence length.
            device: The device to place tensors on.
            dtype: The data type of tensors.
        """
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


def rotate_half(x):
    """
    Rotates half the hidden dimensions of the input.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with half of its hidden dimensions rotated.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        cos (torch.Tensor): Cosine values.
        sin (torch.Tensor): Sine values.
        position_ids (torch.Tensor): Position IDs.

    Returns:
        torch.Tensor: Query and key tensors with rotary position embeddings applied.
    """
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    """
    LlamaMLP is a multi-layer perceptron module used in the Llama model.

    Args:
        config: The configuration for the MLP.

    Attributes:
        pretraining_tp (int): The pretraining time periods.
        hidden_size (int): The size of the hidden layer.
        intermediate_size (int): The size of the intermediate layer.
        gate_proj (nn.Linear): The linear projection for gating.
        up_proj (nn.Linear): The linear projection for the up projection.
        down_proj (nn.Linear): The linear projection for the down projection.
        act_fn: The activation function.

    """

    def __init__(self, config):
        super().__init__()
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key and value tensors n times along the specified dimension.

    Args:
        hidden_states (torch.Tensor): Input tensor with shape (batch, num_key_value_heads, seqlen, head_dim).
        n_rep (int): Number of times to repeat.

    Returns:
        torch.Tensor: Repeated tensor with shape (batch, num_key_value_heads * n_rep, seqlen, head_dim).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MiniCPMLongRoPE(LlamaRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        short_factor=None,
        long_factor=None,
        original_max_position_embeddings=None,
    ):
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        scale = max_position_embeddings / self.original_max_position_embeddings
        self.scaling_factor = math.sqrt(
            1 + math.log(scale) / math.log(self.original_max_position_embeddings)
        )
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(
                self.long_factor, dtype=torch.float32, device=device
            )
        else:
            ext_factors = torch.tensor(
                self.short_factor, dtype=torch.float32, device=device
            )

        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype),
        )
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached",
            emb.cos()[None, None, :, :].to(dtype) * self.scaling_factor,
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            emb.sin()[None, None, :, :].to(dtype) * self.scaling_factor,
            persistent=False,
        )


class Llama3RotaryEmbedding(nn.Module):
    """
    Llama Rotary Positional Embedding Module.

    Args:
        dim (int): The dimension of the embedding.
        max_position_embeddings (int, optional): The maximum position for embeddings. Default is 2048.
        device (str, optional): The device on which the computation will be performed. Default is None.
    """

    def __init__(
        self, dim, rope_config, max_position_embeddings, rope_theta, device=None
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = rope_theta
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )

        factor = rope_config["factor"]  # `8` in the original implementation
        low_freq_factor = rope_config[
            "low_freq_factor"
        ]  # `1` in the original implementation
        high_freq_factor = rope_config[
            "high_freq_factor"
        ]  # `4` in the original implementation
        old_context_len = rope_config[
            "original_max_position_embeddings"
        ]  # `8192` in the original implementation

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / inv_freq
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
        )
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            1 - smooth_factor
        ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        self.register_buffer("inv_freq", inv_freq_llama)

        self.attention_scaling = 1

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """
        Set the cosine and sine cache for positional embeddings.

        Args:
            seq_len (int): The sequence length.
            device (str): The device on which the cache tensors will be stored.
            dtype: The data type of the cache tensors.
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x, position_ids=None):
        """
        Forward pass of the LlamaRotaryEmbedding module.

        Args:
            x (torch.Tensor): Input tensor of shape [bs, num_attention_heads, seq_len, head_size].
            seq_len (int): The sequence length. If greater than the cached length, the cache will be updated.

        Returns:
            tuple: A tuple containing two tensors, the cosine and sine embeddings, both of shape [1, 1, seq_len, dim].
        """
        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaAttention(nn.Module):
    """
    LlamaAttention is a multi-headed attention module based on the 'Attention Is All You Need' paper.

    Args:
        config (LlamaConfig): Configuration for the attention module.

    Attributes:
        config (LlamaConfig): Configuration for the attention module.
        hidden_size (int): The size of the hidden layer.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        num_key_value_heads (int): The number of key-value attention heads.
        num_key_value_groups (int): The number of key-value groups.
        pretraining_tp (int): The pretraining time periods.
        max_position_embeddings (int): The maximum position embeddings.

    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
            )
        else:
            if (
                "rope_type" in self.config.rope_scaling
                and self.config.rope_scaling["rope_type"] == "llama3"
            ):
                self.rotary_emb = Llama3RotaryEmbedding(
                    self.head_dim,
                    self.config.rope_scaling,
                    self.config.max_position_embeddings,
                    self.config.rope_theta,
                )
            else:
                scaling_type = self.config.rope_scaling["rope_type"]
                # scaling_factor = self.config.rope_scaling["factor"]
                if scaling_type == "linear":
                    self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                        self.head_dim,
                        max_position_embeddings=self.max_position_embeddings,
                        scaling_factor=self.config.rope_scaling["factor"],
                    )
                elif scaling_type == "dynamic":
                    self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                        self.head_dim,
                        max_position_embeddings=self.max_position_embeddings,
                        scaling_factor=self.config.rope_scaling["factor"],
                    )
                elif scaling_type == "longrope":
                    self.rotary_emb = MiniCPMLongRoPE(
                        self.head_dim,
                        max_position_embeddings=self.max_position_embeddings,
                        short_factor=self.config.rope_scaling["short_factor"],
                        long_factor=self.config.rope_scaling["long_factor"],
                        base=self.config.rope_theta,
                        original_max_position_embeddings=self.config.rope_scaling[
                            "original_max_position_embeddings"
                        ],
                    )
                else:
                    raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        # print("position_ids: ", position_ids)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # [MODIFIED] Using KVCache mechanism for preallocated GPU memory optimization
        # past_key_value is utilized to leverage previously computed key and value states.
        # If past_key_value is available, reuse the states for k, v, and self_attention.
        if past_key_value is not None:
            key_states = past_key_value[0].cat(key_states, dim=2)
            value_states = past_key_value[1].cat(value_states, dim=2)
        # Reset past_key_value to avoid return past_key_value.
        past_key_value = None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


#
#
# class LlamaDecoderLayer(nn.Module):
#     """
#     LlamaDecoderLayer represents a single layer of the Llama decoder.
#
#     Args:
#         config (LlamaConfig): Configuration for the decoder layer.
#
#     Attributes:
#         hidden_size (int): The size of the hidden layer.
#         self_attn (LlamaAttention): Multi-headed self-attention module.
#         mlp (LlamaMLP): Multi-layer perceptron module.
#         input_layernorm (LlamaRMSNorm): Layer normalization for input.
#         post_attention_layernorm (LlamaRMSNorm): Layer normalization after self-attention.
#     """
#
#     def __init__(self, config: LlamaConfig):
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         self.self_attn = LlamaAttention(config=config)
#
#         self.mlp = LlamaMLP(config)
#         self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = LlamaRMSNorm(
#             config.hidden_size, eps=config.rms_norm_eps
#         )
#         self.scale_depth = config.scale_depth
#         self.num_hidden_layers = config.num_hidden_layers
#
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Tuple[torch.Tensor]] = None,
#         output_attentions: Optional[bool] = False,
#         use_cache: Optional[bool] = False,
#     ) -> Tuple[
#         torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
#     ]:
#         """
#         Forward pass for the LlamaDecoderLayer.
#
#         Args:
#             hidden_states (torch.FloatTensor): Input tensor of shape `(batch, seq_len, embed_dim)`.
#             attention_mask (torch.FloatTensor, optional): Attention mask of size
#                 `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
#             position_ids (torch.LongTensor, optional): Positional IDs tensor.
#             past_key_value (Tuple[torch.FloatTensor], optional): Cached past key and value projection states.
#             output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers.
#             use_cache (bool, optional): If set to `True`, `past_key_values` key-value states are returned and can be
#                 used to speed up decoding.
#
#         Returns:
#             Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]: Tuple containing:
#                 - hidden_states (torch.FloatTensor): Output tensor.
#                 - self_attn_weights (Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]): Self-attention weights if
#                   `output_attentions` is `True`.
#                 - present_key_value (Optional[Tuple[torch.FloatTensor]]): Cached key and value projection states if
#                   `use_cache` is `True`.
#         """
#
#         residual = hidden_states
#
#         hidden_states = self.input_layernorm(hidden_states)
#
#         # Self Attention
#         hidden_states, self_attn_weights, present_key_value = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_value=past_key_value,
#             output_attentions=output_attentions,
#             use_cache=use_cache,
#         )
#         hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
#
#         # Fully Connected
#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
#
#         outputs = (hidden_states,)
#
#         if output_attentions:
#             outputs += (self_attn_weights,)
#
#         if use_cache:
#             outputs += (present_key_value,)
#
#         return outputs
#
#
# class LlamaPreTrainedModel(PreTrainedModel):
#     config_class = LlamaConfig
#     base_model_prefix = "model"
#     supports_gradient_checkpointing = True
#     _no_split_modules = ["LlamaDecoderLayer"]
#     _skip_keys_device_placement = "past_key_values"
#
#     def _init_weights(self, module):
#         std = self.config.initializer_range
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#
#     def _set_gradient_checkpointing(self, module, value=False):
#         if isinstance(module, LlamaModel):
#             module.gradient_checkpointing = value
#
#
# class LlamaModel(LlamaPreTrainedModel):
#     """
#     Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
#
#     Args:
#         config: LlamaConfig
#     """
#
#     def __init__(self, config: LlamaConfig):
#         super().__init__(config)
#         self.padding_idx = config.pad_token_id
#         self.vocab_size = config.vocab_size
#
#         self.embed_tokens = nn.Embedding(
#             config.vocab_size, config.hidden_size, self.padding_idx
#         )
#         self.layers = nn.ModuleList(
#             [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
#         )
#         self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#
#         self.gradient_checkpointing = False
#         # Initialize weights and apply final processing
#         self.post_init()
#
#     def get_input_embeddings(self):
#         return self.embed_tokens
#
#     def set_input_embeddings(self, value):
#         self.embed_tokens = value
#
#     # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
#     def _prepare_decoder_attention_mask(
#         self, attention_mask, input_shape, inputs_embeds, past_key_values_length
#     ):
#         # create causal mask
#         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#         combined_attention_mask = None
#         if input_shape[-1] > 1:
#             combined_attention_mask = _make_causal_mask(
#                 input_shape,
#                 # inputs_embeds.dtype,
#                 torch.float32,  # [MODIFIED] force to cast to float32
#                 device=inputs_embeds.device,
#                 past_key_values_length=past_key_values_length,
#             )
#
#         if attention_mask is not None:
#             # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#             expanded_attn_mask = _expand_mask(
#                 attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
#             ).to(inputs_embeds.device)
#             combined_attention_mask = (
#                 expanded_attn_mask
#                 if combined_attention_mask is None
#                 else expanded_attn_mask + combined_attention_mask
#             )
#
#         if hasattr(self, "tree_mask") and self.tree_mask is not None:
#             tree_mask = self.tree_mask
#             tree_len = tree_mask.size(-1)
#             combined_attention_mask[:, :, -tree_len:, -tree_len:][
#                 tree_mask == 0
#                 ] = combined_attention_mask.min()
#
#         return combined_attention_mask
#
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values=None,  # [MODIFIED] past_key_value is KVCache class
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPast]:
#         output_attentions = (
#             output_attentions
#             if output_attentions is not None
#             else self.config.output_attentions
#         )
#         output_hidden_states = (
#             output_hidden_states
#             if output_hidden_states is not None
#             else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#
#         return_dict = (
#             return_dict if return_dict is not None else self.config.use_return_dict
#         )
#
#         # retrieve input_ids and inputs_embeds
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError(
#                 "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
#             )
#         elif input_ids is not None:
#             batch_size, seq_length = input_ids.shape
#         elif inputs_embeds is not None:
#             batch_size, seq_length, _ = inputs_embeds.shape
#         else:
#             raise ValueError(
#                 "You have to specify either decoder_input_ids or decoder_inputs_embeds"
#             )
#
#         seq_length_with_past = seq_length
#         past_key_values_length = 0
#
#         if past_key_values is not None:
#             past_key_values_length = past_key_values[0][0].shape[2]
#             seq_length_with_past = seq_length_with_past + past_key_values_length
#
#         if position_ids is None:
#             device = input_ids.device if input_ids is not None else inputs_embeds.device
#             position_ids = torch.arange(
#                 past_key_values_length,
#                 seq_length + past_key_values_length,
#                 dtype=torch.long,
#                 device=device,
#             )
#             position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
#         else:
#             position_ids = position_ids.view(-1, seq_length).long()
#
#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb
#         # embed positions
#         if attention_mask is None:
#             attention_mask = torch.ones(
#                 (batch_size, seq_length_with_past),
#                 dtype=torch.bool,
#                 device=inputs_embeds.device,
#             )
#         attention_mask = self._prepare_decoder_attention_mask(
#             attention_mask,
#             (batch_size, seq_length),
#             inputs_embeds,
#             past_key_values_length,
#         )
#
#         hidden_states = inputs_embeds
#
#         if self.gradient_checkpointing and self.training:
#             if use_cache:
#                 logger.warning_once(
#                     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
#                 )
#                 use_cache = False
#
#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = () if use_cache else None
#
#         for idx, decoder_layer in enumerate(self.layers):
#             # if idx==16:
#             #     print(idx)
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)
#
#             past_key_value = (
#                 past_key_values[idx] if past_key_values is not None else None
#             )
#
#             if self.gradient_checkpointing and self.training:
#
#                 def create_custom_forward(module):
#                     def custom_forward(*inputs):
#                         # None for past_key_value
#                         return module(*inputs, output_attentions, None)
#
#                     return custom_forward
#
#                 layer_outputs = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(decoder_layer),
#                     hidden_states,
#                     attention_mask,
#                     position_ids,
#                     None,
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=attention_mask,
#                     position_ids=position_ids,
#                     past_key_value=past_key_value,
#                     output_attentions=output_attentions,
#                     use_cache=use_cache,
#                 )
#
#             hidden_states = layer_outputs[0]
#
#             if use_cache:
#                 next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
#
#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)
#
#         hidden_states = self.norm(hidden_states)
#         ### scale before lm_head
#         hidden_states = hidden_states / (self.config.hidden_size / self.config.dim_model_base)
#
#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)
#
#         next_cache = next_decoder_cache if use_cache else None
#         if not return_dict:
#             return tuple(
#                 v
#                 for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
#                 if v is not None
#             )
#         return BaseModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )
#
#
# class LlamaForSequenceClassification(LlamaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.model = LlamaModel(config)
#         self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
#
#         # Initialize weights and apply final processing
#         self.post_init()
#
#     def get_input_embeddings(self):
#         return self.model.embed_tokens
#
#     def set_input_embeddings(self, value):
#         self.model.embed_tokens = value
#
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = (
#             return_dict if return_dict is not None else self.config.use_return_dict
#         )
#
#         transformer_outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         hidden_states = transformer_outputs[0]
#         logits = self.score(hidden_states)
#
#         if input_ids is not None:
#             batch_size = input_ids.shape[0]
#         else:
#             batch_size = inputs_embeds.shape[0]
#
#         if self.config.pad_token_id is None and batch_size != 1:
#             raise ValueError(
#                 "Cannot handle batch sizes > 1 if no padding token is defined."
#             )
#         if self.config.pad_token_id is None:
#             sequence_lengths = -1
#         else:
#             if input_ids is not None:
#                 sequence_lengths = (
#                     torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
#                 ).to(logits.device)
#             else:
#                 sequence_lengths = -1
#
#         pooled_logits = logits[
#             torch.arange(batch_size, device=logits.device), sequence_lengths
#         ]
#
#         loss = None
#         if labels is not None:
#             labels = labels.to(logits.device)
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (
#                     labels.dtype == torch.long or labels.dtype == torch.int
#                 ):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"
#
#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(pooled_logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(
#                     pooled_logits.view(-1, self.num_labels), labels.view(-1)
#                 )
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(pooled_logits, labels)
#         if not return_dict:
#             output = (pooled_logits,) + transformer_outputs[1:]
#             return ((loss,) + output) if loss is not None else output
#
#         return SequenceClassifierOutputWithPast(
#             loss=loss,
#             logits=pooled_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )
#


class MiniCPM4ForCausalLM(PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ):
        if input_embeds is not None:
            input_embeds = input_embeds * self.config.scale_emb
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        hidden_states = hidden_states / self.scale_width
        if self.config.tie_word_embeddings:
            lm_head = self.model.embed_tokens
        else:
            lm_head = self.lm_head
        return self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


EntryClass = [MiniCPM4ForCausalLM]
