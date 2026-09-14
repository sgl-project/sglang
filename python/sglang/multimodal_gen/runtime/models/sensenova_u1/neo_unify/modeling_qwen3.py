# Modified for SGLang; see this directory's README.md for upstream source.

import copy
from typing import Callable, Optional, Union

import torch
import torch._dynamo
from torch import nn
from transformers import Qwen3Config
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg

from .transformers_compat import (
    causal_mask_kwargs,
    model_input_compat,
    tied_weights_keys,
)

try:
    from flash_attn import flash_attn_func  # type: ignore

    _HAS_FLASH_ATTN = True
except ImportError:  # pragma: no cover - exercised only in CPU-only / no-flash envs
    flash_attn_func = None  # type: ignore
    _HAS_FLASH_ATTN = False


# Attention backend dispatch.
#
# Set via :func:`set_attn_backend`. Three modes are accepted:
#   * ``"auto"``  - use flash-attn if available, otherwise SDPA (default).
#   * ``"flash"`` - force flash-attn; raise if ``flash_attn`` is not installed.
#   * ``"sdpa"``  - force torch SDPA (useful for reproducibility tests and
#                    debugging, even when flash-attn is available).
_VALID_ATTN_BACKENDS = ("auto", "flash", "sdpa")
_ATTN_BACKEND: str = "auto"


def set_attn_backend(backend: str) -> str:
    """Choose the attention kernel used by the Qwen3 layers at runtime.

    Returns the backend string that was set. Raises ``ValueError`` for an
    unknown name and ``RuntimeError`` if ``flash`` is requested but the
    ``flash_attn`` package isn't importable.
    """
    global _ATTN_BACKEND
    backend = backend.lower()
    if backend not in _VALID_ATTN_BACKENDS:
        raise ValueError(
            f"Unknown attention backend {backend!r}. "
            f"Expected one of {_VALID_ATTN_BACKENDS}."
        )
    if backend == "flash" and not _HAS_FLASH_ATTN:
        raise RuntimeError(
            "Requested attn_backend='flash' but `flash_attn` is not installed. "
            "Install it (e.g. `uv pip install <flash_attn-*.whl>`) or use "
            "'auto' / 'sdpa'."
        )
    _ATTN_BACKEND = backend
    return _ATTN_BACKEND


def get_attn_backend() -> str:
    """Return the currently active attention backend name."""
    return _ATTN_BACKEND


def effective_attn_backend() -> str:
    """Resolve ``'auto'`` to the kernel that will actually run."""
    if _ATTN_BACKEND != "auto":
        return _ATTN_BACKEND
    return "flash" if _HAS_FLASH_ATTN else "sdpa"


def _sdpa_attn_func(
    q, k, v, dropout_p: float = 0.0, softmax_scale=None, causal: bool = False
):
    """Drop-in SDPA fallback for ``flash_attn_func``.

    ``flash_attn_func`` expects q/k/v in layout ``[B, S, H, D]`` and returns
    ``[B, S_q, H_q, D]``. ``torch.nn.functional.scaled_dot_product_attention``
    expects ``[B, H, S, D]``; we transpose in and out.

    ``flash_attn_func`` natively handles Grouped-Query Attention (GQA) where
    ``H_q > H_kv``. Plain ``scaled_dot_product_attention`` only supports that
    via the ``enable_gqa=True`` kwarg (torch >= 2.5). For broader compatibility
    we just materialize the repeat manually when needed.
    """
    q_bhsd = q.transpose(1, 2)
    k_bhsd = k.transpose(1, 2)
    v_bhsd = v.transpose(1, 2)

    h_q = q_bhsd.shape[1]
    h_kv = k_bhsd.shape[1]
    if h_q != h_kv:
        if h_q % h_kv != 0:
            raise ValueError(
                f"Cannot broadcast key/value heads ({h_kv}) to query heads ({h_q}): not divisible."
            )
        n_rep = h_q // h_kv
        k_bhsd = k_bhsd.repeat_interleave(n_rep, dim=1)
        v_bhsd = v_bhsd.repeat_interleave(n_rep, dim=1)

    # SDPA does not support an explicit `scale` argument on older torch
    # versions; fall back to the manual path in that case.
    try:
        out = torch.nn.functional.scaled_dot_product_attention(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )
    except TypeError:
        if softmax_scale is not None:
            q_bhsd = q_bhsd * softmax_scale
            out = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                dropout_p=dropout_p,
                is_causal=causal,
            )
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                dropout_p=dropout_p,
                is_causal=causal,
            )
    return out.transpose(1, 2).contiguous()


def _flash_or_sdpa(
    q, k, v, dropout_p: float = 0.0, softmax_scale=None, causal: bool = False
):
    backend = effective_attn_backend()
    # flash-attn ships CUDA kernels only. On XPU / CPU we transparently fall
    # back to SDPA even if the user asked for ``flash`` — the alternative
    # (crashing on first forward) is worse, and ``set_attn_backend('flash')``
    # already guarded against the "package missing" case.
    if backend == "flash" and q.device.type == "cuda":
        return flash_attn_func(
            q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal
        )
    return _sdpa_attn_func(
        q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal
    )


def create_block_causal_mask(index: torch.Tensor):
    """
    index: (L)
    return: (1, 1, L, L) block-wise causal attention mask
    """
    L = index.size(0)
    idx_i = index.unsqueeze(1).expand(L, L)
    idx_j = index.unsqueeze(0).expand(L, L)

    arange = torch.arange(L, device=index.device)
    mask = (idx_j == idx_i) | (arange.unsqueeze(0) <= arange.unsqueeze(1))

    return torch.where(
        mask[None, None, :, :] > 0, torch.tensor(0.0), torch.tensor(float("-inf"))
    )


def visualize_mask(mask: torch.Tensor, i: int = 0, j: int = 12):
    """
    mask: (1,1, L, L)
    """
    submask = torch.where(mask[0, 0, :, :] == 0, torch.tensor(1.0), torch.tensor(0.0))
    submask = mask[i:j, i:j].int().cpu().numpy()
    for row in submask:
        print(" ".join(map(str, row)))


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def _compute_default_rope_parameters(config, device=None, **_kwargs):
    """Default RoPE frequencies, inlined to avoid breakage across transformers versions.

    transformers <=4.x exposes this as ``ROPE_INIT_FUNCTIONS["default"]``, but
    5.x dropped the ``"default"`` key from that table. Having a local copy keeps
    ``Qwen3RotaryEmbedding`` working on both.
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, attention_factor


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    compute_default_rope_parameters = staticmethod(_compute_default_rope_parameters)

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        if self.rope_type == "default" or self.rope_type is None:
            base_rope_init_fn = _compute_default_rope_parameters
        else:
            base_rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        def _rope_init_fn_keep_freq_range(cfg: Qwen3Config, dev=None):
            inv_freq, attention_scaling = base_rope_init_fn(cfg, dev)

            cfg2 = copy.deepcopy(cfg)
            head_dim = getattr(cfg2, "head_dim", None)
            if head_dim is None:
                head_dim = cfg2.hidden_size // cfg2.num_attention_heads
                setattr(cfg2, "head_dim", head_dim)
            cfg2.head_dim = int(head_dim) * 2

            inv_freq_full, _ = base_rope_init_fn(cfg2, dev)
            inv_freq = inv_freq_full[::2]

            return inv_freq, attention_scaling

        self.rope_init_fn = _rope_init_fn_keep_freq_range

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.q_proj_mot_gen = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj_mot_gen = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj_mot_gen = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.o_proj_mot_gen = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = Qwen3RMSNorm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.q_norm_mot_gen = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.q_norm_hw = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.q_norm_hw_mot_gen = Qwen3RMSNorm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )

        self.k_norm = Qwen3RMSNorm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        self.k_norm_mot_gen = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm_hw = Qwen3RMSNorm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        self.k_norm_hw_mot_gen = Qwen3RMSNorm(
            self.head_dim // 2, eps=config.rms_norm_eps
        )

        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

        t_config = copy.deepcopy(config)
        t_config.head_dim = config.head_dim // 2
        self.rotary_emb = Qwen3RotaryEmbedding(config=t_config)

        hw_config = copy.deepcopy(config)
        hw_config.head_dim = config.head_dim // 4
        hw_config.rope_theta = config.rope_theta_hw
        if isinstance(getattr(hw_config, "rope_parameters", None), dict):
            hw_config.rope_parameters = {
                **hw_config.rope_parameters,
                "rope_theta": config.rope_theta_hw,
            }
        hw_config.max_position_embeddings = config.max_position_embeddings_hw
        self.rotary_emb_hw = Qwen3RotaryEmbedding(config=hw_config)

    def forward_und(
        self,
        hidden_states: torch.Tensor,
        indexes: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert self.config._attn_implementation == "eager"
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
        query_states_t = self.q_norm(query_states_t).transpose(1, 2)
        query_states_hw = self.q_norm_hw(query_states_hw).transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
        key_states_t = self.k_norm(key_states_t).transpose(1, 2)
        key_states_hw = self.k_norm_hw(key_states_hw).transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        query_states_t, key_states_t = apply_rotary_pos_emb(
            query_states_t, key_states_t, cos_t, sin_t
        )

        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        query_states_h, key_states_h = apply_rotary_pos_emb(
            query_states_h, key_states_h, cos_h, sin_h
        )

        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        query_states_w, key_states_w = apply_rotary_pos_emb(
            query_states_w, key_states_w, cos_w, sin_w
        )

        query_states = torch.cat(
            [query_states_t, query_states_h, query_states_w], dim=-1
        )
        key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            update_cache = kwargs.get("update_cache", True)
            if update_cache:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs=None
                )
            else:
                # only use the past key values but do not append the current one
                layer = past_key_values.layers[self.layer_idx]
                past_k, past_v = layer.keys, layer.values

                if past_k is not None:
                    key_states = torch.cat(
                        [past_k, key_states], dim=2
                    )  # concat on seq_len
                    value_states = torch.cat([past_v, value_states], dim=2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    # def forward_gen(
    #     self,
    #     hidden_states: torch.Tensor,
    #     indexes: Optional[torch.LongTensor],
    #     attention_mask: Optional[torch.Tensor],
    #     past_key_values: Optional[Cache] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     **kwargs: Unpack[FlashAttentionKwargs],
    # ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    #     assert self.config._attn_implementation == "eager"
    #     input_shape = hidden_states.shape[:-1]
    #     hidden_shape = (*input_shape, -1, self.head_dim)

    #     query_states = self.q_proj_mot_gen(hidden_states).view(hidden_shape)
    #     query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
    #     query_states_t = self.q_norm_mot_gen(query_states_t).transpose(1, 2)
    #     query_states_hw = self.q_norm_hw_mot_gen(query_states_hw).transpose(1, 2)
    #     query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

    #     key_states = self.k_proj_mot_gen(hidden_states).view(hidden_shape)
    #     key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
    #     key_states_t = self.k_norm_mot_gen(key_states_t).transpose(1, 2)
    #     key_states_hw = self.k_norm_hw_mot_gen(key_states_hw).transpose(1, 2)
    #     key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

    #     value_states = self.v_proj_mot_gen(hidden_states).view(hidden_shape).transpose(1, 2)

    #     cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
    #     query_states_t, key_states_t = apply_rotary_pos_emb(query_states_t, key_states_t, cos_t, sin_t)

    #     cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
    #     query_states_h, key_states_h = apply_rotary_pos_emb(query_states_h, key_states_h, cos_h, sin_h)

    #     cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
    #     query_states_w, key_states_w = apply_rotary_pos_emb(query_states_w, key_states_w, cos_w, sin_w)

    #     query_states = torch.cat([query_states_t, query_states_h, query_states_w], dim=-1)
    #     key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)

    #     if past_key_values is not None:
    #         # sin and cos are specific to RoPE models; cache_position needed for the static cache
    #         # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #         # key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
    #         update_cache = kwargs.get("update_cache", True)
    #         if update_cache:
    #             key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs=None)
    #         else:
    #             # only use the past key values but do not append the current one
    #             layer = past_key_values.layers[self.layer_idx]
    #             past_k, past_v = layer.keys, layer.values

    #             if past_k is not None:
    #                 key_states   = torch.cat([past_k, key_states], dim=2)   # concat on seq_len
    #                 value_states = torch.cat([past_v, value_states], dim=2)

    #     attention_interface: Callable = eager_attention_forward
    #     if self.config._attn_implementation != "eager":
    #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    #     attn_output, attn_weights = attention_interface(
    #         self,
    #         query_states,
    #         key_states,
    #         value_states,
    #         attention_mask,
    #         dropout=0.0 if not self.training else self.attention_dropout,
    #         scaling=self.scaling,
    #         sliding_window=self.sliding_window,  # diff with Llama
    #         **kwargs,
    #     )

    #     attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    #     attn_output = self.o_proj_mot_gen(attn_output)
    #     return attn_output, attn_weights

    def forward_gen(
        self,
        hidden_states: torch.Tensor,
        indexes: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # -----------------------------
        # Build q / k / v for current tokens
        # Internal layout before flash:
        #   q/k/v: [B, H, S, D]
        # Flash layout:
        #   q/k/v: [B, S, H, D]
        # -----------------------------
        query_states = self.q_proj_mot_gen(hidden_states).view(hidden_shape)
        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
        query_states_t = self.q_norm_mot_gen(query_states_t).transpose(
            1, 2
        )  # [B,H,S,D/2]
        query_states_hw = self.q_norm_hw_mot_gen(query_states_hw).transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states = self.k_proj_mot_gen(hidden_states).view(hidden_shape)
        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
        key_states_t = self.k_norm_mot_gen(key_states_t).transpose(1, 2)  # [B,H,S,D/2]
        key_states_hw = self.k_norm_hw_mot_gen(key_states_hw).transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        value_states = (
            self.v_proj_mot_gen(hidden_states).view(hidden_shape).transpose(1, 2)
        )  # [B,H,S,D]

        # RoPE
        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        query_states_t, key_states_t = apply_rotary_pos_emb(
            query_states_t, key_states_t, cos_t, sin_t
        )

        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        query_states_h, key_states_h = apply_rotary_pos_emb(
            query_states_h, key_states_h, cos_h, sin_h
        )

        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        query_states_w, key_states_w = apply_rotary_pos_emb(
            query_states_w, key_states_w, cos_w, sin_w
        )

        # concat along head_dim
        # query/key current layout: [B, H, S, D]
        query_states = torch.cat(
            [query_states_t, query_states_h, query_states_w], dim=-1
        )
        key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)

        update_cache = kwargs.get("update_cache", True)

        # ------------------------------------------------------------------
        # Flash path:
        # Only use when there is no explicit dense mask.
        # This is exactly the t2i denoising use case:
        #   current image tokens attend to [prefix + current image tokens]
        #   fully bidirectional inside current block => causal=False
        # ------------------------------------------------------------------
        if attention_mask is None:
            # Convert current q/k/v to flash layout [B, S, H, D]
            q = query_states.transpose(1, 2).contiguous()
            k_cur = key_states.transpose(1, 2).contiguous()
            v_cur = value_states.transpose(1, 2).contiguous()

            if past_key_values is not None:
                if update_cache:
                    # Rare path, keep compatibility.
                    # past_key_values.update expects [B,H,S,D]
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, self.layer_idx, cache_kwargs=None
                    )
                    k = key_states.transpose(1, 2).contiguous()
                    v = value_states.transpose(1, 2).contiguous()
                else:
                    # Optimized path:
                    # use preallocated flash_k_cache / flash_v_cache
                    layer = past_key_values.layers[self.layer_idx]

                    if (
                        hasattr(layer, "flash_k_cache")
                        and layer.flash_k_cache is not None
                        and hasattr(layer, "flash_v_cache")
                        and layer.flash_v_cache is not None
                    ):
                        prefix_len = layer.flash_prefix_len
                        cur_len = k_cur.shape[1]

                        # overwrite current segment in-place
                        layer.flash_k_cache[:, prefix_len : prefix_len + cur_len].copy_(
                            k_cur
                        )
                        layer.flash_v_cache[:, prefix_len : prefix_len + cur_len].copy_(
                            v_cur
                        )

                        k = layer.flash_k_cache[:, : prefix_len + cur_len]
                        v = layer.flash_v_cache[:, : prefix_len + cur_len]
                    else:
                        # fallback if user forgot to prepare flash cache
                        layer = past_key_values.layers[self.layer_idx]
                        past_k, past_v = layer.keys, layer.values

                        if past_k is not None:
                            past_k = past_k.transpose(1, 2).contiguous()
                            past_v = past_v.transpose(1, 2).contiguous()
                            k = torch.cat([past_k, k_cur], dim=1)
                            v = torch.cat([past_v, v_cur], dim=1)
                        else:
                            k = k_cur
                            v = v_cur
            else:
                k = k_cur
                v = v_cur

            # sanity checks
            assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
            assert q.shape[0] == k.shape[0] == v.shape[0], (q.shape, k.shape, v.shape)
            assert k.shape[1] == v.shape[1], (k.shape, v.shape)
            assert k.shape[2] == v.shape[2], (k.shape, v.shape)
            assert q.shape[3] == k.shape[3] == v.shape[3], (q.shape, k.shape, v.shape)

            attn_output = _flash_or_sdpa(
                q,
                k,
                v,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                softmax_scale=self.scaling,
                causal=False,
            )  # [B, S_q, H_q, D]

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj_mot_gen(attn_output)
            return attn_output, None

        # ------------------------------------------------------------------
        # Original eager fallback path
        # ------------------------------------------------------------------
        if past_key_values is not None:
            if update_cache:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs=None
                )
            else:
                layer = past_key_values.layers[self.layer_idx]
                past_k, past_v = layer.keys, layer.values
                if past_k is not None:
                    key_states = torch.cat([past_k, key_states], dim=2)
                    value_states = torch.cat([past_v, value_states], dim=2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj_mot_gen(attn_output)
        return attn_output, attn_weights

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(
                hidden_states,
                indexes,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(
                hidden_states,
                indexes,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )

        # Mixed und/gen path: mirrors forward_und / forward_gen per token type.
        # (Fixed per issue #207: the time-dim qk-norm, the `.view(hidden_shape)` before
        # chunking, and the transpose on the time chunk were previously missing.)
        # Note: Remove this raise once fully tested.
        raise NotImplementedError(
            "The mixed und/gen forward path is not yet validated (issue #207): known "
            "issues are fixed, but it has no parity test and no production caller. "
            "Split the sequence at token-type boundaries and use forward_und / forward_gen."
        )

        assert self.config._attn_implementation == "eager"
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = hidden_states.new_zeros(
            (*input_shape, self.config.num_attention_heads * self.head_dim)
        )
        if exist_non_image_gen_tokens:
            query_states[~image_gen_indicators] = self.q_proj(
                hidden_states[~image_gen_indicators]
            )
        if exist_image_gen_tokens:
            query_states[image_gen_indicators] = self.q_proj_mot_gen(
                hidden_states[image_gen_indicators]
            )
        query_states = query_states.view(hidden_shape)  # [B, S, H, D]
        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)

        _query_states_t = query_states_t.new_zeros(query_states_t.shape)
        if exist_non_image_gen_tokens:
            _query_states_t[~image_gen_indicators] = self.q_norm(
                query_states_t[~image_gen_indicators]
            )
        if exist_image_gen_tokens:
            _query_states_t[image_gen_indicators] = self.q_norm_mot_gen(
                query_states_t[image_gen_indicators]
            )
        query_states_t = _query_states_t.transpose(1, 2)  # [B, H, S, D/2]

        _query_states_hw = query_states_hw.new_zeros(query_states_hw.shape)
        if exist_non_image_gen_tokens:
            _query_states_hw[~image_gen_indicators] = self.q_norm_hw(
                query_states_hw[~image_gen_indicators]
            )
        if exist_image_gen_tokens:
            _query_states_hw[image_gen_indicators] = self.q_norm_hw_mot_gen(
                query_states_hw[image_gen_indicators]
            )
        query_states_hw = _query_states_hw.transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states = hidden_states.new_zeros(
            (*input_shape, self.config.num_key_value_heads * self.head_dim)
        )
        if exist_non_image_gen_tokens:
            key_states[~image_gen_indicators] = self.k_proj(
                hidden_states[~image_gen_indicators]
            )
        if exist_image_gen_tokens:
            key_states[image_gen_indicators] = self.k_proj_mot_gen(
                hidden_states[image_gen_indicators]
            )
        key_states = key_states.view(hidden_shape)  # [B, S, H_kv, D]
        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)

        _key_states_t = key_states_t.new_zeros(key_states_t.shape)
        if exist_non_image_gen_tokens:
            _key_states_t[~image_gen_indicators] = self.k_norm(
                key_states_t[~image_gen_indicators]
            )
        if exist_image_gen_tokens:
            _key_states_t[image_gen_indicators] = self.k_norm_mot_gen(
                key_states_t[image_gen_indicators]
            )
        key_states_t = _key_states_t.transpose(1, 2)  # [B, H_kv, S, D/2]

        _key_states_hw = key_states_hw.new_zeros(key_states_hw.shape)
        if exist_non_image_gen_tokens:
            _key_states_hw[~image_gen_indicators] = self.k_norm_hw(
                key_states_hw[~image_gen_indicators]
            )
        if exist_image_gen_tokens:
            _key_states_hw[image_gen_indicators] = self.k_norm_hw_mot_gen(
                key_states_hw[image_gen_indicators]
            )
        key_states_hw = _key_states_hw.transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        value_states = hidden_states.new_zeros(
            (*input_shape, self.config.num_key_value_heads * self.head_dim)
        )
        if exist_non_image_gen_tokens:
            value_states[~image_gen_indicators] = self.v_proj(
                hidden_states[~image_gen_indicators]
            )
        if exist_image_gen_tokens:
            value_states[image_gen_indicators] = self.v_proj_mot_gen(
                hidden_states[image_gen_indicators]
            )
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        query_states_t, key_states_t = apply_rotary_pos_emb(
            query_states_t, key_states_t, cos_t, sin_t
        )

        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        query_states_h, key_states_h = apply_rotary_pos_emb(
            query_states_h, key_states_h, cos_h, sin_h
        )

        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        query_states_w, key_states_w = apply_rotary_pos_emb(
            query_states_w, key_states_w, cos_w, sin_w
        )

        query_states = torch.cat(
            [query_states_t, query_states_h, query_states_w], dim=-1
        )
        key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            update_cache = kwargs.get("update_cache", True)
            if update_cache:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs=None
                )
            else:
                # only use the past key values but do not append the current one
                layer = past_key_values.layers[self.layer_idx]
                past_k, past_v = layer.keys, layer.values

                if past_k is not None:
                    key_states = torch.cat(
                        [past_k, key_states], dim=2
                    )  # concat on seq_len
                    value_states = torch.cat([past_v, value_states], dim=2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        _attn_output = attn_output.new_zeros((*input_shape, self.config.hidden_size))
        if exist_non_image_gen_tokens:
            _attn_output[~image_gen_indicators] = self.o_proj(
                attn_output[~image_gen_indicators]
            )
        if exist_image_gen_tokens:
            _attn_output[image_gen_indicators] = self.o_proj_mot_gen(
                attn_output[image_gen_indicators]
            )

        attn_output = _attn_output
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.mlp_mot_gen = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_mot_gen = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_mot_gen = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

    def forward_und(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_gen(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm_mot_gen(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm_mot_gen(hidden_states)
        hidden_states = self.mlp_mot_gen(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(
                hidden_states,
                image_gen_indicators,
                exist_non_image_gen_tokens,
                exist_image_gen_tokens,
                indexes,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
                **kwargs,
            )
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(
                hidden_states,
                image_gen_indicators,
                exist_non_image_gen_tokens,
                exist_image_gen_tokens,
                indexes,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
                **kwargs,
            )

        # Mixed und/gen path — see the NOTE in Qwen3Attention.forward for caveats.
        raise NotImplementedError(
            "Mixed und/gen decoder-layer forward is not yet validated (issue #207). "
            "Split the sequence at token-type boundaries and use forward_und / forward_gen."
        )

        residual = hidden_states

        _hidden_states = hidden_states.new_zeros(hidden_states.shape)
        if exist_non_image_gen_tokens:
            _hidden_states[~image_gen_indicators] = self.input_layernorm(
                hidden_states[~image_gen_indicators]
            )
        if exist_image_gen_tokens:
            _hidden_states[image_gen_indicators] = self.input_layernorm_mot_gen(
                hidden_states[image_gen_indicators]
            )
        hidden_states = _hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        _hidden_states = hidden_states.new_zeros(hidden_states.shape)
        if exist_non_image_gen_tokens:
            _hidden_states[~image_gen_indicators] = self.mlp(
                self.post_attention_layernorm(hidden_states[~image_gen_indicators])
            )

        if exist_image_gen_tokens:
            _hidden_states[image_gen_indicators] = self.mlp_mot_gen(
                self.post_attention_layernorm_mot_gen(
                    hidden_states[image_gen_indicators]
                )
            )

        hidden_states = _hidden_states
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.current_index = -1

        # Initialize weights and apply final processing
        self.post_init()

    @model_input_compat
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_gen_indicators: Optional[torch.Tensor] = None,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:

        # assert position_ids is not None
        # assert cache_position is not None
        # assert past_key_values is not None

        if image_gen_indicators is None:
            exist_non_image_gen_tokens = True
            exist_image_gen_tokens = False
        else:
            # Normalize once before the decoder loop. Leaving these as CUDA
            # scalar tensors makes every layer's Python branch read them back
            # to the host, serializing the compute stream and defeating weight
            # prefetch overlap.
            exist_non_image_gen_tokens = bool((~image_gen_indicators).any().item())
            exist_image_gen_tokens = bool(image_gen_indicators.any().item())

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            if input_ids is not None:
                mask_kwargs = causal_mask_kwargs(
                    create_causal_mask,
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
                # Create the masks
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                self.current_index += 1
                indexes = torch.LongTensor([[self.current_index], [0], [0]]).to(
                    input_ids.device
                )
            else:
                causal_mask_mapping = {
                    "full_attention": create_block_causal_mask(indexes[0]),
                }
                self.current_index = indexes[0].max()
        else:
            self.current_index = indexes[0].max()
            # raise NotImplementedError('not isinstance(causal_mask_mapping := attention_mask, dict)')

            # The sliding window alternating layers are not always activated depending on the config
            # if self.has_sliding_layers:
            #     causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                image_gen_indicators=image_gen_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        if not exist_image_gen_tokens:
            hidden_states = self.norm(hidden_states)
        elif not exist_non_image_gen_tokens:
            hidden_states = self.norm_mot_gen(hidden_states)
        else:
            _hidden_states = hidden_states.new_zeros(hidden_states.shape)
            _hidden_states[~image_gen_indicators] = self.norm(
                hidden_states[~image_gen_indicators]
            )
            _hidden_states[image_gen_indicators] = self.norm_mot_gen(
                hidden_states[image_gen_indicators]
            )
            hidden_states = _hidden_states
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = tied_weights_keys(
        "lm_head.weight", "model.embed_tokens.weight"
    )
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen3ForSequenceClassification(
    GenericForSequenceClassification, Qwen3PreTrainedModel
):
    pass


class Qwen3ForTokenClassification(GenericForTokenClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForQuestionAnswering(GenericForQuestionAnswering, Qwen3PreTrainedModel):
    base_model_prefix = (
        "transformer"  # For BC, where `transformer` was used instead of `model`
    )


__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering",
    "Qwen3PreTrainedModel",
    "Qwen3Model",
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
]
