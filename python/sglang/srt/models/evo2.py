# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 SGLang Team
"""Inference-only Evo2 (StripedHyena 2) model compatible with Vortex checkpoints.

Evo2 is a hybrid DNA foundation model that interleaves:
- MHA layers: Standard multi-head attention with RoPE
- HCL layers: Hyena-LI (Long Impulse response) using IIR state-space
- HCM layers: Hyena-MR (Medium) using 128-token FIR filters
- HCS layers: Hyena-SE (Short) using 7-token FIR filters

Reference: https://github.com/Zymrael/vortex

Tokenizer:
  Evo2 uses a CharLevelTokenizer from Vortex: each DNA base is encoded as its
  UTF-8 byte value (A=65, C=67, G=71, T=84). The tokenizer is auto-generated
  on first use if tokenizer.json is not found in the model directory.
"""

import json
import logging
import os
from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.evo2 import Evo2Config
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer: auto-generate Vortex-compatible CharLevelTokenizer files
# ──────────────────────────────────────────────────────────────────────────────


def generate_evo2_tokenizer_files(model_path: str, vocab_size: int = 512) -> None:
    """Generate tokenizer.json and tokenizer_config.json for Evo 2.

    Evo 2 uses the Vortex CharLevelTokenizer which encodes each character as its
    raw UTF-8 byte value. DNA bases map to: A=65, C=67, G=71, T=84.
    Special tokens: <eod>=0 (BOS/EOS), <pad>=1.

    This function is idempotent — it only creates files if they don't exist.
    """
    tok_path = os.path.join(model_path, "tokenizer.json")
    cfg_path = os.path.join(model_path, "tokenizer_config.json")

    if os.path.exists(tok_path) and os.path.exists(cfg_path):
        return

    # Build vocabulary: each byte index maps to its character representation
    # In Vortex CharLevelTokenizer: decode_token(n) = chr(max(32, min(n, vocab_size)))
    vocab_dict = {}
    for i in range(vocab_size):
        if i == 0:
            vocab_dict["<eod>"] = i
        elif i == 1:
            vocab_dict["<pad>"] = i
        else:
            char_code = max(32, min(i, vocab_size))
            vocab_dict[chr(char_code)] = i

    tokenizer_json = {
        "version": "1.0",
        "added_tokens": [
            {
                "id": 0,
                "content": "<eod>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 1,
                "content": "<pad>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "unk_token": "<eod>",
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab_dict,
            "merges": [],
        },
    }

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 8192,
        "bos_token": "<eod>",
        "eos_token": "<eod>",
        "pad_token": "<pad>",
        "unk_token": "<eod>",
        "pad_token_id": 1,
        "clean_up_tokenization_spaces": False,
        "extra_special_tokens": {},
    }

    os.makedirs(model_path, exist_ok=True)

    # Atomic write: write to temp files first, then os.replace (atomic on POSIX)
    # prevents race conditions when multiple TP ranks share the model directory
    temp_tok_path = tok_path + ".tmp"
    temp_cfg_path = cfg_path + ".tmp"
    try:
        with open(temp_tok_path, "w") as f:
            json.dump(tokenizer_json, f, indent=2)
        os.replace(temp_tok_path, tok_path)

        with open(temp_cfg_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        os.replace(temp_cfg_path, cfg_path)
    except Exception:
        # Clean up temp files on failure
        for p in (temp_tok_path, temp_cfg_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        raise

    logger.info(f"Generated Evo 2 tokenizer files in {model_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: interleave channels (Evo2 uses Hyena's channel mixing pattern)
# ──────────────────────────────────────────────────────────────────────────────


def _interleave(x: torch.Tensor) -> torch.Tensor:
    """Interleave by stride-3 grouping along the channel dimension.

    Matches vortex.model.utils.interleave exactly:
    For input shape (B, C, L) [channel-first] or (B, L, C) [channel-last]:
    Channels [0,1,2,3,4,5,...] become [0,3,6,..., 1,4,7,..., 2,5,8,...].

    The operation groups channels by their index mod 3.
    """
    if x.dim() == 3 and x.shape[1] % 3 == 0:
        # Channel-first layout: (B, 3*D, L) — used by vortex short filter output
        x1 = x[:, 0::3]
        x2 = x[:, 1::3]
        v = x[:, 2::3]
        return torch.cat([x1, x2, v], dim=1)
    else:
        # Channel-last layout: (B, L, 3*D) — used by our short filter output
        x1 = x[..., 0::3]
        x2 = x[..., 1::3]
        v = x[..., 2::3]
        return torch.cat([x1, x2, v], dim=-1)


def _column_split(
    x: torch.Tensor, num_heads: int, head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split into (x1, x2, v) using column-wise split pattern.

    The input x of shape (B, L, 3*H*head_dim) is split into 3 chunks
    of (H*head_dim) each, reshaped to (B, L, H, head_dim), and split
    across the head dimension into x2, x1, v (one head_dim each).

    Returns:
        (x2, x1, v): each of shape (B, L, hidden_size)
    """
    # x shape: (B, L, 3 * num_heads * head_dim)
    x = x.view(*x.shape[:-1], 3, num_heads, head_dim)
    x2, x1, v = x.unbind(dim=-3)
    return x2.flatten(start_dim=-2), x1.flatten(start_dim=-2), v.flatten(start_dim=-2)


# ──────────────────────────────────────────────────────────────────────────────
# Evo2MLP – Gated MLP with optional Evo2-style (identity for layer > 0)
# ──────────────────────────────────────────────────────────────────────────────


class Evo2MLP(nn.Module):
    """Gated MLP used in both Attention and Hyena blocks.

    Evo2 uses GELU activation (not SiLU) and optionally uses identity
    activation for layers > 0 when evo2_style_activations=True.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "gelu",
        layer_idx: int = 0,
        evo2_style_activations: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.evo2_style = evo2_style_activations
        self.hidden_act = hidden_act

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
            reduce_results=reduce_results,
        )

        # For Evo2, layers > 0 use identity activation (only layer 0 uses GELU)
        if hidden_act == "gelu":
            self.act_fn = nn.GELU(approximate="tanh")
        elif hidden_act == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def forward(self, x, forward_batch=None, use_reduce_scatter: bool = False):
        gate_up, _ = self.gate_up_proj(x)

        # Split gate and up
        gate, up = gate_up.chunk(2, dim=-1)

        # Evo2 style: only layer 0 uses activation, others use identity
        if self.evo2_style and self.layer_idx > 0:
            activation = gate  # identity
        else:
            activation = self.act_fn(gate)

        x = activation * up
        x, _ = self.down_proj(x, skip_all_reduce=use_reduce_scatter)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Evo2Attention – MHA with RoPE + KV cache
# ──────────────────────────────────────────────────────────────────────────────


class Evo2Attention(nn.Module):
    """Multi-head self-attention with RoPE for Evo2 attention layers."""

    def __init__(
        self,
        config: Evo2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rotary_emb_base,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.qkv_proj_bias,
            quant_config=quant_config,
        )

        # output projection (named out_proj in vortex)
        self.out_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.mha_out_proj_bias,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.out_proj(attn_output)
        return output


# ──────────────────────────────────────────────────────────────────────────────
# Evo2AttentionLayer – Full attention block (MHA + MLP with pre-norm residual)
# ──────────────────────────────────────────────────────────────────────────────


class Evo2AttentionLayer(nn.Module):
    """Attention decoder layer: PreNorm → MHA → Residual → PostNorm → MLP → Residual."""

    def __init__(
        self,
        config: Evo2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.self_attn = Evo2Attention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("inner_mha_cls", prefix),
        )

        self.mlp = Evo2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.mlp_activation,
            layer_idx=layer_id,
            evo2_style_activations=config.evo2_style_activations,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm + Attention + Residual
        if residual is None:
            residual = hidden_states
            hidden_states = self.pre_norm(hidden_states)
        else:
            hidden_states, residual = self.pre_norm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Post-norm + MLP + Residual
        hidden_states, residual = self.post_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ──────────────────────────────────────────────────────────────────────────────
# Evo2HyenaFilter – Short FIR + Long filter (IIR or FIR)
# ──────────────────────────────────────────────────────────────────────────────


class Evo2HyenaFilter(nn.Module):
    """Hyena cascade filter: short depthwise FIR → interleave → long filter.

    Handles all three Hyena variants:
    - HCL: IIR long filter with log_poles + residues (state_size poles)
    - HCM: FIR long filter with learned impulse response (128 tokens)
    - HCS: FIR long filter with learned impulse response (7 tokens)
    """

    def __init__(
        self,
        config: Evo2Config,
        layer_idx: int,
        hyena_filter_groups: int = None,
        fir_inner_filter_length: int = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_filters = config.num_filters
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.state_size = config.state_size
        self.short_filter_length = config.short_filter_length
        self.fir_inner_filter_length = fir_inner_filter_length
        self.hyena_filter_groups = (
            hyena_filter_groups
            if hyena_filter_groups is not None
            else config.hidden_size
        )
        self.interleave = config.interleave
        self.column_split_hyena = config.column_split_hyena

        # Short FIR filter: depthwise conv1d (3 tokens)
        # Shape: (3 * hidden_size, 1, short_filter_length)
        self.short_filter_weight = nn.Parameter(
            torch.randn(3 * config.hidden_size, 1, config.short_filter_length)
        )
        if config.short_filter_bias:
            self.short_filter_bias = nn.Parameter(torch.randn(3 * config.hidden_size))
        else:
            self.short_filter_bias = None

        # Long filter
        if fir_inner_filter_length is not None:
            # FIR-based: HCM (128) or HCS (7)
            self.h = nn.Parameter(
                torch.randn(self.hyena_filter_groups, 1, fir_inner_filter_length)
            )
            if fir_inner_filter_length >= 128:
                self.D = nn.Parameter(torch.zeros(self.hidden_size))
            else:
                self.D = None
            self.log_poles = None
            self.residues = None
        else:
            # IIR-based: HCL with log_poles + residues
            self.log_poles = nn.Parameter(
                torch.randn(
                    self.hyena_filter_groups, self.state_size, 1, dtype=torch.float32
                )
            )
            self.residues = nn.Parameter(
                torch.randn(
                    self.hyena_filter_groups, self.state_size, dtype=torch.float32
                )
            )
            self.D = nn.Parameter(torch.zeros(self.hidden_size))
            self.h = None

        # Precomputed time vector for IIR mode
        self.register_buffer("t", None, persistent=False)

    def _update_time(self, L: int, device: torch.device):
        """Update the time vector for impulse response computation."""
        if self.t is None or self.t.device != device or self.t.shape[-1] < L:
            self.t = torch.arange(L, device=device, dtype=torch.float32)[None, None]

    def _compute_iir_filter(self, L: int, device: torch.device) -> torch.Tensor:
        """Compute IIR impulse response h[t] = sum_k residues_k * exp(log_poles_k * t)."""
        self._update_time(L, device)
        residues = self.residues.to(torch.float32)  # (G, S)
        log_poles = self.log_poles.to(torch.float32)  # (G, S, 1)
        t = self.t[:, :, :L]  # (1, 1, L)
        # h shape: (1, G, L) then squeeze to (G, L)
        h = (residues[..., None] * (log_poles * t).exp()).sum(dim=1)  # (G, L)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        return h.unsqueeze(1)  # (G, 1, L)

    def _apply_short_filter(
        self,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Apply short depthwise FIR filter.

        Args:
            u: (B, L, 3*hidden_size)
        Returns:
            u_filt: (B, L, 3*hidden_size)
        """
        # Handle 2D input during CUDA graph capture (no batch dim)
        if u.dim() == 2:
            u = u.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        u_t = u.transpose(1, 2)  # (B, 3*hidden_size, L)
        # Pad for causal conv
        pad = self.short_filter_length - 1
        u_padded = F.pad(u_t, (pad, 0))
        u_filt = F.conv1d(
            u_padded,
            self.short_filter_weight.to(u.dtype),
            bias=(
                self.short_filter_bias.to(u.dtype)
                if self.short_filter_bias is not None
                else None
            ),
            groups=3 * self.hidden_size,
        )
        u_filt = u_filt.transpose(1, 2)  # (B, L, 3*hidden_size)
        if squeeze_back:
            u_filt = u_filt.squeeze(0)
        return u_filt

    def _apply_long_filter(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the long (IIR or FIR) filter.

        Args:
            z: (B, L, 3*hidden_size) – already short-filtered
        Returns:
            y: (B, L, hidden_size)
        """
        # Handle 2D input during CUDA graph capture
        if z.dim() == 2:
            z = z.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        B, L, _ = z.shape
        orig_dtype = z.dtype

        # Run long filter in float32 for numerical stability
        z_f = z.float()

        # Split into x1, x2, v based on column_split mode
        if self.column_split_hyena:
            x2, x1, v = _column_split(z_f, self.num_attention_heads, self.head_dim)
        else:
            x1, x2, v = z_f.split(self.hidden_size, dim=-1)

        # Prepare filter h
        if self.h is not None:
            # FIR mode (HCM/HCS)
            h = self.h.float()  # (G, 1, fir_len)
            if self.hyena_filter_groups > 1:
                h = h.repeat_interleave(
                    self.hidden_size // self.hyena_filter_groups, dim=0
                )
            gate = x1 * v
            gate_t = gate.transpose(1, 2)  # (B, D, L) float32
            if self.fir_inner_filter_length >= 128:
                # HCM: 128-token FIR — use FFT-based convolution
                # matching vortex fftconv_func for numerical fidelity.
                seqlen = gate_t.shape[-1]
                fft_size = 2 * seqlen
                h_squeezed = h.squeeze(1)  # (D, fir_len)
                h_use = h_squeezed[..., :seqlen]  # (D, min(fir_len, seqlen))
                k_f = torch.fft.rfft(h_use, n=fft_size) / fft_size  # (D, F)
                k_f = k_f.unsqueeze(0)  # (1, D, F) for broadcast
                u_f = torch.fft.rfft(gate_t, n=fft_size)  # (B, D, F)
                y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[
                    ..., :seqlen
                ]  # (B, D, L)
            else:
                # HCS: 7-token FIR — direct conv1d (vortex uses same)
                h_use = h
                pad = h_use.shape[-1] - 1
                y = F.conv1d(F.pad(gate_t, (pad, 0)), h_use, groups=gate_t.shape[1])
            if self.D is not None:
                y = y + self.D.float()[None, :, None] * gate_t
            y = y.transpose(1, 2)  # (B, L, D)
            y = y * x2
        else:
            # IIR mode (HCL)
            h_raw = self._compute_iir_filter(L, z_f.device)  # (G, 1, L) float32
            if self.hyena_filter_groups > 1:
                h_raw = h_raw.repeat_interleave(
                    self.hidden_size // self.hyena_filter_groups, dim=0
                )
            h = h_raw.squeeze(1)  # (D, L) float32

            gate = x1 * v  # (B, L, D) float32

            # FFT convolution in float32
            gate_t = gate.transpose(1, 2)  # (B, D, L)
            fft_len = 2 * L
            k_f = torch.fft.rfft(h, n=fft_len) / fft_len  # (D, F)
            u_f = torch.fft.rfft(gate_t, n=fft_len)  # (B, D, F)
            y = torch.fft.irfft(u_f * k_f, n=fft_len, norm="forward")[
                ..., :L
            ]  # (B, D, L)
            y = y.transpose(1, 2)  # (B, L, D)

            if self.D is not None:
                y = y + self.D.float()[None, None, :] * gate
            y = y * x2

        # Cast back
        y = y.to(orig_dtype)

        if squeeze_back:
            y = y.squeeze(0)
        return y

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass for prefill (batch processing).

        Args:
            u: (B, L, 3*hidden_size) – output of input projection
        Returns:
            y: (B, L, hidden_size)
        """
        # Handle 2D input during CUDA graph capture
        if u.dim() == 2:
            u = u.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        L = u.shape[1]

        # 1. Short FIR filter — run in float32 for 7B stability
        z = self._apply_short_filter(u.float()).to(u.dtype)  # (B, L, 3*hidden_size)

        # 2. Interleave channels (Evo2 style)
        if self.interleave:
            z = _interleave(z)

        # 3. Long filter (IIR or FIR) — runs float32 internally
        y = self._apply_long_filter(z)  # (B, L, hidden_size)

        if squeeze_back:
            y = y.squeeze(0)
        return y


# ──────────────────────────────────────────────────────────────────────────────
# Evo2HyenaConvLayer – Full Hyena convolution block
# ──────────────────────────────────────────────────────────────────────────────


class Evo2HyenaConvLayer(nn.Module):
    """Hyena convolution decoder layer.

    Flow: PreNorm → InputProj(→3×hidden) → HyenaFilter → OutProj + Residual
          → PostNorm → MLP + Residual
    """

    def __init__(
        self,
        config: Evo2Config,
        layer_id: int,
        hyena_filter_groups: int = None,
        fir_inner_filter_length: int = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.fir_inner_filter_length = fir_inner_filter_length
        self.hyena_filter_groups = (
            hyena_filter_groups
            if hyena_filter_groups is not None
            else config.hidden_size
        )
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Input projection: hidden_size → 3 * hidden_size (x1, x2, v)
        self.projections = MergedColumnParallelLinear(
            config.hidden_size,
            [config.hidden_size] * 3,
            bias=config.qkv_proj_bias,
            quant_config=quant_config,
            prefix=add_prefix("projections", prefix),
        )

        # Hyena cascade filter
        self.filter = Evo2HyenaFilter(
            config=config,
            layer_idx=layer_id,
            hyena_filter_groups=self.hyena_filter_groups,
            fir_inner_filter_length=fir_inner_filter_length,
        )

        # Output projection
        self.out_filter_dense = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=config.hyena_out_proj_bias,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("out_filter_dense", prefix),
        )

        # MLP
        self.mlp = Evo2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.mlp_activation,
            layer_idx=layer_id,
            evo2_style_activations=config.evo2_style_activations,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm for filter (matches sglang residual pattern)
        if residual is None:
            residual = hidden_states
            hidden_states = self.pre_norm(hidden_states)
        else:
            hidden_states, residual = self.pre_norm(hidden_states, residual)

        # Input projection: (B, L, D) → (B, L, 3*D)
        proj_out, _ = self.projections(hidden_states)

        # All-gather if using TP, because the Hyena filter's interleave step
        # mixes channels pairwise across the full hidden dimension.
        if self.tp_size > 1:
            proj_out = tensor_model_parallel_all_gather(proj_out)

        # Hyena filter (causal: requires forward_batch for sequence boundaries)
        z = self.filter(proj_out)  # (B, L, D)

        # Split back for row-parallel output projection
        if self.tp_size > 1:
            z = z.chunk(self.tp_size, dim=-1)[self.tp_rank]

        # Output projection
        z_out, _ = self.out_filter_dense(z)

        # Post-norm for MLP (adds filter output to residual stream)
        hidden_states, residual = self.post_norm(z_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ──────────────────────────────────────────────────────────────────────────────
# Evo2Model – The full StripedHyena 2 model
# ──────────────────────────────────────────────────────────────────────────────


class Evo2Model(nn.Module):
    """Evo2 (StripedHyena 2) model with interleaved attention and Hyena layers."""

    def __init__(
        self,
        config: Evo2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers

        # Embedding
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Build layers: determine type for each layer index
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_layers):
            if layer_idx in config.attn_layer_idxs:
                layer = Evo2AttentionLayer(
                    config=config,
                    layer_id=layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
            elif layer_idx in config.hcl_layer_idxs:
                # HCL: IIR filter, no FIR inner filter
                layer = Evo2HyenaConvLayer(
                    config=config,
                    layer_id=layer_idx,
                    hyena_filter_groups=config.hcl_filter_groups,
                    fir_inner_filter_length=None,  # IIR mode
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
            elif layer_idx in config.hcm_layer_idxs:
                # HCM: FIR filter with 128-token length
                layer = Evo2HyenaConvLayer(
                    config=config,
                    layer_id=layer_idx,
                    hyena_filter_groups=config.hcm_filter_groups,
                    fir_inner_filter_length=config.hcm_filter_length,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
            elif layer_idx in config.hcs_layer_idxs:
                # HCS: FIR filter with 7-token length
                layer = Evo2HyenaConvLayer(
                    config=config,
                    layer_id=layer_idx,
                    hyena_filter_groups=config.hcs_filter_groups,
                    fir_inner_filter_length=config.hcs_filter_length,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
            else:
                raise ValueError(
                    f"Layer {layer_idx} not assigned to any block type. "
                    f"Check attn_layer_idxs, hcl_layer_idxs, hcm_layer_idxs, hcs_layer_idxs."
                )
            self.layers.append(layer)

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )

        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states


# ──────────────────────────────────────────────────────────────────────────────
# Evo2ForCausalLM – Model with LM head and weight loading
# ──────────────────────────────────────────────────────────────────────────────


class Evo2ForCausalLM(nn.Module):
    """Evo2 model with causal LM head, compatible with Vortex checkpoints.

    Supports:
    - Evo2-1B (hidden=2048, layers=32)
    - Evo2-7B (hidden=4096, layers=32)
    - Evo2-40B (hidden=8192, layers=50)
    """

    def __init__(
        self,
        config: Evo2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.model = Evo2Model(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        # Tied embeddings: lm_head uses the embedding weight
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=inputs_embeds,
        )
        # Evo2 uses tied embeddings; during autoregressive decode the IIR
        # filter can accumulate bf16 numerical error that produces NaN in
        # the logits.  nan_to_num here keeps the sampler from crashing
        # while preserving the model's output distribution.
        hidden_states = torch.nan_to_num(
            hidden_states, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        output = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        return output

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from a Vortex checkpoint.

        Handles weight name remapping from Vortex format to sglang format,
        including stacked parameters (qkv_proj, gate_up_proj) that require
        shard_id-aware loading.
        """
        # (param_name, shard_name, shard_id) — used for stacked parameters
        stacked_params_mapping = [
            (".gate_up_proj", ".l1", 0),
            (".gate_up_proj", ".l2", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            # Skip non-parameter keys
            if name.endswith("._extra_state"):
                continue

            # Skip FP8 extra states, non-weight buffers, and optimizer states
            if "fp8" in name.lower() or "_extra_state" in name:
                continue
            if any(x in name for x in (".inv_freq", "filter.t")):
                continue
            # Skip optimizer/scheduler states from .pt checkpoints
            if name.startswith(("optimizer", "scheduler", "iteration")):
                continue

            # Remap Vortex weight name → sglang name (before stacking logic)
            sglang_name = self._remap_weight_name(name)
            if sglang_name is None:
                continue

            # --- Handle stacked parameters via shard_id-aware loading ---
            matched = False
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in sglang_name:
                    continue
                stacked_name = sglang_name.replace(shard_name, param_name)
                if stacked_name not in params_dict:
                    continue
                param = params_dict[stacked_name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is None:
                    weight_loader = default_weight_loader
                weight_loader(param, loaded_weight, loaded_shard_id=shard_id)
                loaded_params.add(stacked_name)
                matched = True
                break

            if matched:
                continue

            # --- Direct weight loading (non-stacked parameters) ---
            if sglang_name not in params_dict:
                logger.warning(
                    f"Parameter {sglang_name} (from {name}) not found in model. Skipping."
                )
                continue

            param = params_dict[sglang_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)

            # Handle shape mismatches (e.g., 2D→3D unsqueeze, dtype differences)
            if param.size() != loaded_weight.size():
                try:
                    loaded_weight = loaded_weight.reshape(param.size())
                except RuntimeError:
                    logger.warning(
                        f"Shape mismatch for {sglang_name}: "
                        f"param={list(param.shape)}, loaded={list(loaded_weight.shape)}. Skipping."
                    )
                    continue

            weight_loader(param, loaded_weight)
            loaded_params.add(sglang_name)

        # Log any unloaded parameters
        unloaded = set(params_dict.keys()) - loaded_params
        if unloaded:
            logger.warning(f"Unloaded parameters: {unloaded}")

    def _remap_weight_name(self, vortex_name: str) -> str:
        """Remap a Vortex checkpoint weight name to sglang parameter name.

        Vortex naming conventions:
          embedding_layer.weight          → embed_tokens.weight
          blocks.{i}.pre_norm.scale       → layers.{i}.pre_norm.weight
          blocks.{i}.post_norm.scale      → layers.{i}.post_norm.weight
          blocks.{i}.inner_mha_cls.Wqkv.weight → layers.{i}.self_attn.Wqkv.weight
          blocks.{i}.inner_mha_cls.out_proj.weight → layers.{i}.self_attn.out_proj.weight
          blocks.{i}.mlp.l1.weight        → layers.{i}.mlp.l1.weight (gate, stacked)
          blocks.{i}.mlp.l2.weight        → layers.{i}.mlp.l2.weight (up, stacked)
          blocks.{i}.mlp.l3.weight        → layers.{i}.mlp.down_proj.weight
          ...

        Note: .l1/.l2 and .Wqkv suffixes are preserved for stacked parameter
        loading in load_weights(), which replaces them with .gate_up_proj
        and .qkv_proj respectively using shard_id-aware loading.
        """
        name = vortex_name

        # Top-level mappings
        if name == "embedding_layer.weight":
            return "model.embed_tokens.weight"
        if name == "norm.scale":
            return "model.norm.weight"
        if name.startswith("unembed"):
            return "lm_head.weight" if not self.config.tie_word_embeddings else None

        # Block-level mappings
        # blocks.{i}.xxx → layers.{i}.xxx
        if name.startswith("blocks."):
            parts = name.split(".", 2)
            # parts[0] = "blocks", parts[1] = layer_idx, parts[2] = rest
            layer_idx = parts[1]
            rest = parts[2]

            # Pre/post norm
            if rest == "pre_norm.scale":
                return f"model.layers.{layer_idx}.pre_norm.weight"
            if rest == "post_norm.scale":
                return f"model.layers.{layer_idx}.post_norm.weight"

            # MHA — Wqkv is a single combined QKV tensor (not stacked)
            if rest == "inner_mha_cls.Wqkv.weight":
                return f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
            if rest == "inner_mha_cls.out_proj.weight":
                return f"model.layers.{layer_idx}.self_attn.out_proj.weight"
            if rest == "inner_mha_cls.out_proj.bias":
                return f"model.layers.{layer_idx}.self_attn.out_proj.bias"

            # MLP — l1/l2 are stacked into gate_up_proj; l3 is direct
            if rest == "mlp.l1.weight":
                return f"model.layers.{layer_idx}.mlp.l1.weight"
            if rest == "mlp.l2.weight":
                return f"model.layers.{layer_idx}.mlp.l2.weight"
            if rest == "mlp.l3.weight":
                return f"model.layers.{layer_idx}.mlp.down_proj.weight"

            # Hyena projections
            if rest == "projections.weight":
                return f"model.layers.{layer_idx}.projections.weight"

            # Hyena filter
            if rest == "filter.short_filter_weight":
                return f"model.layers.{layer_idx}.filter.short_filter_weight"
            if rest == "filter.short_filter_bias":
                return f"model.layers.{layer_idx}.filter.short_filter_bias"
            if rest == "filter.h":
                return f"model.layers.{layer_idx}.filter.h"
            if rest == "filter.D":
                return f"model.layers.{layer_idx}.filter.D"
            if rest == "filter.log_poles":
                return f"model.layers.{layer_idx}.filter.log_poles"
            if rest == "filter.residues":
                return f"model.layers.{layer_idx}.filter.residues"

            # Hyena output projection
            if rest == "out_filter_dense.weight":
                return f"model.layers.{layer_idx}.out_filter_dense.weight"
            if rest == "out_filter_dense.bias":
                return f"model.layers.{layer_idx}.out_filter_dense.bias"

        return name


# ──────────────────────────────────────────────────────────────────────────────
# Model registry entry point
# ──────────────────────────────────────────────────────────────────────────────

EntryClass = Evo2ForCausalLM
