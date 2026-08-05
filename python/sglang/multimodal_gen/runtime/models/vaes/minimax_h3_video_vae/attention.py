# SPDX-License-Identifier: Apache-2.0
# Attention module for the MiniMax H3 visual VAE (inference-only bundle).
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.utils import logging

from .flash import flash_attn
from .vit_utils import _env_flag, apply_rotary_pos_emb_qk

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _vit_norm_input(module, hidden_states):
    if _env_flag("MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM", "1"):
        return hidden_states.float()
    weight = getattr(module, "weight", None)
    return hidden_states.to(getattr(weight, "dtype", hidden_states.dtype))


def _apply_qk_norm(module, hidden_states):
    if (
        _env_flag("MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM", "1")
        and isinstance(module, (nn.LayerNorm, nn.RMSNorm))
        and getattr(module, "weight", None) is None
        and getattr(module, "bias", None) is None
        and hidden_states.is_cuda
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    ):
        # CUDA LayerNorm/RMSNorm accumulates half/bfloat16 inputs in FP32.
        # With no affine parameters its half output is bit-identical to the
        # released FP32-norm-then-cast recipe, without two full-tensor casts.
        with torch.autocast("cuda", enabled=False):
            return module(hidden_states)
    return module(_vit_norm_input(module, hidden_states)).to(hidden_states.dtype)


class Attention(nn.Module):
    def __init__(
        self,
        heads,
        dim_head,
        embed_dim: Optional[int] = None,
        qk_norm_type: Optional[str] = None,
        qk_norm_affine: bool = False,
        bias: bool = True,
        out_bias: Optional[bool] = None,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.attn_inner_dim = dim_head * heads
        self.embed_dim = embed_dim if embed_dim is not None else self.attn_inner_dim

        out_bias = out_bias if out_bias is not None else bias

        if qk_norm_type is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm_type == "layer_norm":
            self.norm_q = nn.LayerNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
            self.norm_k = nn.LayerNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
        elif qk_norm_type == "rms_norm":
            self.norm_q = nn.RMSNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
            self.norm_k = nn.RMSNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
        else:
            raise ValueError(
                f"unknown qk_norm_type: {qk_norm_type}. Should be None,'layer_norm','rms_norm'"
            )

        self.to_qkv = nn.Linear(self.embed_dim, self.attn_inner_dim * 3, bias=bias)

        self.to_out = nn.Linear(self.attn_inner_dim, self.embed_dim, bias=out_bias)

        if len(kwargs) > 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            logger.warning(f"Unused kwargs: {kwargs}")

    def _perform_attention(self, query, key, value, pack_info):
        cu_seqlens = pack_info.get("cu_seqlens", None)
        mask_mod = pack_info.get("mask_mod", None)
        block_sparse = pack_info.get("block_sparse", None)
        valid_seq_len = pack_info.get("valid_seq_len", None)

        if cu_seqlens is not None:
            raise NotImplementedError(
                "varlen attention is not supported in this inference-only bundle"
            )

        padded_seq_len = query.shape[1]
        if valid_seq_len is not None:
            valid_seq_len = int(valid_seq_len)
            if not 0 < valid_seq_len <= padded_seq_len:
                raise ValueError(
                    "valid_seq_len must be in (0, padded_seq_len], got "
                    f"{valid_seq_len} for padded_seq_len={padded_seq_len}"
                )
            query = query[:, :valid_seq_len]
            key = key[:, :valid_seq_len]
            value = value[:, :valid_seq_len]

        if mask_mod is not None:
            hidden_states = flash_attn(
                query,
                key,
                value,
                mask_mod=mask_mod,
                block_sparse=block_sparse,
            )
        else:
            hidden_states = flash_attn(
                query,
                key,
                value,
            )

        if valid_seq_len is not None and valid_seq_len < padded_seq_len:
            hidden_states = torch.cat(
                [
                    hidden_states,
                    hidden_states.new_zeros(
                        hidden_states.shape[0],
                        padded_seq_len - valid_seq_len,
                        hidden_states.shape[2],
                        hidden_states.shape[3],
                    ),
                ],
                dim=1,
            )

        return hidden_states

    def perform_attention(self, query, key, value, pack_info={}):
        return self._perform_attention(query, key, value, pack_info)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        pack_info: dict = {},
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.to_qkv(hidden_states)
        qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        if self.norm_q is not None:
            query = _apply_qk_norm(self.norm_q, query)
        if self.norm_k is not None:
            key = _apply_qk_norm(self.norm_k, key)

        if rotary_pos_emb is not None:
            query, key = apply_rotary_pos_emb_qk(query, key, rotary_pos_emb)

        hidden_states = self.perform_attention(query, key, value, pack_info)

        hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
        hidden_states = self.to_out(hidden_states)

        return hidden_states
