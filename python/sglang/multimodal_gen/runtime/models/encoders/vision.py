# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/vision.py

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_C = TypeVar("_C", bound=PretrainedConfig)


class VisionEncoderInfo(ABC, Generic[_C]):

    def __init__(self, vision_config: _C) -> None:
        super().__init__()

        self.vision_config = vision_config

    @abstractmethod
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_max_image_tokens(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_image_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_patch_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_patch_grid_length(self) -> int:
        raise NotImplementedError


def get_vit_attn_backend(
    attn_implementation: str | None = None,
) -> AttentionBackendEnum:
    """Resolve the best available attention backend for Vision Transformer.

    Centralizes backend selection with automatic fallback:
    FlashAttention (if available) → PyTorch SDPA.

    Args:
        attn_implementation: Optional backend hint from model config
            (e.g. HuggingFace ``_attn_implementation``).
            ``"flash_attention_2"`` or ``None`` → try FA, fallback to SDPA.
            ``"sdpa"`` → use PyTorch SDPA directly.
    """
    if attn_implementation == "sdpa":
        return AttentionBackendEnum.TORCH_SDPA

    try:
        import flash_attn  # noqa: F401

        return AttentionBackendEnum.FA
    except ImportError:
        logger.warning(
            "flash_attn is not installed. " "Falling back to SDPA for ViT attention."
        )
        return AttentionBackendEnum.TORCH_SDPA


def _vit_flash_attn_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """FlashAttention variable-length attention for ViT.

    Args:
        query/key/value: [total_seq_len, num_heads, head_dim]
        cu_seqlens: [batch_size + 1] cumulative sequence lengths
        max_seqlen: maximum sequence length in the batch
    """
    from flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout_p,
        causal=False,
    )


def _vit_sdpa_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """SDPA attention with variable-length sequences for ViT.

    Args:
        query/key/value: [1, num_heads, total_seq_len, head_dim]
        cu_seqlens: [batch_size + 1] cumulative sequence lengths
    """
    outputs = []
    for i in range(len(cu_seqlens) - 1):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        output_i = F.scaled_dot_product_attention(
            query[:, :, start_idx:end_idx, :].contiguous(),
            key[:, :, start_idx:end_idx, :].contiguous(),
            value[:, :, start_idx:end_idx, :].contiguous(),
            dropout_p=dropout_p,
            is_causal=False,
        )
        outputs.append(output_i)
    return torch.cat(outputs, dim=2)


def vit_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: Optional[int] = None,
    attn_backend: AttentionBackendEnum = AttentionBackendEnum.FA,
    scaling: float = 1.0,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Unified attention forward for ViT supporting multiple backends.

    Args:
        query/key/value: [seq_len, num_heads, head_dim]
        cu_seqlens: cumulative sequence lengths
        attn_backend: AttentionBackendEnum.FA or AttentionBackendEnum.TORCH_SDPA
    Returns:
        output: [seq_len, hidden_dim]
    """
    seq_len, num_heads, head_dim = query.shape

    if attn_backend == AttentionBackendEnum.FA:
        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        output = _vit_flash_attn_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=dropout_p,
        )
        return output.reshape(seq_len, -1).contiguous()

    # Reshape for SDPA: [seq, heads, dim] -> [1, heads, seq, dim]
    query = query.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)

    output = _vit_sdpa_varlen(
        query,
        key,
        value,
        cu_seqlens=cu_seqlens,
        dropout_p=dropout_p,
    )

    # [1, heads, seq, dim] -> [seq, hidden]
    return output.squeeze(0).transpose(0, 1).reshape(seq_len, -1).contiguous()


def resolve_visual_encoder_outputs(
    encoder_outputs: torch.Tensor | list[torch.Tensor],
    feature_sample_layers: list[int] | None,
    post_layer_norm: torch.nn.LayerNorm | None,
    max_possible_layers: int,
) -> torch.Tensor:
    """Given the outputs a visual encoder module that may correspond to the
    output of the last layer, or a list of hidden states to be stacked,
    handle post normalization and resolve it into a single output tensor.

    Args:
        encoder_outputs: Output of encoder's last layer or all hidden states.
        feature_sample_layers: Optional layer indices to grab from the encoder
            outputs; if provided, encoder outputs must be a list.
        post_layer_norm: Post norm to apply to the output of the encoder.
        max_possible_layers: Total layers in the fully loaded visual encoder.

    """
    if feature_sample_layers is None:
        if post_layer_norm is not None:
            return post_layer_norm(encoder_outputs)
        return encoder_outputs

    # Get the hidden states corresponding to the layer indices.
    # Negative values are relative to the full visual encoder,
    # so offset them depending on how many layers were loaded.
    # NOTE: this assumes that encoder_outputs is a list containing
    # the inputs to the visual encoder, followed by the hidden states
    # of each layer.
    num_loaded_layers = len(encoder_outputs) - 1
    offset = max_possible_layers - num_loaded_layers
    hs_pool = [
        (
            encoder_outputs[layer_idx]
            if layer_idx >= 0
            else encoder_outputs[layer_idx + offset]
        )
        for layer_idx in feature_sample_layers
    ]

    # Apply post-norm on the final hidden state if we are using it
    uses_last_layer = feature_sample_layers[-1] in (len(hs_pool) - 1, -1)
    if post_layer_norm is not None and uses_last_layer:
        hs_pool[-1] = post_layer_norm(encoder_outputs)
    return torch.cat(hs_pool, dim=-1)
