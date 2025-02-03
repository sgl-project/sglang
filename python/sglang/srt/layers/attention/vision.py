from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed import utils as dist_utils
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization import QuantizationConfig


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb_torch(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> torch.Tensor:
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(
        cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    sin = repeat(
        sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


def apply_rotary_pos_emb_vision(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    t_ = t.float()
    cos = freqs.cos()
    sin = freqs.sin()
    output = apply_rotary_emb_torch(t_, cos, sin).type_as(t)
    return output


class VisionAttention(nn.Module):
    r"""
        Multi-headed attention without any cache, mostly used for ViT.


    Args:
        use_qkv_parallel (bool, optional): If True, use QKV-parallel attention.
        use_context_forward (bool, default to True):
            if ``True``, a flash_attn style attention will be applied
            Otherwise, a full-sequence attention will be applied.
        use_full_precision_softmax (bool, default to False):
            if ``True``, the softmax will be performed in full-precision
            Otherwise, it will be performed in half-precision

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        use_qkv_parallel: bool,
        quant_config: Optional[QuantizationConfig] = None,
        dropout: float = 0.0,
        use_context_forward: bool = True,
        use_full_precision_softmax: bool = False,
        flatten_batch: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.use_context_forward = use_context_forward
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.dropout = dropout
        self.head_size = embed_dim // num_heads
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, world_size
        )

        if self.use_context_forward:
            self.qkv_backend = VisionTritonAttention()
        else:
            self.qkv_backend = VisionSdpaAttention(
                head_size=self.head_size,
                dropout=dropout,
                flatten_batch=flatten_batch,
                use_full_precision_softmax=use_full_precision_softmax,
            )

        self.use_qkv_parallel = use_qkv_parallel
        if use_qkv_parallel:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=self.head_size,
                total_num_heads=num_heads,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
        else:
            self.qkv_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=3 * projection_size,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
        self.proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        rotary_pos_emb: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            x: [b, s, embed_dim]
            cu_seqlens: [b]
        Returns:
             [s, b, num_heads * head]
        """
        bsz, s, _ = x.shape
        if self.use_qkv_parallel:
            # [b, s, embed_dim] --> [b, s, embed_dim]
            qkv, _ = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)

            # [b, s, embed_dim] --> [b * s, num_heads, head_size]
            q, k, v = [
                x.reshape(
                    bsz * s, self.num_attention_heads_per_partition, -1
                ).contiguous()
                for x in (q, k, v)
            ]
        else:
            # [b, s, embed_dim] --> [s, b, embed_dim]
            x = rearrange(x, "b s ... -> s b ...")
            # [s, b, embed_dim] --> [s, b, head * 3 * head_size]
            qkv, _ = self.qkv_proj(x)
            # [s, b, head * 3 * head_size] --> [s, b, head, 3 * head_size]
            new_x_shape = qkv.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            qkv = qkv.view(*new_x_shape)

            # [s, b, head, 3 * head_size] --> 3 [s, b, head, head_size]
            q, k, v = dist_utils.split_tensor_along_last_dim(qkv, 3)

            # [s, b, head, head_size] --> [b, s, head, head_size]
            q, k, v = [
                rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)
            ]

        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        if self.use_qkv_parallel:
            pass
        else:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]

        output = self.qkv_backend.forward(q, k, v, bsz, cu_seqlens, attention_mask)

        if self.use_qkv_parallel:
            # [b * s, h, head_size] --> [b, s, h * head_size]
            output = rearrange(output, "(b s) ... h d -> b s ... (h d)", b=bsz)

            # [b, s, h * head_size] --> [b, s, h * head_size]
            output, _ = self.proj(output)
        else:
            # [b * s, h, head_size] --> [s, b, h * head_size]
            context_layer = rearrange(
                output, "(b s) h d -> s b (h d)", b=bsz, s=s
            ).contiguous()

            # [s, b, h * head_size] --> [s, b, h * head_size]
            output, _ = self.proj(context_layer)

            # [s, b, h * head_size] --> [b, s, h * head_size]
            output = output.view(bsz, s, -1)

        return output


class VisionSdpaAttention(nn.Module):
    r"""
    Scaled Dot Product Attention inner product

    """

    # TODO: Should it be released after used?
    _mask_cache = {}

    def __init__(
        self,
        head_size: int,
        dropout: float = 0.0,
        flatten_batch: bool = False,
        use_full_precision_softmax: bool = False,
    ):
        super().__init__()
        self.head_size = head_size
        self.flatten_batch = flatten_batch
        self.use_full_precision_softmax = use_full_precision_softmax
        self.dropout = dropout

    def generate_patch_attention_mask(
        self,
        s: int,
        bsz: int,
        device,
        cu_seqlens: Optional[torch.Tensor],
        flatten_batch: bool = False,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        r"""
        Creates a non-causal 4D mask of shape `(b, 1, s, s)` or `(1, 1, s, s)`.

        When `flatten_batch` is True:
            - All sequences in the batch are flattened into a single dimension
            - `s` represents the total number of tokens across all sequences in the batch
            - Returns a unified mask of shape `(1, 1, s, s)`

        When `flatten_batch` is False:
            - Each sequence has its own attention mask
            - `s` represents the maximum sequence length in the batch
            - Returns separate masks of shape `(b, 1, s, s)`

        Args:
            flatten_batch: (bool):
                If True, treats all sequences in the batch as a single flattened sequence
                If False, generates separate masks for each sequence

        Returns:
            Tensor of shape `(b, 1, s, s)` or `(1, 1, s, s)`.
        """

        cache_key = (s, bsz, flatten_batch, tuple(cu_seqlens.cpu().tolist()))

        if cache_key in VisionSdpaAttention._mask_cache:
            cached_mask = VisionSdpaAttention._mask_cache[cache_key]
            # print(f"cache hit for key: {cache_key}")
            return cached_mask.to(device=device, dtype=dtype)

        if cu_seqlens is None:
            raise ValueError("Internal Error: cu_seqlens cannot be None")

        if flatten_batch:
            mask = torch.zeros([1, s, s], device=device, dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                start = cu_seqlens[i - 1]
                end = cu_seqlens[i]
                mask[
                    ...,
                    start:end,
                    start:end,
                ] = True
        else:
            # [1, 1, 1, s]
            row_indices = torch.arange(s, device=device).view(1, 1, 1, s)
            # [1, 1, s, 1]
            col_indices = torch.arange(s, device=device).view(1, 1, s, 1)
            # [b, 1, 1, 1]
            seq_lens = (
                (cu_seqlens[1:] - cu_seqlens[:-1]).to(device=device).view(-1, 1, 1, 1)
            )

            mask = (row_indices < seq_lens) & (col_indices < seq_lens)

        # Convert to attention mask format (False -> 0, True -> -inf)
        mask = (~mask).to(dtype) * torch.finfo(dtype).min

        VisionSdpaAttention._mask_cache[cache_key] = mask

        return mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """

        s = q.shape[0] // bsz

        # [b, 1, s, s]
        if attention_mask is None:
            attention_mask = self.generate_patch_attention_mask(
                s, bsz, q.device, cu_seqlens, self.flatten_batch, q.dtype
            )
        q, k, v = [rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in [q, k, v]]
        # [b, 1, s]
        if self.use_full_precision_softmax:
            scale = self.head_size**-0.5
            k_transposed = rearrange(k, "b h s d -> b h d s")
            attn_weights = torch.matmul(q, k_transposed) * scale
            del k, k_transposed
            attn_weights = attn_weights + attention_mask
            del attention_mask
            # full-precision
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=False
            )
            output = torch.matmul(attn_weights, v)
            del attn_weights, v
        else:
            # SDPA
            # [b, h, s, head_size]
            output = F.scaled_dot_product_attention(
                q, k, v, attention_mask, dropout_p=self.dropout
            )

        # [b, h, s, head_size] --> [b * s, h, head_size]
        output = rearrange(output, "b h s d -> (b s) h d")

        return output


class VisionTritonAttention(nn.Module):
    """
    Triton-implemented attention without a causal mask
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        _bsz: int,
        cu_seqlens: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """

        # [b * s, head, head_size]
        output = torch.empty_like(q)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()
        context_attention_fwd(
            q,
            k,
            v,
            output,
            cu_seqlens.cuda(),
            seq_lens.cuda(),
            max_seqlen,
            is_causal=False,
        )

        return output
