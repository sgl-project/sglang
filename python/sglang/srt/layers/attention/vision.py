from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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

# def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
#     if not interleaved:
#         x1, x2 = x.chunk(2, dim=-1)
#         return torch.cat((-x2, x1), dim=-1)
#     else:
#         x1, x2 = x[..., ::2], x[..., 1::2]
#         return rearrange(
#             torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
#         )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_emb_torch(
#     x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
# ) -> torch.Tensor:
#     """
#     x: (batch_size, seqlen, nheads, headdim)
#     cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
#     """
#     ro_dim = cos.shape[-1] * 2
#     assert ro_dim <= x.shape[-1]
#     cos = repeat(
#         cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
#     )
#     sin = repeat(
#         sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
#     )
#     return torch.cat(
#         [
#             x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
#             x[..., ro_dim:],
#         ],
#         dim=-1,
#     )


# 另一个版本：返回更短的hash值
def tensor_hash_short(tensor, bits=32):
    """
    计算tensor的简单hash值，返回较短的结果

    参数:
    tensor: torch.Tensor - 输入张量
    bits: int - 结果的位数

    返回:
    int - hash值
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    tensor = tensor.float()

    # 使用张量操作计算hash
    hash_value = torch.sum(tensor).item()
    hash_value = hash_value * torch.prod(torch.tensor(tensor.shape)).item()

    # 加入一些张量特征
    std_value = torch.std(tensor).item()
    mean_value = torch.mean(tensor).item()

    # 组合所有特征
    final_hash = int((hash_value + std_value * 1e6 + mean_value * 1e4) * 1e6)

    # 取模以限制位数
    return final_hash % (2**bits)


# def apply_rotary_pos_emb_vision(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
#     t_ = t.float()
#     cos = freqs.cos()
#     sin = freqs.sin()
#     output = apply_rotary_emb_torch(t_, cos, sin).type_as(t)
#     return output


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if firstVA:
        print(f"q shape {q.shape}")
        print(f"k shape {k.shape}")
        print(f"cos shape {cos.shape}")
        print(f"sin shape {sin.shape}")
        print(f"q {tensor_hash_short(q)}")
        print(f"k {tensor_hash_short(k)}")
        print(f"cos {tensor_hash_short(cos)}")
        print(f"sin {tensor_hash_short(sin)}")
        torch.save(q, "q_torch")
        torch.save(k, "k_torch")
        torch.save(cos, "cos_torch")
        torch.save(sin, "sin_torch")

    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    if firstVA:
        print(f"q {tensor_hash_short(q)}")
        print(f"k {tensor_hash_short(k)}")

    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    if firstVA:
        print(f"q {tensor_hash_short(q_embed)}")
        print(f"k {tensor_hash_short(k_embed)}")
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    if firstVA:
        print(f"q {tensor_hash_short(q_embed)}")
        print(f"k {tensor_hash_short(k_embed)}")
    return q_embed, k_embed


class VisionAttention(nn.Module):
    """
        Multi-headed attention without any cache, mostly used for ViT.

    Args:
        use_qkv_parallel (bool, optional):
            If ``True``, use QKV-parallel attention.
        use_context_forward (bool, default to True):
            if ``True``, a flash_attn style attention will be applied
            Otherwise, a full-sequence attention will be applied.
        use_full_precision_softmax (bool, default to False):
            if ``True``, the softmax will be performed in full-precision (float32)
            Otherwise, it will be performed in the same precision with qkv

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
        position_embeddings: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            x: [b, s, embed_dim]
            cu_seqlens: [b]
        Returns:
             [s, b, head * head_size]
        """
        bsz, s, _ = x.shape
        head = self.num_attention_heads_per_partition
        if self.use_qkv_parallel:
            # [b, s, embed_dim] --> [b, s, embed_dim]
            qkv, _ = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)

            # [b, s, embed_dim] --> [b * s, head, head_size]
            q, k, v = [x.reshape(bsz * s, head, -1).contiguous() for x in (q, k, v)]
        else:
            # [b, s, embed_dim] --> [s, b, embed_dim]
            x = rearrange(x, "b s ... -> s b ...")
            # [s, b, embed_dim] --> [s, b, head * 3 * head_size]
            qkv, _ = self.qkv_proj(x)
            # [s, b, head * 3 * head_size] --> [s, b, head, 3 * head_size]
            new_x_shape = qkv.size()[:-1] + (
                head,
                3 * self.hidden_size_per_attention_head,
            )
            qkv = qkv.view(*new_x_shape)

            # [s, b, head, 3 * head_size] --> 3 [s, b, head, head_size]
            q, k, v = dist_utils.split_tensor_along_last_dim(qkv, 3)

            # [s, b, head, head_size] --> [b, s, head, head_size]
            q, k, v = [
                rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)
            ]

        global firstVA
        if firstVA:
            save_tensor(q, "q_before")
            save_tensor(k, "k_before")

        if position_embeddings is not None:
            cos, sin = position_embeddings
            # rotary position embedding has to
            original_shape = q.shape
            q, k = q.view(s, head, -1), k.view(s, head, -1)
            if firstVA:
                save_tensor(cos, "cos")
                save_tensor(sin, "sin")
                print(f"q rotary shape: {q.shape}")
                print(f"{q.dtype} {k.dtype} {cos.dtype} {sin.dtype}")
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            if firstVA:
                save_tensor(q, "q_after")
            q, k = q.reshape(original_shape), k.reshape(original_shape)

        # if rotary_pos_emb is not None:
        #     q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        #     k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

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

        firstVA = False

        return output


def save_tensor(tensor, filename):
    import os

    # 将 tensor 转换为 ndarray
    if os.path.exists(filename):
        return
    torch.save(tensor, filename)


first = True
firstVA = True


class VisionSdpaAttention(nn.Module):
    r"""
    Scaled Dot Product Attention inner product

    """

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

    @staticmethod
    @lru_cache(maxsize=128)
    def _generate_mask_cache(
        s: int, flatten_batch: bool, cu_seqlens: tuple
    ) -> torch.BoolTensor:
        """
        Generate a boolean attention mask with caching mechanism.
        Args:
            s: sequence length
            flatten_batch: whether to flatten batch dimension
            cu_seqlens: tuple of cumulative sequencelengths
        Returns:
            attention mask tensor
        """
        if flatten_batch:
            mask = torch.zeros([1, s, s], dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                start = cu_seqlens[i - 1]
                end = cu_seqlens[i]
                mask[..., start:end, start:end] = True
        else:
            # [1, 1, 1, s]
            row_indices = torch.arange(s).view(1, 1, 1, s)
            # [1, 1, s, 1]
            col_indices = torch.arange(s).view(1, 1, s, 1)
            # [b, 1, 1, 1]
            seq_lens = torch.tensor(
                [end - start for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])],
            ).view(-1, 1, 1, 1)

            mask = (row_indices < seq_lens) & (col_indices < seq_lens)

        return mask

    def generate_patch_attention_mask(
        self,
        s: int,
        cu_seqlens: Optional[torch.Tensor],
        flatten_batch: bool = False,
    ) -> Optional[torch.Tensor]:
        r"""
        Creates a non-causal 4D mask of shape `(b, 1, s, s)` or `(1, 1, s, s)`.
        Args:
            s: sequence length
            cu_seqlens: cumulative sequence lengths tensor. If not, returns an empty mask
            flatten_batch: whether to flatten batch dimension
        Returns:
            attention mask tensor or None
        """
        if cu_seqlens is None:
            return None

        cu_seqlens_tuple = tuple(cu_seqlens.cpu().tolist())

        return self._generate_mask_cache(s, flatten_batch, cu_seqlens_tuple)

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
                s, cu_seqlens, flatten_batch=self.flatten_batch
            )

        if attention_mask is None:
            if self.use_full_precision_softmax:
                raise RuntimeError("Empty attention mask")
        else:
            attention_mask = attention_mask.to(device=q.device)

        q, k, v = [rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in [q, k, v]]

        global first

        # [b, 1, s]
        if self.use_full_precision_softmax:
            scale = self.head_size**-0.5
            k_transposed = rearrange(k, "b h s d -> b h d s")
            attn_weights = torch.matmul(q, k_transposed) * scale
            del k, k_transposed
            # invert and multiply with min
            attention_mask = (~attention_mask) * torch.finfo(q.dtype).min
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
            if first:
                save_tensor(q, "q")
                save_tensor(k, "k")
                save_tensor(attention_mask, "attention_mask")
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout,
                is_causal=False,
            )

        # [b, h, s, head_size] --> [b * s, h, head_size]
        output = rearrange(output, "b h s d -> (b s) h d")
        first = False
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
