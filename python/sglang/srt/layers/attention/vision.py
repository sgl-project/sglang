from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
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
    """Multi-headed attention without any cache, mostly used for ViT."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        use_qkv_parallel: bool,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        world_size = parallel_state.get_tensor_model_parallel_world_size()

        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, world_size
        )
        # self.tp_size = get_tensor_model_parallel_world_size()
        # num_heads = self.num_heads_per_partition
        self.use_qkv_parallel = use_qkv_parallel
        if use_qkv_parallel:
            self.head_dim = embed_dim // num_heads
            self.qkv_proj = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=self.head_dim,
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
    ) -> torch.Tensor:
        """
        Input shape: [b, s, embed_dim]
        Output shape: [s, b, num_heads * head_size]
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
            # [s, b, embed_dim] --> [s, b, head * 3 * head_dim]
            qkv, _ = self.qkv_proj(x)
            # [s, b, head * 3 * head_dim] --> [s, b, head, 3 * head_dim]
            new_x_shape = qkv.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            qkv = qkv.view(*new_x_shape)

            # [s, b, head, 3 * head_dim] --> 3 [s, b, head, head_dim]
            q, k, v = dist_utils.split_tensor_along_last_dim(qkv, 3)

            # [s, b, head, head_dim] --> [b, s, head, head_dim]
            q, k, v = [
                rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)
            ]

        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        if self.use_qkv_parallel:
            pass
        else:
            # [b, s, head, head_dim] --> [b * s, head, head_dim]
            q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]

        # [b * s, num_heads, head_size]
        output = torch.empty_like(q)

        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).cuda()
        max_seqlen = seq_lens.max().item()

        context_attention_fwd(
            q,
            k,
            v,
            output,
            cu_seqlens.cuda(),
            seq_lens,
            max_seqlen,
            is_causal=False,
        )

        if self.use_qkv_parallel:

            # [b * s, head, head_dim] --> [b, s, head * head_dim]
            output = rearrange(output, "(b s) ... h d -> b s ... (h d)", b=bsz)

            # [b, s, head, head_dim] --> [b, s, head, head_dim]
            output, _ = self.proj(output)
        else:
            # [b * s, head, head_dim] --> [b, s, head, head_dim]
            context_layer = rearrange(output, "(b s) ... -> b s ...", b=bsz)

            # [s, b, num_heads * head_size]
            context_layer = rearrange(
                context_layer, "b s h d -> s b (h d)"
            ).contiguous()

            # [s, b, num_heads * head_size] --> [s, b, num_heads * head_size]
            output, _ = self.proj(context_layer)

            output = output.view(bsz, s, -1)

        return output
