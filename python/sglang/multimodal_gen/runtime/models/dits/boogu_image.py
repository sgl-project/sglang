# SPDX-License-Identifier: Apache-2.0

import math
from typing import List, Optional, Tuple

import msgspec
import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits.boogu_image import BooguImageDitConfig
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul
from sglang.multimodal_gen.runtime.layers.attention import (
    USPAttention,
    build_varlen_mask_meta_from_lengths,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    apply_qk_norm_with_optional_rope,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import _apply_rotary_emb
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

ADALN_EMBED_DIM = 1024
TIMESTEP_FREQ_DIM = 256
NUM_ADALN_MODULATION_PARAMS = 4


class BooguRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps
        self.hidden_size = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        out = x.float()
        out = out * torch.rsqrt(
            out.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        out = out.to(orig_dtype)
        return out * self.weight.to(device=x.device, dtype=orig_dtype)


def rmsnorm_tanh_mul_add(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    norm: BooguRMSNorm,
) -> torch.Tensor:
    return residual + torch.tanh(gate) * norm(x)


class FeedForward(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.w13 = MergedColumnParallelLinear(
            dim,
            [hidden_dim, hidden_dim],
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w13",
        )
        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )
        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x13, _ = self.w13(x)
        out, _ = self.w2(self.act(x13))
        return out


def compute_ffn_hidden_dim(
    dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]
) -> int:
    inner_dim = 4 * dim
    if ffn_dim_multiplier is not None:
        inner_dim = int(ffn_dim_multiplier * inner_dim)
    return multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)


def interleave_instruct_image(
    instruct: torch.Tensor,
    img: torch.Tensor,
    encoder_seq_lengths: List[int],
    seq_lengths: List[int],
) -> torch.Tensor:
    max_seq_len = max(seq_lengths)
    packed = [
        torch.cat(
            [
                instruct[i, :enc_len],
                img[i, : seq_len - enc_len],
                instruct.new_zeros(max_seq_len - seq_len, instruct.shape[-1]),
            ]
        )
        for i, (enc_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths))
    ]
    return torch.stack(packed)


def split_instruct_image(
    joint: torch.Tensor,
    encoder_seq_lengths: List[int],
    seq_lengths: List[int],
    img_seq_len: int,
    instruct_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    instruct = joint.new_zeros(joint.shape[0], instruct_seq_len, joint.shape[-1])
    img = joint.new_zeros(joint.shape[0], img_seq_len, joint.shape[-1])
    instruct_rows, img_rows = [], []
    for i, (enc_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
        instruct_rows.append(
            torch.cat(
                [joint[i, :enc_len], instruct[i, enc_len:]],
            )
        )
        img_rows.append(
            torch.cat(
                [joint[i, enc_len:seq_len], img[i, seq_len - enc_len :]],
            )
        )
    return torch.stack(instruct_rows), torch.stack(img_rows)


def apply_rope_per_sample(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs_cis
    batch_size, seq_len = q.shape[:2]
    if cos.dim() == 3:
        cos = cos.reshape(batch_size * seq_len, -1)
        sin = sin.reshape(batch_size * seq_len, -1)
    q_flat = q.reshape(batch_size * seq_len, *q.shape[2:])
    k_flat = k.reshape(batch_size * seq_len, *k.shape[2:])
    q_out = _apply_rotary_emb(q_flat, cos, sin, is_neox_style=False)
    k_out = _apply_rotary_emb(k_flat, cos, sin, is_neox_style=False)
    return q_out.view_as(q), k_out.view_as(k)


class BooguAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-5,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qk_norm = qk_norm

        tp_size = get_tp_world_size()
        if num_heads % tp_size != 0 or num_kv_heads % tp_size != 0:
            raise ValueError(
                f"num_heads ({num_heads}) and num_kv_heads ({num_kv_heads}) must "
                f"both be divisible by the tp world size ({tp_size})"
            )
        self.local_num_heads = num_heads // tp_size
        self.local_num_kv_heads = num_kv_heads // tp_size
        kv_dim = self.head_dim * num_kv_heads

        self.to_q = ColumnParallelLinear(
            dim,
            dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_q",
        )
        self.to_k = ColumnParallelLinear(
            dim,
            kv_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_k",
        )
        self.to_v = ColumnParallelLinear(
            dim,
            kv_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_v",
        )

        if self.qk_norm:
            self.norm_q = BooguRMSNorm(self.head_dim, eps=eps)
            self.norm_k = BooguRMSNorm(self.head_dim, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None

        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    dim,
                    dim,
                    bias=False,
                    input_is_parallel=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.to_out.0",
                )
            ]
        )

        self._padded_head_dim = 128  # Boogu head_dim=120 → next FA-supported bucket
        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self._padded_head_dim,
            num_kv_heads=self.local_num_kv_heads,
            dropout_rate=0,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
        )

    def _qkv(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, _ = self.to_q(hidden_states)
        k, _ = self.to_k(hidden_states)
        v, _ = self.to_v(hidden_states)
        return self._split_heads(q, k, v)

    def _split_heads(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = q.view(*q.shape[:-1], self.local_num_heads, self.head_dim)
        k = k.view(*k.shape[:-1], self.local_num_kv_heads, self.head_dim)
        v = v.view(*v.shape[:-1], self.local_num_kv_heads, self.head_dim)
        return q, k, v

    def _norm_and_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.qk_norm:
            q, k = apply_qk_norm_with_optional_rope(
                q=q,
                k=k,
                q_norm=self.norm_q,
                k_norm=self.norm_k,
                head_dim=self.head_dim,
                allow_inplace=False,
            )
        if freqs_cis is not None:
            q, k = apply_rope_per_sample(q, k, freqs_cis)
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask_meta: Optional[dict] = None,
    ) -> torch.Tensor:
        q, k, v = self._qkv(hidden_states)
        q, k = self._norm_and_rope(q, k, freqs_cis)
        pad = self._padded_head_dim - self.head_dim
        if pad:
            q = nn.functional.pad(q, (0, pad))
            k = nn.functional.pad(k, (0, pad))
            v = nn.functional.pad(v, (0, pad))
        out = self.attn(q, k, v, attn_mask_meta=attn_mask_meta)
        if pad:
            out = out[..., : self.head_dim]
        out, _ = self.to_out[0](out.flatten(2))
        return out


class BooguJointAttnProjections(nn.Module):

    def __init__(
        self,
        dim: int,
        kv_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        for name, out_dim in (
            ("img_to_q", dim),
            ("img_to_k", kv_dim),
            ("img_to_v", kv_dim),
            ("instruct_to_q", dim),
            ("instruct_to_k", kv_dim),
            ("instruct_to_v", kv_dim),
        ):
            setattr(
                self,
                name,
                ColumnParallelLinear(
                    dim,
                    out_dim,
                    bias=False,
                    gather_output=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.{name}",
                ),
            )
        self.img_out = RowParallelLinear(
            dim,
            dim,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.img_out",
        )
        self.instruct_out = RowParallelLinear(
            dim,
            dim,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.instruct_out",
        )


class BooguJointAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-5,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm

        tp_size = get_tp_world_size()
        if num_heads % tp_size != 0 or num_kv_heads % tp_size != 0:
            raise ValueError(
                f"num_heads ({num_heads}) and num_kv_heads ({num_kv_heads}) must "
                f"both be divisible by the tp world size ({tp_size})"
            )
        self.local_num_heads = num_heads // tp_size
        self.local_num_kv_heads = num_kv_heads // tp_size

        self.processor = BooguJointAttnProjections(
            dim=dim,
            kv_dim=self.head_dim * num_kv_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.processor",
        )

        if self.qk_norm:
            self.norm_q = BooguRMSNorm(self.head_dim, eps=eps)
            self.norm_k = BooguRMSNorm(self.head_dim, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None

        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    dim,
                    dim,
                    bias=False,
                    input_is_parallel=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.to_out.0",
                )
            ]
        )

        self._padded_head_dim = 128  # Boogu head_dim=120 → next FA-supported bucket
        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self._padded_head_dim,
            num_kv_heads=self.local_num_kv_heads,
            dropout_rate=0,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
        )

    def forward(
        self,
        img_hidden_states: torch.Tensor,
        instruct_hidden_states: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        encoder_seq_lengths: List[int],
        seq_lengths: List[int],
        attn_mask_meta: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.processor
        img_q, _ = p.img_to_q(img_hidden_states)
        img_k, _ = p.img_to_k(img_hidden_states)
        img_v, _ = p.img_to_v(img_hidden_states)
        instruct_q, _ = p.instruct_to_q(instruct_hidden_states)
        instruct_k, _ = p.instruct_to_k(instruct_hidden_states)
        instruct_v, _ = p.instruct_to_v(instruct_hidden_states)

        query = interleave_instruct_image(
            instruct=instruct_q,
            img=img_q,
            encoder_seq_lengths=encoder_seq_lengths,
            seq_lengths=seq_lengths,
        )
        key = interleave_instruct_image(
            instruct=instruct_k,
            img=img_k,
            encoder_seq_lengths=encoder_seq_lengths,
            seq_lengths=seq_lengths,
        )
        value = interleave_instruct_image(
            instruct=instruct_v,
            img=img_v,
            encoder_seq_lengths=encoder_seq_lengths,
            seq_lengths=seq_lengths,
        )

        query = query.view(*query.shape[:-1], self.local_num_heads, self.head_dim)
        key = key.view(*key.shape[:-1], self.local_num_kv_heads, self.head_dim)
        value = value.view(*value.shape[:-1], self.local_num_kv_heads, self.head_dim)

        if self.qk_norm:
            query, key = apply_qk_norm_with_optional_rope(
                q=query,
                k=key,
                q_norm=self.norm_q,
                k_norm=self.norm_k,
                head_dim=self.head_dim,
                allow_inplace=False,
            )
        query, key = apply_rope_per_sample(query, key, freqs_cis)

        pad = self._padded_head_dim - self.head_dim
        if pad:
            query = nn.functional.pad(query, (0, pad))
            key = nn.functional.pad(key, (0, pad))
            value = nn.functional.pad(value, (0, pad))
        joint = self.attn(query, key, value, attn_mask_meta=attn_mask_meta)
        if pad:
            joint = joint[..., : self.head_dim]
        joint = joint.flatten(2)

        instruct_out, img_out = split_instruct_image(
            joint,
            encoder_seq_lengths,
            seq_lengths,
            img_seq_len=img_hidden_states.shape[1],
            instruct_seq_len=instruct_hidden_states.shape[1],
        )
        instruct_out, _ = p.instruct_out(instruct_out)
        img_out, _ = p.img_out(img_out)

        joint = interleave_instruct_image(
            instruct=instruct_out,
            img=img_out,
            encoder_seq_lengths=encoder_seq_lengths,
            seq_lengths=seq_lengths,
        )
        joint, _ = self.to_out[0](joint)
        return split_instruct_image(
            joint,
            encoder_seq_lengths,
            seq_lengths,
            img_seq_len=img_hidden_states.shape[1],
            instruct_seq_len=instruct_hidden_states.shape[1],
        )


class BooguRMSNormZero(nn.Module):

    def __init__(self, embedding_dim: int, norm_eps: float):
        super().__init__()
        self.linear = ReplicatedLinear(
            min(embedding_dim, ADALN_EMBED_DIM),
            NUM_ADALN_MODULATION_PARAMS * embedding_dim,
            bias=True,
        )
        self.norm = BooguRMSNorm(embedding_dim, eps=norm_eps)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        modulation, _ = self.linear(torch.nn.functional.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = modulation.chunk(
            NUM_ADALN_MODULATION_PARAMS, dim=1
        )
        x = self.norm(x) * (1 + scale_msa[:, None])
        return x, gate_msa, scale_mlp, gate_mlp


class BooguLayerNormContinuous(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        out_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.linear_1 = ReplicatedLinear(
            conditioning_embedding_dim, embedding_dim, bias=True
        )
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=False)
        self.linear_2 = ReplicatedLinear(embedding_dim, out_dim, bias=True)

    def forward(
        self, x: torch.Tensor, conditioning_embedding: torch.Tensor
    ) -> torch.Tensor:
        scale, _ = self.linear_1(
            torch.nn.functional.silu(conditioning_embedding).to(x.dtype)
        )
        x = self.norm(x) * (1 + scale)[:, None, :]
        out, _ = self.linear_2(x)
        return out


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size: int, freq_dim: int = TIMESTEP_FREQ_DIM):
        super().__init__()
        self.freq_dim = freq_dim
        self.linear_1 = ReplicatedLinear(freq_dim, hidden_size, bias=True)
        self.linear_2 = ReplicatedLinear(hidden_size, hidden_size, bias=True)

    def _timestep_features(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        features = self._timestep_features(t).to(dtype)
        hidden, _ = self.linear_1(features)
        out, _ = self.linear_2(torch.nn.functional.silu(hidden))
        return out


class BooguTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.modulation = modulation
        self.attn = BooguAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qk_norm=qk_norm,
            eps=norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=compute_ffn_hidden_dim(
                dim=dim,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
            ),
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
        )
        if modulation:
            self.norm1 = BooguRMSNormZero(dim, norm_eps)
        else:
            self.norm1 = BooguRMSNorm(dim, eps=norm_eps)
        self.norm2 = BooguRMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = BooguRMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = BooguRMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        temb: Optional[torch.Tensor] = None,
        attn_mask_meta: Optional[dict] = None,
    ) -> torch.Tensor:
        if self.modulation:
            normed, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
            attn_out = self.attn(
                hidden_states=normed,
                freqs_cis=freqs_cis,
                attn_mask_meta=attn_mask_meta,
            )
            hidden_states = rmsnorm_tanh_mul_add(
                attn_out, gate_msa.unsqueeze(1), hidden_states, self.norm2
            )
            mlp_out = self.feed_forward(
                self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1))
            )
            return rmsnorm_tanh_mul_add(
                mlp_out, gate_mlp.unsqueeze(1), hidden_states, self.ffn_norm2
            )

        attn_out = self.attn(
            hidden_states=self.norm1(hidden_states),
            freqs_cis=freqs_cis,
            attn_mask_meta=attn_mask_meta,
        )
        hidden_states = hidden_states + self.norm2(attn_out)
        mlp_out = self.feed_forward(self.ffn_norm1(hidden_states))
        return hidden_states + self.ffn_norm2(mlp_out)


class BooguDoubleStreamBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        norm_eps: float,
        qk_norm: bool,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_dim = compute_ffn_hidden_dim(
            dim=dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        self.img_instruct_attn = BooguJointAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qk_norm=qk_norm,
            eps=norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.img_instruct_attn",
        )
        self.img_self_attn = BooguAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qk_norm=qk_norm,
            eps=norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.img_self_attn",
        )
        self.img_feed_forward = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.img_feed_forward",
        )
        self.instruct_feed_forward = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.instruct_feed_forward",
        )

        self.img_norm1 = BooguRMSNormZero(dim, norm_eps)
        self.img_norm2 = BooguRMSNormZero(dim, norm_eps)
        self.img_norm3 = BooguRMSNormZero(dim, norm_eps)
        self.instruct_norm1 = BooguRMSNormZero(dim, norm_eps)
        self.instruct_norm2 = BooguRMSNormZero(dim, norm_eps)

        self.img_attn_norm = BooguRMSNorm(dim, eps=norm_eps)
        self.img_self_attn_norm = BooguRMSNorm(dim, eps=norm_eps)
        self.img_ffn_norm1 = BooguRMSNorm(dim, eps=norm_eps)
        self.img_ffn_norm2 = BooguRMSNorm(dim, eps=norm_eps)
        self.instruct_attn_norm = BooguRMSNorm(dim, eps=norm_eps)
        self.instruct_ffn_norm1 = BooguRMSNorm(dim, eps=norm_eps)
        self.instruct_ffn_norm2 = BooguRMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        img_hidden_states: torch.Tensor,
        instruct_hidden_states: torch.Tensor,
        img_freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        joint_freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        temb: torch.Tensor,
        encoder_seq_lengths: List[int],
        seq_lengths: List[int],
        img_attn_mask_meta: Optional[dict] = None,
        joint_attn_mask_meta: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp = self.img_norm1(
            img_hidden_states, temb
        )
        img_norm2_out, img_shift_mlp, _, _ = self.img_norm2(img_hidden_states, temb)
        img_norm3_out, img_gate_self, _, _ = self.img_norm3(img_hidden_states, temb)
        (
            instruct_norm1_out,
            instruct_gate_msa,
            instruct_scale_mlp,
            instruct_gate_mlp,
        ) = self.instruct_norm1(instruct_hidden_states, temb)
        instruct_norm2_out, instruct_shift_mlp, _, _ = self.instruct_norm2(
            instruct_hidden_states, temb
        )

        instruct_attn_out, img_attn_out = self.img_instruct_attn(
            img_hidden_states=img_norm1_out,
            instruct_hidden_states=instruct_norm1_out,
            freqs_cis=joint_freqs_cis,
            encoder_seq_lengths=encoder_seq_lengths,
            seq_lengths=seq_lengths,
            attn_mask_meta=joint_attn_mask_meta,
        )
        img_self_attn_out = self.img_self_attn(
            hidden_states=img_norm3_out,
            freqs_cis=img_freqs_cis,
            attn_mask_meta=img_attn_mask_meta,
        )

        img_hidden_states = rmsnorm_tanh_mul_add(
            img_attn_out,
            img_gate_msa.unsqueeze(1),
            img_hidden_states,
            self.img_attn_norm,
        )
        img_hidden_states = rmsnorm_tanh_mul_add(
            img_self_attn_out,
            img_gate_self.unsqueeze(1),
            img_hidden_states,
            self.img_self_attn_norm,
        )
        img_mlp_input = (1 + img_scale_mlp.unsqueeze(1)) * img_norm2_out + (
            img_shift_mlp.unsqueeze(1)
        )
        img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_mlp_input))
        img_hidden_states = rmsnorm_tanh_mul_add(
            img_mlp_out,
            img_gate_mlp.unsqueeze(1),
            img_hidden_states,
            self.img_ffn_norm2,
        )

        instruct_hidden_states = rmsnorm_tanh_mul_add(
            instruct_attn_out,
            instruct_gate_msa.unsqueeze(1),
            instruct_hidden_states,
            self.instruct_attn_norm,
        )
        instruct_mlp_input = (
            1 + instruct_scale_mlp.unsqueeze(1)
        ) * instruct_norm2_out + instruct_shift_mlp.unsqueeze(1)
        instruct_mlp_out = self.instruct_feed_forward(
            self.instruct_ffn_norm1(instruct_mlp_input)
        )
        instruct_hidden_states = rmsnorm_tanh_mul_add(
            instruct_mlp_out,
            instruct_gate_mlp.unsqueeze(1),
            instruct_hidden_states,
            self.instruct_ffn_norm2,
        )
        return img_hidden_states, instruct_hidden_states


class BooguRopeBundle(msgspec.Struct, frozen=True):

    joint: Tuple[torch.Tensor, torch.Tensor]
    context: Tuple[torch.Tensor, torch.Tensor]
    noise: Tuple[torch.Tensor, torch.Tensor]
    combined_img: Tuple[torch.Tensor, torch.Tensor]
    instruction_seq_lengths: List[int]
    seq_lengths: List[int]
    combined_img_seq_lengths: List[int]


class BooguRopeEmbedder(nn.Module):

    def __init__(
        self,
        theta: float,
        axes_dims: Tuple[int, int, int],
        axes_lens: Tuple[int, int, int],
        patch_size: int,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.patch_size = patch_size
        self.num_pos_axes = len(axes_dims)
        self._tables: Optional[List[torch.Tensor]] = None

    def _axis_tables(self, device: torch.device) -> List[torch.Tensor]:
        if self._tables is None or self._tables[0].device != device:
            from diffusers.models.embeddings import get_1d_rotary_pos_embed

            self._tables = [
                get_1d_rotary_pos_embed(
                    dim, length, theta=self.theta, freqs_dtype=torch.float64
                ).to(device)
                for dim, length in zip(self.axes_dims, self.axes_lens)
            ]
        return self._tables

    def _gather(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tables = self._axis_tables(position_ids.device)
        parts = []
        for axis, table in enumerate(tables):
            index = (
                position_ids[:, :, axis : axis + 1]
                .repeat(1, 1, table.shape[-1])
                .to(torch.int64)
            )
            parts.append(
                torch.gather(
                    table.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index
                )
            )
        freqs = torch.cat(parts, dim=-1)
        return freqs.real.float(), freqs.imag.float()

    def _build_position_ids(
        self,
        instruction_seq_lengths: List[int],
        ref_img_lengths: List[List[int]],
        img_lengths: List[int],
        ref_img_sizes: List[Optional[List[Tuple[int, int]]]],
        img_sizes: List[Tuple[int, int]],
        seq_lengths: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = len(seq_lengths)
        position_ids = torch.zeros(
            batch_size,
            max(seq_lengths),
            self.num_pos_axes,
            dtype=torch.int32,
            device=device,
        )
        for i, cap_len in enumerate(instruction_seq_lengths):
            position_ids[i, :cap_len] = (
                torch.arange(cap_len, dtype=torch.int32, device=device)
                .unsqueeze(-1)
                .repeat(1, self.num_pos_axes)
            )
            pe_shift = cap_len
            pe_shift_len = cap_len
            if ref_img_sizes[i] is not None:
                for (height, width), ref_len in zip(
                    ref_img_sizes[i], ref_img_lengths[i]
                ):
                    rows, cols = self._token_grid(height, width, device)
                    end = pe_shift_len + ref_len
                    position_ids[i, pe_shift_len:end, 0] = pe_shift
                    position_ids[i, pe_shift_len:end, 1] = rows
                    position_ids[i, pe_shift_len:end, 2] = cols
                    pe_shift += max(height // self.patch_size, width // self.patch_size)
                    pe_shift_len += ref_len

            height, width = img_sizes[i]
            rows, cols = self._token_grid(height, width, device)
            start = seq_lengths[i] - img_lengths[i]
            position_ids[i, start : seq_lengths[i], 0] = pe_shift
            position_ids[i, start : seq_lengths[i], 1] = rows
            position_ids[i, start : seq_lengths[i], 2] = cols
        return position_ids

    def _token_grid(
        self, height: int, width: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_tokens = height // self.patch_size
        w_tokens = width // self.patch_size
        rows = torch.arange(h_tokens, dtype=torch.int32, device=device)
        cols = torch.arange(w_tokens, dtype=torch.int32, device=device)
        return (
            rows.view(-1, 1).repeat(1, w_tokens).flatten(),
            cols.view(1, -1).repeat(h_tokens, 1).flatten(),
        )

    @torch.compiler.disable
    def forward(
        self,
        instruction_attention_mask: torch.Tensor,
        ref_img_lengths: List[List[int]],
        img_lengths: List[int],
        ref_img_sizes: List[Optional[List[Tuple[int, int]]]],
        img_sizes: List[Tuple[int, int]],
        device: torch.device,
    ) -> "BooguRopeBundle":
        instruction_seq_lengths = instruction_attention_mask.sum(dim=1).tolist()
        seq_lengths = [
            cap_len + sum(ref_lens) + img_len
            for cap_len, ref_lens, img_len in zip(
                instruction_seq_lengths, ref_img_lengths, img_lengths
            )
        ]
        combined_img_seq_lengths = [
            sum(ref_lens) + img_len
            for ref_lens, img_len in zip(ref_img_lengths, img_lengths)
        ]

        position_ids = self._build_position_ids(
            instruction_seq_lengths=instruction_seq_lengths,
            ref_img_lengths=ref_img_lengths,
            img_lengths=img_lengths,
            ref_img_sizes=ref_img_sizes,
            img_sizes=img_sizes,
            seq_lengths=seq_lengths,
            device=device,
        )
        joint = self._gather(position_ids)
        zeros = [0] * len(seq_lengths)
        return BooguRopeBundle(
            joint=joint,
            context=_slice_freqs(
                freqs=joint, starts=zeros, lengths=instruction_seq_lengths
            ),
            noise=_slice_freqs(
                freqs=joint,
                starts=[seq - img for seq, img in zip(seq_lengths, img_lengths)],
                lengths=img_lengths,
            ),
            combined_img=_slice_freqs(
                freqs=joint,
                starts=instruction_seq_lengths,
                lengths=combined_img_seq_lengths,
            ),
            instruction_seq_lengths=instruction_seq_lengths,
            seq_lengths=seq_lengths,
            combined_img_seq_lengths=combined_img_seq_lengths,
        )


def _slice_freqs(
    freqs: Tuple[torch.Tensor, torch.Tensor],
    starts: List[int],
    lengths: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs
    max_len = max(lengths)
    out_cos = cos.new_ones(cos.shape[0], max_len, cos.shape[-1])
    out_sin = sin.new_zeros(sin.shape[0], max_len, sin.shape[-1])
    for i, (start, length) in enumerate(zip(starts, lengths)):
        out_cos[i, :length] = cos[i, start : start + length]
        out_sin[i, :length] = sin[i, start : start + length]
    return out_cos, out_sin


class BooguCombinedTimestepCaptionEmbedding(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        instruction_feat_dim: int,
        norm_eps: float,
        timestep_scale: float,
    ):
        super().__init__()
        self.timestep_scale = timestep_scale
        self.timestep_embedder = TimestepEmbedder(
            min(hidden_size, ADALN_EMBED_DIM), freq_dim=TIMESTEP_FREQ_DIM
        )
        self.caption_embedder = nn.Sequential(
            BooguRMSNorm(instruction_feat_dim, eps=norm_eps),
            ReplicatedLinear(instruction_feat_dim, hidden_size, bias=True),
        )

    def forward(
        self,
        timestep: torch.Tensor,
        instruction_hidden_states: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temb = self.timestep_embedder(timestep * self.timestep_scale, dtype)
        normed = self.caption_embedder[0](instruction_hidden_states)
        caption, _ = self.caption_embedder[1](normed)
        return temb, caption


def patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    channels, height, width = image.shape
    h_tokens = height // patch_size
    w_tokens = width // patch_size
    tokens = image.view(channels, h_tokens, patch_size, w_tokens, patch_size)
    tokens = tokens.permute(1, 3, 2, 4, 0)
    return tokens.reshape(h_tokens * w_tokens, patch_size * patch_size * channels)


def unpatchify(
    tokens: torch.Tensor, height: int, width: int, patch_size: int, channels: int
) -> torch.Tensor:
    h_tokens = height // patch_size
    w_tokens = width // patch_size
    out = tokens.view(h_tokens, w_tokens, patch_size, patch_size, channels)
    out = out.permute(4, 0, 2, 1, 3)
    return out.reshape(channels, h_tokens * patch_size, w_tokens * patch_size)


def flat_and_pad_to_seq(
    hidden_states: List[torch.Tensor],
    ref_image_hidden_states: Optional[List[List[torch.Tensor]]],
    patch_size: int,
) -> Tuple[
    torch.Tensor,
    Optional[List[List[torch.Tensor]]],
    List[int],
    List[List[int]],
    List[Optional[List[Tuple[int, int]]]],
    List[Tuple[int, int]],
]:
    img_sizes = [(img.shape[-2], img.shape[-1]) for img in hidden_states]
    flat = [patchify(img, patch_size) for img in hidden_states]
    img_lengths = [tokens.shape[0] for tokens in flat]
    max_len = max(img_lengths)
    padded = torch.stack(
        [
            torch.cat(
                [tokens, tokens.new_zeros(max_len - tokens.shape[0], tokens.shape[-1])]
            )
            for tokens in flat
        ]
    )

    if ref_image_hidden_states is None:
        return (
            padded,
            None,
            img_lengths,
            [[0]] * len(hidden_states),
            [None] * len(hidden_states),
            img_sizes,
        )

    ref_sizes: List[Optional[List[Tuple[int, int]]]] = []
    ref_lengths: List[List[int]] = []
    ref_flat: List[List[torch.Tensor]] = []
    for refs in ref_image_hidden_states:
        ref_sizes.append([(ref.shape[-2], ref.shape[-1]) for ref in refs])
        per_sample = [patchify(ref, patch_size) for ref in refs]
        ref_flat.append(per_sample)
        ref_lengths.append([tokens.shape[0] for tokens in per_sample])
    return padded, ref_flat, img_lengths, ref_lengths, ref_sizes, img_sizes


class BooguImageTransformer2DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["BooguTransformerBlock", "BooguDoubleStreamBlock"]
    _fsdp_shard_conditions = BooguImageDitConfig().arch_config._fsdp_shard_conditions
    param_names_mapping = BooguImageDitConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = (
        BooguImageDitConfig().arch_config.reverse_param_names_mapping
    )
    packed_modules_mapping = {
        "w13": ["linear_1", "linear_3"],
    }

    def __init__(
        self,
        config: BooguImageDitConfig,
        hf_config: dict,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch_config = config.arch_config
        self.config_data = config
        self.in_channels = arch_config.in_channels
        self.out_channels = arch_config.out_channels
        self.dim = arch_config.dim
        self.patch_size = arch_config.all_patch_size[0]
        self.num_instruction_feature_layers = arch_config.num_instruction_feature_layers
        self.instruction_reduce_type = arch_config.instruction_reduce_type
        self.gradient_checkpointing = False

        patch_dim = self.patch_size * self.patch_size * self.in_channels
        self.x_embedder = ReplicatedLinear(patch_dim, self.dim, bias=True)
        self.ref_image_patch_embedder = ReplicatedLinear(patch_dim, self.dim, bias=True)
        self.image_index_embedding = nn.Parameter(
            torch.zeros(arch_config.max_ref_images, self.dim)
        )

        self.time_caption_embed = BooguCombinedTimestepCaptionEmbedding(
            hidden_size=self.dim,
            instruction_feat_dim=arch_config.preprocessed_cap_feat_dim,
            norm_eps=arch_config.norm_eps,
            timestep_scale=arch_config.t_scale,
        )

        block_kwargs = dict(
            dim=self.dim,
            num_heads=arch_config.num_attention_heads,
            num_kv_heads=arch_config.n_kv_heads,
            multiple_of=arch_config.multiple_of,
            ffn_dim_multiplier=arch_config.ffn_dim_multiplier,
            norm_eps=arch_config.norm_eps,
            qk_norm=arch_config.qk_norm,
            quant_config=quant_config,
        )

        def refiner_group(name: str, modulation: bool) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    BooguTransformerBlock(
                        modulation=modulation,
                        prefix=f"{name}.{layer_id}",
                        **block_kwargs,
                    )
                    for layer_id in range(arch_config.n_refiner_layers)
                ]
            )

        self.noise_refiner = refiner_group("noise_refiner", modulation=True)
        self.ref_image_refiner = refiner_group("ref_image_refiner", modulation=True)
        self.context_refiner = refiner_group("context_refiner", modulation=False)

        self.double_stream_layers = nn.ModuleList(
            [
                BooguDoubleStreamBlock(
                    prefix=f"double_stream_layers.{layer_id}", **block_kwargs
                )
                for layer_id in range(arch_config.num_double_stream_layers)
            ]
        )
        self.single_stream_layers = nn.ModuleList(
            [
                BooguTransformerBlock(
                    modulation=True,
                    prefix=f"single_stream_layers.{layer_id}",
                    **block_kwargs,
                )
                for layer_id in range(arch_config.num_single_stream_layers)
            ]
        )

        self.norm_out = BooguLayerNormContinuous(
            embedding_dim=self.dim,
            conditioning_embedding_dim=min(self.dim, ADALN_EMBED_DIM),
            out_dim=self.patch_size * self.patch_size * self.out_channels,
        )
        self.rope_embedder = BooguRopeEmbedder(
            theta=arch_config.rope_theta,
            axes_dims=arch_config.axes_dims,
            axes_lens=arch_config.axes_lens,
            patch_size=self.patch_size,
        )

    def _reduce_instruction_features(
        self, instruction_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if instruction_hidden_states.dim() == 3:
            return instruction_hidden_states
        if self.instruction_reduce_type == "mean":
            return instruction_hidden_states.mean(dim=0)
        return torch.cat(list(instruction_hidden_states), dim=-1)

    def _embed_and_refine_images(
        self,
        noise_tokens: torch.Tensor,
        ref_tokens: Optional[List[List[torch.Tensor]]],
        img_lengths: List[int],
        ref_lengths: List[List[int]],
        rope: BooguRopeBundle,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        noise_hidden_states, _ = self.x_embedder(noise_tokens)
        noise_mask_meta = build_varlen_mask_meta_from_lengths(
            img_lengths, noise_hidden_states.shape[1], noise_hidden_states.device
        )
        for layer in self.noise_refiner:
            noise_hidden_states = layer(
                hidden_states=noise_hidden_states,
                freqs_cis=rope.noise,
                temb=temb,
                attn_mask_meta=noise_mask_meta,
            )

        if ref_tokens is None:
            return noise_hidden_states

        flat_refs, flat_temb, flat_lengths = [], [], []
        for sample_idx, per_sample in enumerate(ref_tokens):
            for ref_idx, tokens in enumerate(per_sample):
                embedded, _ = self.ref_image_patch_embedder(tokens)
                flat_refs.append(embedded + self.image_index_embedding[ref_idx])
                flat_temb.append(temb[sample_idx])
                flat_lengths.append(tokens.shape[0])
        max_ref_len = max(flat_lengths)
        ref_hidden_states = torch.stack(
            [
                torch.cat(
                    [tokens, tokens.new_zeros(max_ref_len - tokens.shape[0], self.dim)]
                )
                for tokens in flat_refs
            ]
        )
        ref_freqs = _slice_freqs(
            freqs=rope.combined_img,
            starts=[0] * len(flat_lengths),
            lengths=flat_lengths,
        )
        ref_mask_meta = build_varlen_mask_meta_from_lengths(
            flat_lengths, max_ref_len, ref_hidden_states.device
        )
        for layer in self.ref_image_refiner:
            ref_hidden_states = layer(
                hidden_states=ref_hidden_states,
                freqs_cis=ref_freqs,
                temb=torch.stack(flat_temb),
                attn_mask_meta=ref_mask_meta,
            )

        combined = []
        cursor = 0
        max_combined = max(rope.combined_img_seq_lengths)
        for sample_idx, per_sample_lengths in enumerate(ref_lengths):
            parts = []
            for length in per_sample_lengths:
                parts.append(ref_hidden_states[cursor, :length])
                cursor += 1
            parts.append(noise_hidden_states[sample_idx, : img_lengths[sample_idx]])
            packed = torch.cat(parts)
            combined.append(
                torch.cat(
                    [
                        packed,
                        packed.new_zeros(max_combined - packed.shape[0], self.dim),
                    ]
                )
            )
        return torch.stack(combined)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        instruction_hidden_states: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
        ref_image_hidden_states: Optional[List[List[torch.Tensor]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        squeezed_frame_axis = hidden_states.dim() == 5
        if squeezed_frame_axis:
            hidden_states = hidden_states.squeeze(2)

        device = hidden_states.device
        dtype = hidden_states.dtype

        instruction_hidden_states = self._reduce_instruction_features(
            instruction_hidden_states
        )
        temb, instruct_hidden_states = self.time_caption_embed(
            timestep=timestep,
            instruction_hidden_states=instruction_hidden_states,
            dtype=dtype,
        )

        latents = list(hidden_states)
        (
            noise_tokens,
            ref_tokens,
            img_lengths,
            ref_lengths,
            ref_sizes,
            img_sizes,
        ) = flat_and_pad_to_seq(latents, ref_image_hidden_states, self.patch_size)

        rope = self.rope_embedder(
            instruction_attention_mask=instruction_attention_mask,
            ref_img_lengths=ref_lengths,
            img_lengths=img_lengths,
            ref_img_sizes=ref_sizes,
            img_sizes=img_sizes,
            device=device,
        )

        context_mask_meta = build_varlen_mask_meta_from_lengths(
            rope.instruction_seq_lengths, instruct_hidden_states.shape[1], device
        )
        for layer in self.context_refiner:
            instruct_hidden_states = layer(
                hidden_states=instruct_hidden_states,
                freqs_cis=rope.context,
                attn_mask_meta=context_mask_meta,
            )

        img_hidden_states = self._embed_and_refine_images(
            noise_tokens=noise_tokens,
            ref_tokens=ref_tokens,
            img_lengths=img_lengths,
            ref_lengths=ref_lengths,
            rope=rope,
            temb=temb,
        )

        img_mask_meta = build_varlen_mask_meta_from_lengths(
            rope.combined_img_seq_lengths, img_hidden_states.shape[1], device
        )
        joint_mask_meta = build_varlen_mask_meta_from_lengths(
            rope.seq_lengths, max(rope.seq_lengths), device
        )
        for layer in self.double_stream_layers:
            img_hidden_states, instruct_hidden_states = layer(
                img_hidden_states=img_hidden_states,
                instruct_hidden_states=instruct_hidden_states,
                img_freqs_cis=rope.combined_img,
                joint_freqs_cis=rope.joint,
                temb=temb,
                encoder_seq_lengths=rope.instruction_seq_lengths,
                seq_lengths=rope.seq_lengths,
                img_attn_mask_meta=img_mask_meta,
                joint_attn_mask_meta=joint_mask_meta,
            )

        joint_hidden_states = interleave_instruct_image(
            instruct=instruct_hidden_states,
            img=img_hidden_states,
            encoder_seq_lengths=rope.instruction_seq_lengths,
            seq_lengths=rope.seq_lengths,
        )
        for layer in self.single_stream_layers:
            joint_hidden_states = layer(
                hidden_states=joint_hidden_states,
                freqs_cis=rope.joint,
                temb=temb,
                attn_mask_meta=joint_mask_meta,
            )

        joint_hidden_states = self.norm_out(joint_hidden_states, temb)
        out = torch.stack(
            [
                unpatchify(
                    joint_hidden_states[i, seq_len - img_len : seq_len],
                    height=img_sizes[i][0],
                    width=img_sizes[i][1],
                    patch_size=self.patch_size,
                    channels=self.out_channels,
                )
                for i, (seq_len, img_len) in enumerate(
                    zip(rope.seq_lengths, img_lengths)
                )
            ]
        )
        if squeezed_frame_axis:
            out = out.unsqueeze(2)
        return out


EntryClass = BooguImageTransformer2DModel
