import dataclasses
import enum
from typing import Optional, Tuple

import torch

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


class FP8KVCacheLayout(enum.Enum):
    MODEL1_FP8Sparse = 2

    def get_meta(self) -> Tuple[int, int, int, int, int]:
        assert self is FP8KVCacheLayout.MODEL1_FP8Sparse
        return (512, 448, 64, 64, 7)


@dataclasses.dataclass
class ExtraDecodeParams:
    b: int


@dataclasses.dataclass
class SparseDecodeParams:
    s_q: int
    h_q: int
    h_kv: int
    d_qk: int
    d_v: int
    decode: ExtraDecodeParams


@dataclasses.dataclass
class KVScope:
    blocked_k: torch.Tensor
    indices_in_kvcache: torch.Tensor
    topk_length: Optional[torch.Tensor]


@dataclasses.dataclass
class SparseDecodeInputs:
    q: torch.Tensor
    attn_sink: Optional[torch.Tensor]
    sm_scale: float
    kv_scope: KVScope
    extra_kv_scope: Optional[KVScope]


def dequantize_k_cache(
    quant_k_cache: torch.Tensor,
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    assert quant_k_cache.dtype == FP8_DTYPE
    d, d_nope, d_rope, _tile_size, _num_tiles = kvcache_layout.get_meta()
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1

    result = torch.empty(
        (num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device
    )
    quant_k_cache = quant_k_cache.view(num_blocks, -1)
    input_nope_rope = quant_k_cache[:, : block_size * (d_nope + 2 * d_rope)].view(
        num_blocks, block_size, d_nope + 2 * d_rope
    )
    input_nope = input_nope_rope[:, :, :d_nope]
    input_rope = input_nope_rope[:, :, d_nope:].view(torch.bfloat16)
    input_scale = (
        quant_k_cache[:, block_size * (d_nope + 2 * d_rope) :]
        .view(num_blocks, block_size, 8)[:, :, :7]
        .view(torch.float8_e8m0fnu)
    )

    result[..., d_nope:] = input_rope
    for tile_idx in range(0, 7):
        start = tile_idx * 64
        end = (tile_idx + 1) * 64
        cur_nope = input_nope[..., start:end].to(torch.bfloat16)
        cur_scales = input_scale[:, :, tile_idx].to(torch.bfloat16).unsqueeze(-1)
        result[..., start:end] = cur_nope * cur_scales

    return result.view(num_blocks, block_size, 1, d)


def ref_sparse_attn_decode(
    params: SparseDecodeParams,
    inputs: SparseDecodeInputs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert params.h_kv == 1
    b = params.decode.b

    def process_kv_scope(kv_scope: KVScope) -> Tuple[torch.Tensor, torch.Tensor]:
        topk = kv_scope.indices_in_kvcache.size(-1)
        fixed_indices = torch.clamp_min(kv_scope.indices_in_kvcache, 0)
        gathered_kv = (
            kv_scope.blocked_k.view(-1, params.d_qk)
            .index_select(0, fixed_indices.view(-1))
            .view(b, params.s_q, topk, params.d_qk)
        )
        invalid_mask = kv_scope.indices_in_kvcache == -1
        if kv_scope.topk_length is not None:
            topk_mask = (
                torch.arange(0, topk, device=invalid_mask.device)
                .view(1, 1, topk)
                .broadcast_to(b, params.s_q, topk)
            )
            invalid_mask |= topk_mask >= kv_scope.topk_length.view(b, 1, 1)
        return gathered_kv, invalid_mask

    gathered_kv, invalid_mask = process_kv_scope(inputs.kv_scope)
    if inputs.extra_kv_scope is not None:
        extra_gathered_kv, extra_invalid_mask = process_kv_scope(inputs.extra_kv_scope)
        gathered_kv = torch.cat([gathered_kv, extra_gathered_kv], dim=2)
        invalid_mask = torch.cat([invalid_mask, extra_invalid_mask], dim=2)

    gathered_kv = gathered_kv.view(b * params.s_q, -1, params.d_qk).float()
    gathered_kv[gathered_kv != gathered_kv] = 0.0
    q = inputs.q.float().view(b * params.s_q, params.h_q, params.d_qk)
    attn_weight = q @ gathered_kv.transpose(-1, -2)
    attn_weight *= inputs.sm_scale
    attn_weight[
        invalid_mask.view(b * params.s_q, 1, -1).broadcast_to(
            b * params.s_q, params.h_q, invalid_mask.size(-1)
        )
    ] = float("-inf")
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
    output = attn_weight @ gathered_kv[..., : params.d_v]
    output = output.view(b, params.s_q, params.h_q, params.d_v)
    lse = lse.view(b, params.s_q, params.h_q)

    if inputs.attn_sink is not None:
        output *= (
            1.0 / (1.0 + torch.exp(inputs.attn_sink.view(1, 1, params.h_q) - lse))
        ).unsqueeze(-1)

    lonely_q_mask = lse == float("-inf")
    output[
        lonely_q_mask.unsqueeze(-1).broadcast_to(b, params.s_q, params.h_q, params.d_v)
    ] = 0.0
    lse[lonely_q_mask] = float("+inf")
    return output.to(torch.bfloat16), lse.transpose(1, 2)
