# SPDX-License-Identifier: Apache-2.0
"""SM120 SubBlock/CAKE consumer of prepared Sage2 QKV operands.

FlashInfer's prepared Sage2 exchange supplies INT8 Q/K, FP8 V and scales.
Routing uses reconstructed quantized operands, so its selected blocks can
change relative to routing the original BF16 Q/K. This is an explicit
approximate path, including when layout/pooling fusion is disabled.
"""

from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse import (
    SubBlockRouter,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
    SubBlockSparseSchedule,
    _cached_block_sizes,
    _load_sm120_sage_ops,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from flashinfer.comm.ulysses import UlyssesQKV

logger = init_logger(__name__)
BLOCK = 64


def load_bsa_attn_sm120_blk64_sage_fwd():
    return _load_sm120_sage_ops()[1]


def _dequantize_for_routing(
    q_int8: torch.Tensor, k_int8: torch.Tensor, k_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference quantized-operand estimator for the fused pooling path.

    Apply K's per-64-token scale after rounding both factors to BF16. Q scales
    and the centered K channel mean are omitted. This approximation can change
    SubBlock's pooled scores and top-k relative to original BF16 routing.
    """
    q_bf16 = q_int8.to(torch.bfloat16)
    seq_len = k_int8.shape[2]
    tiles = -(-seq_len // BLOCK)
    per_token = (
        k_scale[:, :, :tiles]
        .to(torch.bfloat16)
        .repeat_interleave(BLOCK, dim=-1)[:, :, :seq_len]
    )
    k_bf16 = k_int8.to(torch.bfloat16) * per_token.unsqueeze(-1)
    return (q_bf16, k_bf16)


class PreparedSubBlockSageAttention:
    """Route and consume prepared INT8/FP8 operands without requantization.

    The caller admits only sparse steps; dense warmup retains the native path.
    """

    def __init__(self, schedule: SubBlockSparseSchedule, query_block_mask=None):
        self.schedule = schedule
        self.query_block_mask = query_block_mask
        self.router = SubBlockRouter(n_k=schedule.n_k, n_q=schedule.n_q)

    def __call__(self, operands: UlyssesQKV, *, softmax_scale: float) -> torch.Tensor:
        if operands.layout != "sage2_sm89_sm120":
            raise ValueError("SM120 SubBlock CAKE requires sage2_sm89_sm120 operands")
        q_int8, k_int8, v_fp8 = operands.q, operands.k, operands.v
        q_scale, k_scale, v_scale = (
            operands.q_scale,
            operands.k_scale,
            operands.v_scale,
        )
        used_sequence = operands.used_sequence
        global_sequence = operands.logical_sequence
        output_dtype = operands.input_dtype
        out = torch.empty_like(q_int8, dtype=output_dtype)
        fused = (
            q_int8.is_cuda
            and os.environ.get("SGLANG_DIFFUSION_H3_SAGE_FUSED_PREPOST", "1") != "0"
        )
        if fused:
            from .cake_prepost_kernels import (
                mixed_row_indices,
                pack_cake_operands,
                route_prequantized,
                write_cake_output,
            )

            q_bhsd, k_bhsd, v_bhds, qs, ks = pack_cake_operands(
                q_int8, k_int8, v_fp8, q_scale, k_scale, used_sequence
            )
            index, topk = route_prequantized(
                q_int8,
                k_int8,
                k_scale,
                used_sequence,
                self.schedule.n_q,
                self.schedule.n_k,
                self.schedule.sparsity,
                softmax_scale,
            )
        else:
            q_bhsd = q_int8[:, :used_sequence].transpose(1, 2).contiguous()
            k_bhsd = k_int8[:, :used_sequence].transpose(1, 2).contiguous()
            v_slots = (used_sequence + BLOCK - 1) // BLOCK * BLOCK
            v_bhds = v_fp8[..., :v_slots].transpose(1, 2).contiguous()
            qs = q_scale[:, :, : (used_sequence + 127) // 128 * 4].contiguous()
            ks = k_scale[:, :, : (used_sequence + BLOCK - 1) // BLOCK].contiguous()
            q_route, k_route = _dequantize_for_routing(q_bhsd, k_bhsd, ks)
            plan = self.router.route(
                q_route.transpose(1, 2),
                k_route.transpose(1, 2),
                sparsity=self.schedule.sparsity,
                softmax_scale=softmax_scale,
            )
            index, topk = plan.index, plan.topk
        batch, heads = q_bhsd.shape[:2]
        num_blocks = (used_sequence + BLOCK - 1) // BLOCK
        logger.info_once(
            f"Prepared Sage SubBlock active: S={used_sequence} heads={heads} "
            f"keeping {topk}/{num_blocks} key blocks per sparse query block"
        )
        block_counts = None
        if self.query_block_mask is not None:
            if fused:
                index, block_counts = mixed_row_indices(
                    index, self.query_block_mask, num_blocks
                )
            else:
                mask = self.query_block_mask.to(
                    device=q_bhsd.device, dtype=torch.bool
                ).view(-1)
                if mask.numel() != num_blocks:
                    raise ValueError(
                        "H3 sparse query-block mask does not match the live sequence"
                    )
                dense = (
                    torch.arange(num_blocks, device=q_bhsd.device, dtype=torch.int32)
                    .view(1, 1, 1, -1)
                    .expand(batch, heads, num_blocks, -1)
                    .clone()
                )
                dense[..., :topk] = torch.where(
                    mask.view(1, 1, -1, 1), index, dense[..., :topk]
                )
                block_counts = (
                    torch.where(mask.view(1, 1, -1), topk, num_blocks)
                    .expand(batch, heads, num_blocks)
                    .to(torch.int32)
                    .contiguous()
                )
                index = dense
            topk = num_blocks
        attn = load_bsa_attn_sm120_blk64_sage_fwd()(
            q_bhsd,
            k_bhsd,
            v_bhds,
            qs,
            ks,
            v_scale,
            index,
            topk,
            block_sizes=_cached_block_sizes(used_sequence, q_bhsd.device),
            q2k_block_nums=block_counts,
            softmax_scale=softmax_scale,
            out=torch.empty_like(q_bhsd, dtype=torch.bfloat16),
            uniform_block_count=block_counts is None,
            contiguous_block_indices=False,
            backend="cake",
        )
        if fused:
            write_cake_output(attn, out, used_sequence, global_sequence)
        else:
            if used_sequence != global_sequence:
                out.zero_()
            out[:, :used_sequence] = attn.transpose(1, 2).to(output_dtype)
        return out


def prepared_subblock_sage_attention_forward(
    query,
    key,
    value,
    *,
    impl,
    cu_seqlens_host,
    max_seqlen,
    ulysses_active,
    ring_active,
    query_block_mask=None,
    transport=None,
):
    """Return rank-local output, or decline before prepared QKV exchange.

    SM120 Sage DiT warmup/skipped steps use dense Sage; admitted sparse
    steps use the SubBlock/CAKE consumer with the same prepared transport.
    Once prepare/scatter starts, errors propagate on this path.
    """
    if not impl.schedule.quantize_before_a2a:
        return None
    if (
        not impl.layer_enabled
        or not impl._step_enabled()
        or max_seqlen < impl.schedule.min_seq_len
    ):
        dense_impl = getattr(impl, "dense_impl", None)
        if not getattr(dense_impl, "quantize_before_a2a", False):
            return None
        from sglang.multimodal_gen.runtime.layers.attention.backends.prepared_sage_attn import (
            prepared_dense_sage_attention_forward,
        )

        return prepared_dense_sage_attention_forward(
            query,
            key,
            value,
            impl=dense_impl,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            ulysses_active=ulysses_active,
            ring_active=ring_active,
            transport=transport,
        )
    from sglang.multimodal_gen.runtime.distributed import sage_ulysses
    from sglang.multimodal_gen.runtime.distributed.sage_ulysses_route import (
        resolve_prepared_sage_route,
    )

    decision, group = resolve_prepared_sage_route(
        query,
        key,
        value,
        cu_seqlens_host=cu_seqlens_host,
        max_seqlen=max_seqlen,
        ulysses_active=ulysses_active,
        ring_active=ring_active,
        transport=transport,
    )
    if not decision.enabled:
        logger.info_once(
            f"Prepared Sage QKV declined before exchange: {decision.reason}"
        )
        return None
    used = cu_seqlens_host[1]
    blocks = (used + BLOCK - 1) // BLOCK
    world = cu_seqlens_host[2] // query.shape[0]
    if (query.shape[1] // world) * blocks * blocks > 2**29:
        logger.info_once(
            "Prepared SubBlock Sage declined: routing metadata exceeds per-call limit"
        )
        return None
    if (
        query_block_mask is None
        or query_block_mask.numel() != (used + BLOCK - 1) // BLOCK
    ):
        raise ValueError(
            "H3 prepared Sage QKV requires the live sparse query-block mask"
        )
    attention = partial(
        PreparedSubBlockSageAttention(impl.schedule, query_block_mask),
        softmax_scale=impl.softmax_scale,
    )
    result = sage_ulysses.prepared_sage_ulysses_forward(
        query,
        key,
        value,
        group=group,
        used_sequence=used,
        attention=attention,
        transport=transport,
    )
    logger.info_once("MiniMax H3 prepared Sage QKV + CAKE SSA active: input a2a=nccl")
    return result
