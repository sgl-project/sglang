# SPDX-License-Identifier: Apache-2.0
"""Opt-in SM120 SubBlock attention with quantization before NCCL Ulysses.

FlashInfer's prepared Sage2 exchange supplies INT8 Q/K, FP8 V and scales.
Routing uses reconstructed quantized operands, so its selected blocks can
change relative to routing the original BF16 Q/K. This is an explicit
approximate path, including when layout/pooling fusion is disabled.
"""

from __future__ import annotations

import os

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


class _SparseSageA2ACompute:
    """Route and consume prepared INT8/FP8 operands without requantization.

    The caller admits only sparse steps; dense warmup retains the native path.
    """

    def __init__(self, schedule: SubBlockSparseSchedule, query_block_mask=None):
        self.schedule = schedule
        self.query_block_mask = query_block_mask
        self.router = SubBlockRouter(n_k=schedule.n_k, n_q=schedule.n_q)

    def __call__(
        self,
        *,
        q_int8,
        k_int8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        out,
        used_sequence,
        global_sequence,
        softmax_scale,
        output_dtype,
    ):
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


class _PreparedSageTransport:
    def __init__(self, group, query, used):
        import torch.distributed as dist
        from flashinfer.comm import UlyssesCommunicator

        if not hasattr(UlyssesCommunicator, "prepare_qkv"):
            raise ImportError(
                "quantize_before_a2a requires FlashInfer with the prepared Sage2 "
                "QKV API (flashinfer-ai/flashinfer#4924); the pinned 0.7.3 "
                "release does not provide it"
            )
        self._local_used = max(
            0, min(query.shape[0], used - dist.get_rank(group) * query.shape[0])
        )
        # 3 * Q bytes conservatively covers the complete combined lowp operand,
        # including the received payload and scales, not just one source shard.
        self.comm = UlyssesCommunicator(
            group,
            max_bytes=3 * query.numel() * query.element_size(),
            dtype=query.dtype,
            device=query.device,
            backend="nccl",
        )
        self.workspace = self.comm.prepare_qkv(
            (1, *query.shape), quantization="sage2", used_sequence=used
        )
        self.encoded = None
        self.gathered = torch.empty(
            (1, *query.shape), dtype=query.dtype, device=query.device
        )

    def exchange(self, q, k, v):
        # prepare_qkv requires zero input rows beyond the global live prefix.
        # These are disposable projection rows; clear only padding in place.
        if self._local_used < q.shape[0]:
            for operand in (q, k, v):
                operand[self._local_used :].zero_()
        self.encoded = self.comm.scatter_qkv(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            workspace=self.workspace,
            out=self.encoded,
        )
        return self.encoded

    def gather(self, output):
        return self.comm.gather_heads(output, out=self.gathered)

    def close(self):
        self.comm.close()


_TRANSPORTS: dict[tuple, _PreparedSageTransport] = {}
_HARDWARE_SUPPORT: dict[tuple, bool] = {}
_MAX_GEOMETRIES = 8


def _hardware_supported_by_all_ranks(group, device):
    # Negotiate once: a rank-local architecture rejection could leave peers
    # entering prepared collectives on a mixed-GPU group.
    import torch.distributed as dist

    key = (group, device)
    if key not in _HARDWARE_SUPPORT:
        supported = torch.version.hip is None and torch.cuda.get_device_capability(
            device
        ) == (12, 0)
        vote = torch.tensor([int(supported)], dtype=torch.int32, device=device)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
        _HARDWARE_SUPPORT[key] = bool(vote.item())
    return _HARDWARE_SUPPORT[key]


def _get_transport(group, query, used):
    # Geometry and request order must agree on all ranks, as for ordinary
    # Ulysses. No rank-local fallback is attempted after collective preparation.
    key = (group, tuple(query.shape), query.dtype, query.device, used)
    if key not in _TRANSPORTS:
        if len(_TRANSPORTS) >= _MAX_GEOMETRIES:
            raise RuntimeError("prepared Sage QKV geometry budget exhausted")
        _TRANSPORTS[key] = _PreparedSageTransport(group, query, used)
    return _TRANSPORTS[key]


def shutdown_prepared_sage_ulysses():
    for transport in _TRANSPORTS.values():
        transport.close()
    _TRANSPORTS.clear()
    _HARDWARE_SUPPORT.clear()


def prepared_sage_ulysses_forward(
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
    """Return rank-local output, or decline before any communication.

    Dense warmup steps and disabled layers retain the existing BF16 path.
    Once prepare/scatter starts, errors propagate on this path.
    """
    if (
        not impl.schedule.quantize_before_a2a
        or not impl.layer_enabled
        or not impl._step_enabled()
        or max_seqlen < impl.schedule.min_seq_len
    ):
        return None
    group = None
    if transport is None:
        # Execution mode and packed metadata agree across the SGLang ranks.
        if (
            not query.is_cuda
            or torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_sp_group,
        )

        group = get_sp_group().ulysses_group
        if group is None or not _hardware_supported_by_all_ranks(group, query.device):
            return None
    from .sage_ulysses_route import check_prepared_sage_route

    decision = check_prepared_sage_route(
        query,
        key,
        value,
        cu_seqlens_host=cu_seqlens_host,
        max_seqlen=max_seqlen,
        ulysses_active=ulysses_active,
        ring_active=ring_active,
    )
    if not decision.enabled:
        logger.info_once(
            f"Prepared Sage QKV declined before exchange: {decision.reason}"
        )
        return None
    used = cu_seqlens_host[1]
    if (
        query_block_mask is None
        or query_block_mask.numel() != (used + BLOCK - 1) // BLOCK
    ):
        raise ValueError(
            "H3 prepared Sage QKV requires the live sparse query-block mask"
        )
    if transport is None:
        transport = _get_transport(group, query, used)
    encoded = transport.exchange(query, key, value)
    output = torch.empty_like(encoded.q, dtype=query.dtype)
    _SparseSageA2ACompute(impl.schedule, query_block_mask)(
        q_int8=encoded.q,
        k_int8=encoded.k,
        v_fp8=encoded.v,
        q_scale=encoded.q_scale,
        k_scale=encoded.k_scale,
        v_scale=encoded.v_scale,
        out=output,
        used_sequence=encoded.used_sequence,
        global_sequence=encoded.logical_sequence,
        softmax_scale=impl.softmax_scale,
        output_dtype=query.dtype,
    )
    result = transport.gather(output)
    logger.info_once("MiniMax H3 prepared Sage QKV + CAKE SSA active: input a2a=nccl")
    return result[0]
