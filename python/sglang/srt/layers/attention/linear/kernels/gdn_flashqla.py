# Copyright (c) 2026 The Qwen team, Alibaba Group.
# SPDX-License-Identifier: MIT
# CP state orchestration adapted from FlashQLA 0.1.3.
"""Opt-in Hopper/Blackwell GDN prefill with FlashQLA 0.1.3.

CP borrows the attention TP group only inside GDN. Convolution runs before
redistribution, so its history and the model's weights keep their TP layout.
Final states retain TP ownership. The MegaMoE path keeps outputs token-sharded.
"""

import logging

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.torch_symm_mem import (
    TorchSymmMemCommunicator,
)
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.layers.cp.utils import (
    ContextParallelOutput,
    RingTensorExchange,
    ShardedContextParallelMetadata,
    _context_to_heads,
    _cp_to_token_shard,
    _heads_to_context,
    select_cp_tactic,
)
from sglang.srt.utils.nvtx_pytorch_hooks import profile_range

logger = logging.getLogger(__name__)


_PREFIX_TACTICS = {}


class FlashQLAGDNKernel(LinearAttnKernelBase):
    def __init__(
        self, *, fallback: LinearAttnKernelBase, cp_group=None, symm_state=True
    ):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
            (9, 0),
            (10, 0),
            (10, 3),
        ):
            raise ValueError("FlashQLA GDN requires SM90, SM100 or SM103")
        try:
            import tilelang
            from packaging.version import Version

            if Version(tilelang.__version__) < Version("0.1.12"):
                raise ImportError("FlashQLA requires TileLang >= 0.1.12")
            from flash_qla import chunk_gated_delta_rule
            from flash_qla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
            from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context
            from flash_qla.utils import index as varlen_index
            from flash_qla.utils import l2norm_fwd
        except ImportError as exc:
            raise ImportError(
                "FlashQLA GDN requires flash-qla==0.1.3 and tilelang>=0.1.12"
            ) from exc
        self._forward = chunk_gated_delta_rule
        self._forward_states = chunk_gated_delta_rule_fwd
        self._normalize = l2norm_fwd
        self._build_context = build_cp_context
        # Older FlashQLA wheels only provide an evictable tensor cache. Use the
        # prepared-metadata API when available without breaking those wheels.
        self._prepare_varlen_metadata = getattr(
            varlen_index, "prepare_varlen_metadata", None
        )
        self._fallback = fallback
        self._group = cp_group
        self._world = dist.get_world_size(cp_group) if cp_group is not None else 1
        self._rank = dist.get_rank(cp_group) if cp_group is not None else 0
        # Contexts contain sequence/topology metadata, never activations or state.
        # Keep tensor identity stable for FlashQLA's own metadata caches as well.
        self._cp_contexts = {}
        self._prefix_workspace = None
        self._state_broadcast_workspace = None
        self._state_exchange = None
        self._prefix_algorithm = "auto"
        self._prefix_tuning = None
        self._symm_state_unavailable = (
            TorchSymmMemCommunicator.multicast_unavailable_reason(
                self._group, enabled=symm_state
            )
        )
        logger.info(
            "FlashQLA GDN prefill: CP size=%d, rank=%d; intermediate radix "
            "snapshots use Triton",
            self._world,
            self._rank,
        )
        if self._world > 1:
            logger.info(
                "GDN state transport: cp_size=%d backend=%s reason=%s",
                self._world,
                "Torch_CUDA_symm+multicast" if self.symm_state_enabled() else "nccl",
                self._symm_state_unavailable or "capabilities available on all ranks",
            )

    def symm_state_enabled(self):
        return self._symm_state_unavailable is None

    def broadcast_state(self, value, owner):
        if self.symm_state_enabled():
            size = value.numel() * value.element_size()
            if (
                self._state_broadcast_workspace is None
                or self._state_broadcast_workspace.buffers.numel() < size
            ):
                if self._state_broadcast_workspace is not None:
                    self._state_broadcast_workspace.close()
                communicator = TorchSymmMemCommunicator.get(self._group, value.device)
                self._state_broadcast_workspace = (
                    communicator.allocate_broadcast_workspace(size * 2)
                )
            workspace = self._state_broadcast_workspace
            with profile_range("sglang.cp.state.broadcast_multicast"):
                return workspace.communicator.broadcast(workspace, value, owner)
        dist.broadcast(
            value, src=dist.get_global_rank(self._group, owner), group=self._group
        )
        return value

    def state_prefix(self, packed, pre, out=None):
        from sglang.kernels.ops.attention.fla.chunk_delta_h import (
            gated_delta_rule_state_prefix,
        )

        if (
            packed.ndim != 4
            or packed.shape[0] != 2
            or packed.shape[-2:] != (128, 128)
            or packed.dtype != torch.bfloat16
        ):
            raise ValueError(
                "Peer state prefix requires BF16 H/M summaries with 128x128 state"
            )
        if (
            self._prefix_workspace is None
            or self._prefix_workspace.buffers.shape != packed.shape
        ):
            if self._prefix_workspace is not None:
                self._prefix_workspace.close()
            communicator = TorchSymmMemCommunicator.get(self._group, packed.device)
            self._prefix_workspace = communicator.allocate_peer_workspace(
                packed.shape, packed.dtype
            )
        workspace = self._prefix_workspace
        tuning = self._prefix_tuning or _PREFIX_TACTICS.get(
            (packed.device.index, self._world, packed.shape[1]), {"bv": 64, "warps": 4}
        )
        workspace.communicator.publish(workspace, packed)
        return gated_delta_rule_state_prefix(
            workspace.peer_ptrs,
            rank=self._rank,
            pre=pre,
            heads=packed.shape[1],
            out=out,
            value_tile=tuning["bv"],
            num_warps=tuning["warps"],
        )

    def _forward_context_shard(self, q, k, v, g, beta, initial, context, *, output_h):
        from flash_qla.ops.gated_delta_rule.chunk import (
            CHUNK_SIZE,
            aggregate_card_state,
            correct_initial_states,
            fused_gdr_fwd,
            fused_gdr_h,
            get_warmup_chunks_bidi,
            kkt_solve,
        )
        from flash_qla.ops.gated_delta_rule.chunk.cp.comm import pack_hm, unpack_hm
        from flash_qla.ops.utils import chunk_local_cumsum
        from flash_qla.utils import l2norm_fwd

        if not context.is_inter:
            raise ValueError(
                "State prefix correction requires inter-rank context parallelism"
            )
        k, _ = l2norm_fwd(k)
        g = chunk_local_cumsum(
            g=g, cu_seqlens=context.cu_seqlens, chunk_size=CHUNK_SIZE
        )
        a = kkt_solve(k=k, b=beta, cu_seqlens=context.cu_seqlens, chunk_size=CHUNK_SIZE)
        if context.is_intra:
            warmup, _, fallback, _ = get_warmup_chunks_bidi(
                g=g,
                cu_seqlens=context.intra_cp_cu_seqlens,
                ht_mask_fwd=context.ht_mask,
                ht_mask_bwd=context.ht_mask_bwd,
                chunk_size=CHUNK_SIZE,
                seq_map_r2c=context.seq_map_r2c,
                force_inter_boundaries=True,
            )
            _, ht, mt = fused_gdr_h(
                k=k,
                v=v,
                a=a,
                g=g,
                b=beta,
                initial_state=None,
                output_final_state=True,
                output_h=False,
                cu_seqlens=context.intra_cp_cu_seqlens,
                num_warmup_chunks=warmup,
                state_v_first=True,
            )
            raw_h0 = initial
            if not context.is_first_rank:
                raw_h0 = initial.clone()
                raw_h0[0] = 0
            h_seq, m_seq = aggregate_card_state(
                ht,
                mt,
                fallback,
                context.seq_map_r2c,
                raw_h0=raw_h0,
                state_v_first=True,
                compute_m=True,
            )
        else:
            lengths = context.cu_seqlens_cpu.tolist()
            n = len(lengths) - 1
            warmup = torch.zeros(
                (n, v.shape[2]), dtype=context.cu_seqlens.dtype, device=k.device
            )
            warmup[-1] = (lengths[-1] - lengths[-2] + CHUNK_SIZE - 1) // CHUNK_SIZE
            if n > 1:
                warmup[0] = (lengths[1] - lengths[0] + CHUNK_SIZE - 1) // CHUNK_SIZE
            _, h_seq, m_seq = fused_gdr_h(
                k=k,
                v=v,
                a=a,
                g=g,
                b=beta,
                initial_state=initial,
                output_final_state=True,
                output_h=False,
                cu_seqlens=context.cu_seqlens,
                num_warmup_chunks=warmup,
                state_v_first=True,
            )
        prefix_algorithm = self._prefix_algorithm
        if prefix_algorithm == "auto":
            prefix_algorithm = (
                "peer_compute"
                if h_seq.dtype == m_seq.dtype == torch.bfloat16
                and h_seq.shape[-2:] == m_seq.shape[-2:] == (128, 128)
                else "ring"
            )
        card_h0 = None
        if prefix_algorithm == "peer_compute":
            card_h0 = initial if context.is_first_rank else initial.clone()
            if h_seq.dtype != torch.bfloat16 or m_seq.dtype != torch.bfloat16:
                raise ValueError("Peer state prefix requires BF16 H/M summaries")
            with profile_range("sglang.cp.state.peer_prefix"):
                self.state_prefix(
                    torch.stack((h_seq[-1], m_seq[-1])),
                    context.pre_num_ranks,
                    out=card_h0[0] if not context.is_first_rank else None,
                )
        else:
            with profile_range("sglang.cp.state_ring.prepare"):
                packed = pack_hm(h_seq[-1], m_seq[-1])
                if (
                    self._state_exchange is None
                    or self._state_exchange.buffers.shape[1:] != packed.shape
                ):
                    self._state_exchange = RingTensorExchange(
                        group=self._group,
                        shape=packed.shape,
                        device=packed.device,
                        dtype=packed.dtype,
                        transport="symm_rma",
                    )
                pending = self._state_exchange.prepare(packed)
        # Q is unused by state-summary production. Enqueue its normalization,
        # then start the peer transfer against the event captured BEFORE this computation.
        with profile_range("sglang.cp.gdn.prepare_q_and_state"):
            q, _ = l2norm_fwd(q)
            if card_h0 is None:
                card_h0 = initial if context.is_first_rank else initial.clone()
        if prefix_algorithm == "ring":
            with profile_range("sglang.cp.state_ring.transfer"):
                pending.start()
                gathered = pending.wait()
            rank, pre = self._rank, context.pre_num_ranks
            h_buf, m_buf = unpack_hm(
                gathered[rank - pre : rank + 1], h_seq[-1], m_seq[-1]
            )
        if not context.is_first_rank and prefix_algorithm == "ring":
            seq_map, mask = context.get_fwd_scan_tensors(v.shape[2])
            corrected = correct_initial_states(
                raw_h0=None,
                ht_buffer=h_buf,
                mt_buffer=m_buf,
                fallback_mask=mask,
                seq_map_r2c=seq_map,
                state_v_first=True,
            )
            card_h0[0] = corrected[h_buf.shape[0] - 1]
        if context.is_intra:
            card_h0 = correct_initial_states(
                raw_h0=card_h0,
                ht_buffer=ht,
                mt_buffer=mt,
                fallback_mask=fallback,
                seq_map_r2c=context.seq_map_r2c,
                state_v_first=True,
            )
        out, h, final = fused_gdr_fwd(
            q=q,
            k=k,
            v=v,
            a=a,
            g=g,
            b=beta,
            scale=k.shape[-1] ** -0.5,
            initial_state=card_h0,
            output_final_state=True,
            output_h=output_h,
            output_o=True,
            cu_seqlens=(
                context.intra_cp_cu_seqlens if context.is_intra else context.cu_seqlens
            ),
            cp_seq_map=context.seq_map_c2r,
            raw_cu_seqlens=context.cu_seqlens,
            state_v_first=True,
        )
        return out.to(q.dtype), h, final

    def decode(self, *args, **kwargs):
        raise NotImplementedError("Select FlashQLA with --linear-attn-prefill-backend")

    def extend(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        ssm_states,
        cache_indices,
        query_start_loc,
        inplace_update=True,
        seq_lens_cpu=None,
        output_intermediate_states=False,
        output=None,
        return_cp_output=False,
        cp_metadata=None,
        initial_state=None,
        checkpoint_indices=(),
        state_dtype=None,
        **kwargs,
    ):
        if cp_metadata is not None:
            if (
                self._world <= 1
                or cp_metadata.world != self._world
                or cp_metadata.rank != self._rank
                or cp_metadata.group is not self._group
            ):
                raise ValueError(
                    "GDN CP metadata does not match the selected prefill group"
                )
            if initial_state is None or inplace_update:
                raise ValueError(
                    "Sharded GDN extend requires loaded initial_state and inplace_update=False"
                )
            if return_cp_output or output is not None:
                raise ValueError(
                    "Sharded GDN extend already returns local token output"
                )
            if q.dtype != torch.bfloat16 or k.shape[-1] != 128 or v.shape[-1] != 128:
                raise ValueError("FlashQLA GDN requires BF16 QKV and K=V=128")
            return self._extend_context_parallel(
                (q, k, v, g, beta),
                initial_state,
                checkpoint_indices=checkpoint_indices,
                state_dtype=state_dtype,
            )
        # FlashQLA's public inference API does not expose radix checkpoints.
        # Graph capture also cannot perform its CPU-dependent CP planning.
        if output_intermediate_states or torch.cuda.is_current_stream_capturing():
            result, final, h = self._fallback.extend(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                inplace_update=inplace_update,
                **kwargs,
            )
        else:
            if q.dtype != torch.bfloat16 or k.shape[-1] != 128 or v.shape[-1] != 128:
                raise ValueError("FlashQLA GDN requires BF16 QKV and K=V=128")
            if seq_lens_cpu is None:
                seq_lens_cpu = query_start_loc.diff().cpu().tolist()
            if sum(seq_lens_cpu) > q.shape[1] or any(n < 0 for n in seq_lens_cpu):
                raise ValueError("FlashQLA GDN sequence lengths exceed input tokens")
            if len(seq_lens_cpu) != cache_indices.numel():
                raise ValueError("FlashQLA GDN requires one state slot per sequence")
            valid_slots = cache_indices >= 0
            initial = ssm_states[cache_indices.clamp_min(0)].float().contiguous()
            initial = torch.where(valid_slots[:, None, None, None], initial, 0)
            return_cp_output = return_cp_output and self._world > 1
            if return_cp_output and (output is not None or v.shape[1] % self._world):
                raise ValueError(
                    "CP output requires equally sized token shards and no TP output buffer"
                )
            result = None if return_cp_output else torch.zeros_like(v)
            cp_chunks = []
            final = initial.clone()
            start = 0
            for index, length in enumerate(seq_lens_cpu):
                if length:
                    local = [
                        x[:, start : start + length].contiguous()
                        for x in (q, k, v, g, beta)
                    ]
                    out, state = self._sequence(
                        local,
                        initial[index : index + 1],
                        return_cp_output=return_cp_output,
                    )
                    if return_cp_output:
                        cp_chunks.append(out[0] * valid_slots[index])
                    else:
                        result[:, start : start + length] = out * valid_slots[index]
                    final[index : index + 1] = state
                elif return_cp_output:
                    cp_chunks.append(
                        v.new_empty((0, v.shape[2] * self._world, v.shape[3]))
                    )
                start += length
            if inplace_update:
                # Advanced indexing supports page-major pools with strided slots.
                # -1 is the runtime's ghost-row sentinel, not the last pool slot.
                ssm_states[cache_indices[valid_slots]] = final[valid_slots].to(
                    ssm_states.dtype
                )
            if return_cp_output:
                result = ContextParallelOutput(
                    _cp_to_token_shard(
                        cp_chunks,
                        lengths=seq_lens_cpu,
                        total_tokens=v.shape[1],
                        group=self._group,
                    )
                )
            h = None
        if output is not None:
            output.copy_(result.reshape_as(output))
            result = output.view_as(result)
        return result, final, h

    def _sequence(self, inputs, initial, *, return_cp_output=False):
        q, k, v, g, beta = inputs
        length = q.shape[1]
        if self._world == 1:
            return self._forward(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                state_v_first=True,
                auto_cp=True,
                enable_fwd_cp_cache=False,
            )

        # Each CP rank must own complete 64-token kernel chunks. Zero gates
        # and beta make padding an identity state transition, including g=0.
        quantum = 64 * self._world
        padded_length = (length + quantum - 1) // quantum * quantum
        shards = []
        for x in inputs:
            padded = x.new_zeros(padded_length, x.shape[2], *x.shape[3:])
            padded[:length] = x[0]
            shard = _heads_to_context(
                padded.reshape(padded_length, x.shape[2], -1), group=self._group
            ).unsqueeze(0)
            shards.append(shard.squeeze(-1) if x.ndim == 3 else shard)
        q, k, v, g, beta = shards

        gathered = initial.new_empty(self._world * initial.shape[1], *initial.shape[2:])
        dist.all_gather_into_tensor(
            gathered, initial[0].contiguous(), group=self._group
        )
        out, final, _ = self._extend_context_parallel(shards, gathered.unsqueeze(0))
        local_heads = initial.shape[1]
        state = final[:, self._rank * local_heads : (self._rank + 1) * local_heads]
        output = (
            out
            if return_cp_output
            else _context_to_heads(out[0], group=self._group)[:length].unsqueeze(0)
        )
        return output, state

    def prepare_cp_context(self, *, device, padded_length, value_heads):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (device, padded_length, value_heads)
        context = self._cp_contexts.get(key)
        if context is None:
            cu_cpu = torch.tensor([0, padded_length], dtype=torch.long)
            context = self._build_context(
                cu_cpu.to(device),
                cu_seqlens_cpu=cu_cpu,
                enable_inter=True,
                enable_intra=True,
                group=self._group,
                num_v_heads=value_heads,
                chunk_size=64,
                is_train=False,
            )
            if self._prepare_varlen_metadata is not None:
                self._prepare_varlen_metadata(
                    context.cu_seqlens, context.cu_seqlens_cpu
                )
                if context.is_intra:
                    self._prepare_varlen_metadata(
                        context.intra_cp_cu_seqlens,
                        context.intra_cp_cu_seqlens.cpu(),
                    )
            self._cp_contexts[key] = context
        return context

    def warmup_cp_contexts(self, *, device, max_tokens, value_heads):
        """Retain metadata for every per-sequence shard bucket on this instance.

        A packed batch still executes each GDN sequence separately. Warming only
        the batch's maximum length misses tails, even when the batch is full.
        """
        alignment = 64 * self._world
        padded_max = (max_tokens + alignment - 1) // alignment * alignment
        with torch.cuda.device(device), torch.inference_mode():
            for length in range(alignment, padded_max + 1, alignment):
                self.prepare_cp_context(
                    device=device, padded_length=length, value_heads=value_heads
                )
            torch.cuda.synchronize(device)

    def _extend_context_parallel(
        self, inputs, initial, *, checkpoint_indices=(), state_dtype=None
    ):
        """Run a full-head shard and replicate final/prefix states for caching.

        Shard ends use the FP32 recurrent accumulator after CP prefix correction.
        Other aligned checkpoints use FlashQLA's BF16 intermediate h export;
        promoting h to the pool dtype cannot recover its discarded precision.
        Padding must have zero beta/g, making it an identity state transition.
        """
        q, k, v, g, beta = inputs
        state_dtype = initial.dtype if state_dtype is None else state_dtype
        width = q.shape[1]
        if not width or width % 64:
            raise ValueError("FlashQLA CP shards must contain complete 64-token chunks")
        padded_length = width * self._world
        if any(
            not 0 <= index <= padded_length or index % 64
            for index in checkpoint_indices
        ):
            raise ValueError(
                "FlashQLA CP checkpoints must align to 64-token boundaries"
            )
        owners = tuple(max(0, (index - 1) // width) for index in checkpoint_indices)
        output_h = any(
            owner == self._rank and index % width
            for index, owner in zip(checkpoint_indices, owners)
        )
        h0 = initial.contiguous() if self._rank == 0 else torch.zeros_like(initial)
        context = self.prepare_cp_context(
            device=q.device, padded_length=padded_length, value_heads=v.shape[2]
        )
        arguments = dict(
            v=v,
            g=g,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            state_v_first=True,
            auto_cp=False,
            enable_fwd_cp_cache=False,
            cp_context=context,
        )
        if self.symm_state_enabled():
            out, h, local_final = self._forward_context_shard(
                q, k, v, g, beta, h0, context, output_h=output_h
            )
        elif output_h:
            q, _ = self._normalize(q)
            k, _ = self._normalize(k)
            _, _, out, h, local_final, _ = self._forward_states(
                q=q, k=k, output_h=True, is_train=False, **arguments
            )
        else:
            out, local_final = self._forward(
                q=q, k=k, use_qk_l2norm_in_kernel=True, **arguments
            )
        local_final = local_final.to(dtype=state_dtype).contiguous()
        if not checkpoint_indices:
            self.broadcast_state(local_final, self._world - 1)
            return out, local_final, ()

        checkpoints = [None] * len(checkpoint_indices)
        # Keep each local final intact until its snapshots have been copied.
        # Pack checkpoints with the final state when they share an owner.
        for owner in sorted(set(owners) | {self._world - 1}):
            selected = [i for i, rank in enumerate(owners) if rank == owner]
            has_final = owner == self._world - 1
            states = initial.new_empty(
                (len(selected) + int(has_final), *initial.shape), dtype=state_dtype
            )
            if self._rank == owner:
                if has_final:
                    states[0].copy_(local_final)
                for offset, i in enumerate(selected, int(has_final)):
                    index = checkpoint_indices[i]
                    if index == 0:
                        state = initial
                    elif index % width == 0:
                        state = local_final
                    else:
                        state = h[:, (index % width) // 64]
                    states[offset].copy_(state)
            self.broadcast_state(states, owner)
            if has_final:
                final = states[0]
            for offset, i in enumerate(selected, int(has_final)):
                checkpoints[i] = states[offset]
        return out, final, tuple(checkpoints)


def _autotune_state_prefix(kernel, packed):
    from sglang.kernels.ops.attention.fla.chunk_delta_h import (
        warmup_gated_delta_rule_state_prefix,
    )

    heads = packed.shape[1]
    for tile in (32, 64, 128):
        warmup_gated_delta_rule_state_prefix(kernel._rank, heads, tile)
    dist.barrier(group=kernel._group)

    def run(tactic):
        kernel._prefix_tuning = tactic
        kernel.state_prefix(packed, kernel._rank)

    tuning = select_cp_tactic(
        [{"bv": bv, "warps": 4} for bv in (32, 64, 128)], run, kernel._group
    )
    kernel._prefix_tuning = tuning
    _PREFIX_TACTICS[(packed.device.index, kernel._world, heads)] = tuning
    return tuning


def _warmup_flashqla_gdn(
    device,
    *,
    geometry=(16, 64, 128, 128),
    dtype=None,
    state_dtype=None,
    cp_group=None,
    cp_size=1,
    cp_rank=0,
    prefix_algorithm="auto",
    prefix_tile=None,
):
    """Run the selected complete recurrence using private scratch and the CP group.

    This reaches normalization, cumsum, KKT solve, state summary/correction,
    output/checkpoint TileLang variants and state publication. It does not
    borrow model weights, live request state or KV pools.
    """
    if cp_group is None:
        raise ValueError("CP FlashQLA warmup requires the initialized CP device group")
    if (
        torch.distributed.get_world_size(cp_group) != cp_size
        or torch.distributed.get_rank(cp_group) != cp_rank
    ):
        raise ValueError("CP warmup rank/size differ from the device group")
    selection = (prefix_algorithm, prefix_tile)
    selections = [None] * cp_size
    with torch.inference_mode(False):
        dist.all_gather_object(selections, selection, group=cp_group)
    if any(peer != selection for peer in selections):
        raise ValueError("CP ranks disagree on state prefix tactic")
    if prefix_algorithm not in ("auto", "ring", "peer_compute") or prefix_tile not in (
        None,
        16,
        32,
        64,
        128,
    ):
        raise ValueError("Invalid state prefix algorithm or tile")
    dtype = dtype or torch.bfloat16
    state_dtype = state_dtype or torch.float32
    kh, vh, kd, vd = geometry
    kernel = FlashQLAGDNKernel(fallback=None, cp_group=cp_group)
    metadata = ShardedContextParallelMetadata(
        group=cp_group, rank=cp_rank, world=cp_size
    )
    try:
        if prefix_algorithm == "auto":
            if kernel.symm_state_enabled():
                packed = torch.zeros((2, vh, vd, kd), device=device, dtype=dtype)
                _autotune_state_prefix(kernel, packed)
        elif kernel.symm_state_enabled():
            kernel._prefix_algorithm = prefix_algorithm
            kernel._prefix_tuning = {"bv": prefix_tile or 64, "warps": 4}
            if prefix_algorithm == "peer_compute":
                from sglang.kernels.ops.attention.fla.chunk_delta_h import (
                    warmup_gated_delta_rule_state_prefix,
                )

                warmup_gated_delta_rule_state_prefix(cp_rank, vh, prefix_tile or 64)
                dist.barrier(group=cp_group)
        # The GDN adapter pads even empty local ranks to complete 64-token
        # blocks. 128 covers multiple chunks without a production-size input.
        for tokens in (64, 128):
            q = torch.zeros((1, tokens, kh, kd), device=device, dtype=dtype)
            v = torch.zeros((1, tokens, vh, vd), device=device, dtype=dtype)
            gates = torch.zeros((1, tokens, vh), device=device, dtype=torch.float32)
            initial = torch.zeros((1, vh, vd, kd), device=device, dtype=torch.float32)
            inputs = [q, q.clone(), v, gates, gates.clone()]
            # Every rank must own a checkpoint once: output_h is a distinct
            # TileLang specialization even though ordinary token counts are dynamic.
            checkpoints = (
                (None, *(owner * tokens + 64 for owner in range(cp_size)))
                if tokens == 128
                else (None,)
            )
            for checkpoint in checkpoints:
                kernel.extend(
                    *inputs,
                    ssm_states=initial,
                    cache_indices=None,
                    query_start_loc=None,
                    cp_metadata=metadata,
                    initial_state=initial,
                    inplace_update=False,
                    checkpoint_indices=() if checkpoint is None else (checkpoint,),
                    state_dtype=state_dtype,
                )
            if tokens == 128:
                # A branch and a shard end can share an owner. Compile their
                # packed state publication before handling real requests.
                for owner in range(cp_size - 1):
                    kernel.extend(
                        *inputs,
                        ssm_states=initial,
                        cache_indices=None,
                        query_start_loc=None,
                        cp_metadata=metadata,
                        initial_state=initial,
                        inplace_update=False,
                        checkpoint_indices=(owner * tokens + 64, (owner + 1) * tokens),
                        state_dtype=state_dtype,
                    )
        torch.cuda.synchronize(device)
    finally:
        # Finish private peer reads before dropping scratch symmetric mappings.
        torch.cuda.synchronize(device)
        del kernel
