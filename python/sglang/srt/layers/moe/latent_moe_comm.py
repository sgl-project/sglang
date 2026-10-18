"""Latent-width MoE exchange for DP attention with ``--moe-a2a-backend none``.

A latent MoE projects each token from the hidden size down to a narrower
latent size before its routed experts and back up after them. Under DP
attention without an A2A backend, the default path all-gathers hidden-width
rows across the EP group, runs the router and the down projection on every
gathered row on every rank, and reduce-scatters hidden-width partial sums.

This path exchanges latent rows instead. Each attention-TP rank routes and
projects its ``C = N // attn_tp_size`` slice of the DP group's ``N`` tokens,
the EP group all-gathers ``[C, latent + top-k weights + top-k ids]`` rows and
reduce-scatters the experts' latent partial sums, and each rank projects its
own reduced rows back up. One attention-TP all-reduce then assembles every
rank's rows together with the shared expert, which is sharded over attention
TP. Gathering the slices in TP rank order reproduces the row layout of the
hidden-width DP gather, so routed-experts capture keeps its indexing.
"""

import functools
import logging
from typing import Callable, Optional, Sequence

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    is_allocation_symmetric,
    is_dp_attention_enabled,
    is_dp_max_padding,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker
from sglang.srt.runtime_context import get_parallel
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

logger = logging.getLogger(__name__)

WEIGHTS_DTYPE = torch.float32
IDS_DTYPE = torch.int32


def rank_chunk_sizes(*, dp_num_tokens: Sequence[int], attn_tp_size: int) -> list[int]:
    """Per-TP-rank row counts of the dispatch buffer, in TP rank order."""
    sizes = []
    for num_tokens in dp_num_tokens:
        if num_tokens % attn_tp_size != 0:
            raise ValueError(
                f"DP token count {num_tokens} is not a multiple of "
                f"attn_tp_size {attn_tp_size}"
            )
        sizes.extend([num_tokens // attn_tp_size] * attn_tp_size)
    return sizes


def _payload_columns(
    *, top_k: int, payload_dtype: torch.dtype, dtype: torch.dtype
) -> int:
    nbytes = top_k * dtype.itemsize
    if nbytes % payload_dtype.itemsize != 0:
        raise ValueError(
            f"top_k={top_k} {dtype} values do not fill whole {payload_dtype} columns"
        )
    return nbytes // payload_dtype.itemsize


def pack_dispatch_payload(
    *, latent: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
) -> torch.Tensor:
    """Pack latents, fp32 weights and int32 ids into one ``[C, D + w + i]``
    buffer for a single all-gather. Weights and ids are bit-cast, so the
    buffer must never be summed."""
    num_rows, latent_dim = latent.shape
    top_k = topk_ids.shape[1]
    weight_cols = _payload_columns(
        top_k=top_k, payload_dtype=latent.dtype, dtype=WEIGHTS_DTYPE
    )
    id_cols = _payload_columns(top_k=top_k, payload_dtype=latent.dtype, dtype=IDS_DTYPE)
    payload = latent.new_empty((num_rows, latent_dim + weight_cols + id_cols))
    payload[:, :latent_dim].copy_(latent)
    payload[:, latent_dim : latent_dim + weight_cols].view(WEIGHTS_DTYPE).copy_(
        topk_weights
    )
    payload[:, latent_dim + weight_cols :].view(IDS_DTYPE).copy_(topk_ids)
    return payload


def unpack_dispatch_payload(
    payload: torch.Tensor, *, latent_dim: int, top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of ``pack_dispatch_payload``; returns contiguous tensors."""
    weight_cols = _payload_columns(
        top_k=top_k, payload_dtype=payload.dtype, dtype=WEIGHTS_DTYPE
    )
    weights_end = latent_dim + weight_cols
    latent = payload[:, :latent_dim].contiguous()
    topk_weights = payload[:, latent_dim:weights_end].view(WEIGHTS_DTYPE).contiguous()
    topk_ids = payload[:, weights_end:].view(IDS_DTYPE).contiguous()
    return latent, topk_weights, topk_ids


def _start_shared(shared_experts, hidden_states, alt_stream):
    """Launch the shared expert's attention-TP partial, on ``alt_stream`` when
    given so it overlaps the EP collectives, and return a waiter."""
    if shared_experts is None:
        return lambda: None
    if alt_stream is None:
        output = shared_experts(hidden_states)
        return lambda: output
    current_stream = torch.cuda.current_stream()
    alt_stream.wait_stream(current_stream)
    with torch.cuda.stream(alt_stream):
        output = shared_experts(hidden_states)

    def wait() -> torch.Tensor:
        current_stream.wait_stream(alt_stream)
        return output

    return wait


def latent_moe_forward(
    hidden_states: torch.Tensor,
    *,
    layer_id: int,
    gate: Callable[[torch.Tensor], torch.Tensor],
    topk: Callable,
    down_proj: Callable[[torch.Tensor], torch.Tensor],
    experts: Callable,
    up_proj: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    shared_experts: Optional[Callable[[torch.Tensor], torch.Tensor]],
    alt_stream: Optional[torch.cuda.Stream],
) -> torch.Tensor:
    """Complete MoE output for this DP group's post-norm ``[N, H]`` tokens,
    replicated over the attention-TP group like the input.

    ``gate`` gives router logits and ``topk`` a standard top-k output for the
    rows it is given; ``experts(latent, topk_output)`` returns this rank's
    ``[G, D]`` partial sum; ``up_proj(latent, addend)`` returns ``addend``
    plus the reduced latent rows projected back to hidden size (after any
    latent norm), possibly in ``addend`` itself.
    """
    comm = DpAttnLatentMoEComm(num_local_tokens=hidden_states.shape[0])
    start = get_parallel().attn_tp_rank * comm.own_rows
    end = start + comm.own_rows
    own_rows = hidden_states[start:end]
    wait_shared = _start_shared(shared_experts, hidden_states, alt_stream)

    topk_output = topk(own_rows, gate(own_rows))
    assert TopKOutputChecker.format_is_standard(topk_output)
    latent = down_proj(own_rows)
    payload = comm.dispatch(
        pack_dispatch_payload(
            latent=latent,
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_output.topk_ids,
        )
    )
    latent, topk_weights, topk_ids = unpack_dispatch_payload(
        payload, latent_dim=latent.shape[1], top_k=topk_output.topk_ids.shape[1]
    )
    # The layer's TopK routed only this rank's slice, so its own capture is
    # off; the gathered ids keep the DP-gathered row layout.
    if (capturer := get_global_experts_capturer()) is not None:
        capturer.capture(layer_id=layer_id, topk_indices=topk_ids)
    partial = experts(
        latent,
        StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            # Router logits stay rank-local; the runners this path allows
            # route from the precomputed top-k instead.
            router_logits=topk_weights.new_empty((topk_weights.shape[0], 0)),
        ),
    )
    reduced_latent = comm.combine(partial)

    output = wait_shared()
    if output is None:
        output = hidden_states.new_zeros(hidden_states.shape)
    own_output = output[start:end]
    result = up_proj(reduced_latent, own_output)
    if result.data_ptr() != own_output.data_ptr():
        own_output.copy_(result)
    # One attention-TP all-reduce sums the shared expert's partials and
    # assembles every rank's rows.
    return get_parallel().attn_tp_group.all_reduce(output)


class DpAttnLatentMoEComm:
    """Dispatch and combine over the TP group, which is the EP group here.
    MAX_LEN padding (CUDA graphs) uses static-shape collectives; SUM_LEN uses
    the variable-size ones unless every rank holds the same row count."""

    def __init__(self, *, num_local_tokens: int):
        parallel = get_parallel()
        attn_tp_size = parallel.attn_tp_size
        if is_dp_max_padding():
            self.rank_sizes = [num_local_tokens // attn_tp_size] * parallel.tp_size
        else:
            self.rank_sizes = rank_chunk_sizes(
                dp_num_tokens=get_dp_global_num_tokens(), attn_tp_size=attn_tp_size
            )
        self.uniform = all(size == self.rank_sizes[0] for size in self.rank_sizes)
        self.own_rows = self.rank_sizes[parallel.tp_rank]
        assert self.own_rows * attn_tp_size == num_local_tokens, (
            f"{self.rank_sizes=} disagrees with {num_local_tokens=}"
        )

    def dispatch(self, payload: torch.Tensor) -> torch.Tensor:
        group = get_tp_group()
        with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
            output = payload.new_empty((sum(self.rank_sizes), payload.shape[1]))
        if self.uniform:
            group.all_gather_into_tensor(output, payload)
        else:
            group.all_gatherv(payload, sizes=self.rank_sizes, output=output)
        return output

    def combine(self, partial: torch.Tensor) -> torch.Tensor:
        group = get_tp_group()
        with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
            output = partial.new_empty((self.own_rows, partial.shape[1]))
        if self.uniform:
            group.reduce_scatter_tensor(output, partial)
        else:
            group.reduce_scatterv(partial, output=output, sizes=self.rank_sizes)
        return output


@functools.cache
def _latent_moe_comm_blockers() -> tuple[str, ...]:
    """Server-wide reasons the latent exchange cannot run."""
    from sglang.srt.layers import layernorm_sp
    from sglang.srt.layers.moe.utils import (
        get_moe_a2a_backend,
        get_moe_runner_backend,
        should_use_flashinfer_cutlass_moe_fp4_allgather,
    )
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        Phase,
        check_cuda_graph_backend,
    )
    from sglang.srt.runtime_context import get_exec, get_flags, get_spec

    parallel = get_parallel()
    moe = get_exec().moe
    checks = (
        (
            is_dp_attention_enabled() and parallel.attn_dp_size > 1,
            "needs --enable-dp-attention with --dp-size > 1",
        ),
        (get_moe_a2a_backend().is_none(), "needs --moe-a2a-backend none"),
        (
            parallel.moe_ep_size > 1 and parallel.moe_ep_size == parallel.tp_size,
            "needs --ep-size equal to --tp-size",
        ),
        # The experts consume precomputed top-k ids and weights; runners that
        # route from the router logits themselves cannot take gathered ids.
        # flashinfer_mxfp4 takes precomputed top-k for SiTU experts (Kimi K3).
        (
            get_moe_runner_backend().is_triton()
            or get_moe_runner_backend().is_flashinfer_mxfp4(),
            "needs --moe-runner-backend triton or flashinfer_mxfp4",
        ),
        (parallel.attn_cp_size == 1, "attention context parallelism"),
        (parallel.dwdp_size <= 1, "DWDP"),
        (get_spec().speculative_algorithm is None, "speculative decoding"),
        (not get_exec().overlap.enable_two_batch_overlap, "two-batch overlap"),
        (
            not moe.enable_eplb and moe.ep_num_redundant_experts == 0,
            "EPLB / redundant experts",
        ),
        (moe.elastic_ep_backend is None, "elastic EP"),
        (not layernorm_sp.layernorm_sp_enabled(), "LayerNorm sequence parallel"),
        (
            not should_use_flashinfer_cutlass_moe_fp4_allgather(),
            "flashinfer cutlass FP4 all-gather",
        ),
        (not get_flags().capture.enable_torch_compile, "torch.compile"),
        (
            not any(
                check_cuda_graph_backend(phase, Backend.TC_PIECEWISE)
                for phase in Phase.ALL
            ),
            "tc_piecewise CUDA graphs",
        ),
    )
    blockers = tuple(reason for supported, reason in checks if not supported)
    if blockers:
        logger.warning(
            "SGLANG_OPT_LATENT_MOE_COMM is set but unsupported here (%s); "
            "MoE layers keep the hidden-width DP gather.",
            "; ".join(blockers),
        )
    else:
        logger.info("Latent-width MoE exchange enabled.")
    return blockers


def use_latent_moe_comm(*, is_latent_moe: bool) -> bool:
    """Whether a latent MoE layer exchanges latent rows. Opt-in via
    SGLANG_OPT_LATENT_MOE_COMM; unsupported configurations log why once and
    keep the default path."""
    return (
        is_latent_moe
        and envs.SGLANG_OPT_LATENT_MOE_COMM.get()
        and not _latent_moe_comm_blockers()
    )
