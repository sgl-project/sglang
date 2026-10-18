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

``latent_moe_forward`` is torch-only; callers inject the model's compute and
the collectives (``DpAttnLatentMoEComm`` on GPUs).
"""

import functools
import logging
from typing import Callable, Optional, Protocol, Sequence

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
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

WEIGHTS_DTYPE = torch.float32
IDS_DTYPE = torch.int32


class LatentMoEComm(Protocol):
    def dispatch(self, payload: torch.Tensor) -> torch.Tensor:
        """All-gather this rank's ``[C, W]`` rows into ``[G, W]`` across the
        EP group, in TP rank order."""

    def combine(self, partial: torch.Tensor) -> torch.Tensor:
        """Sum ``[G, D]`` partial expert outputs over the EP group and return
        this rank's ``[C, D]`` rows (reduce-scatter)."""

    def attn_tp_all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Sum over this rank's attention-TP group."""


def chunk_bounds(
    *, num_local_tokens: int, attn_tp_rank: int, attn_tp_size: int
) -> tuple[int, int]:
    """This attention-TP rank's contiguous slice of the DP group's tokens."""
    if num_local_tokens % attn_tp_size != 0:
        raise ValueError(
            f"DP-local token count {num_local_tokens} is not a multiple of "
            f"attn_tp_size {attn_tp_size}"
        )
    chunk = num_local_tokens // attn_tp_size
    return attn_tp_rank * chunk, (attn_tp_rank + 1) * chunk


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


def latent_moe_forward(
    hidden_states: torch.Tensor,
    *,
    attn_tp_rank: int,
    attn_tp_size: int,
    route: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    down_proj: Callable[[torch.Tensor], torch.Tensor],
    experts: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    up_proj_accumulate: Callable[[torch.Tensor, torch.Tensor], None],
    start_shared: Callable[[torch.Tensor], Callable[[], Optional[torch.Tensor]]],
    comm: LatentMoEComm,
    on_global_topk_ids: Optional[Callable[[torch.Tensor], None]] = None,
) -> torch.Tensor:
    """Complete MoE output for this DP group's post-norm ``[N, H]`` tokens,
    replicated over the attention-TP group like the input.

    ``route`` returns fp32 top-k weights and int32 ids for its rows;
    ``down_proj`` maps hidden rows to latent rows; ``experts`` returns this
    rank's ``[G, D]`` partial sum; ``up_proj_accumulate(latent, out)`` adds
    the reduced latent rows, projected back to hidden size (after any latent
    norm), into ``out``; ``start_shared`` launches the shared expert's
    attention-TP partial and returns a waiter (which may return None);
    ``on_global_topk_ids`` sees the gathered ``[G, k]`` ids.
    """
    start, end = chunk_bounds(
        num_local_tokens=hidden_states.shape[0],
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
    )
    own_rows = hidden_states[start:end]
    wait_shared = start_shared(hidden_states)

    topk_weights, topk_ids = route(own_rows)
    latent = down_proj(own_rows)
    latent_dim, top_k = latent.shape[1], topk_ids.shape[1]
    payload = comm.dispatch(
        pack_dispatch_payload(
            latent=latent, topk_weights=topk_weights, topk_ids=topk_ids
        )
    )
    latent, topk_weights, topk_ids = unpack_dispatch_payload(
        payload, latent_dim=latent_dim, top_k=top_k
    )
    if on_global_topk_ids is not None:
        on_global_topk_ids(topk_ids)
    reduced_latent = comm.combine(experts(latent, topk_weights, topk_ids))

    # One attention-TP all-reduce sums the shared expert's partials and
    # assembles every rank's rows.
    output = wait_shared()
    if output is None:
        output = hidden_states.new_zeros(hidden_states.shape)
    up_proj_accumulate(reduced_latent, output[start:end])
    return comm.attn_tp_all_reduce(output)


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

    def attn_tp_all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return get_parallel().attn_tp_group.all_reduce(x)


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
