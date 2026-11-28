"""Muse Spark with attention data parallelism on ROCm: each rank's rows reach the other ranks' experts through MORI-EP.

Under attention DP at attention TP1 (``--attn-dp-size`` = ``--ep-size`` = ``--tp-size``) every rank runs its own
requests and holds 256 / EP of each layer's experts. A MoE layer routes and pools this rank's rows, dispatches every
row once to each rank holding one of its top-k experts, runs this rank's experts on the rows it received up to the
per-(row, expert) post-expert norm and the route-weighted partial sum, and the combine returns the partial sums to
the rows' source ranks, which add and unpool them. A rank without rows of its own still serves its peers.

The receive side is a fixed-size view whose first rows are live; the live count stays on the device (CUDA graphs).
A destination receives a row at most once, so the view is cut to the DP-synchronized per-rank row counts, and the
rows past the live count get expert id -1, which every expert kernel drops.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# The receive view is cut to a multiple of this many rows (the compact A4 experts take multiples of 8).
_ROW_ALIGN = 8
# Receive view sizes a process logs.
_LOGGED_VIEWS = 16


def enabled() -> bool:
    return get_parallel().attn_dp_size > 1


def validate() -> None:
    """The one layout this path runs: attention TP1 on every rank, EP over all ranks, rows moved by MORI-EP."""
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    parallel = get_parallel()
    if (
        parallel.attn_tp_size != 1
        or parallel.attn_cp_size != 1
        or parallel.moe_ep_size != parallel.tp_size
        or parallel.pp_size != 1
    ):
        raise NotImplementedError(
            "Muse Spark attention DP on ROCm runs attention TP1 with EP over all ranks: "
            "--attn-dp-size and --ep-size equal to --tp-size, no context or pipeline parallelism "
            f"(got tp {parallel.tp_size}, attention dp {parallel.attn_dp_size}, attention tp "
            f"{parallel.attn_tp_size}, attention cp {parallel.attn_cp_size}, ep {parallel.moe_ep_size}, "
            f"pp {parallel.pp_size})"
        )
    if not get_moe_a2a_backend().is_mori():
        raise NotImplementedError(
            "Muse Spark attention DP on ROCm moves the expert rows through MORI-EP: "
            "use --moe-a2a-backend mori"
        )
    if not parallel.enable_dp_lm_head:
        raise NotImplementedError(
            "Muse Spark attention DP on ROCm keeps the LM head on each rank: use --enable-dp-lm-head"
        )
    from sglang.srt.rocm_muse_spark import mxfp4_moe

    if mxfp4_moe.GFX1250_A8W4:
        raise NotImplementedError(
            "Muse Spark attention DP on ROCm is implemented for the gfx950 expert kernels"
        )


def _speculative() -> bool:
    from sglang.srt.runtime_context import get_spec

    return get_spec().speculative_algorithm is not None


def verify_step() -> bool:
    """Whether every rank runs a target-verify batch this step (a speculative server with no rank extending)."""
    from sglang.srt.layers.dp_attention import get_is_extend_in_batch

    return _speculative() and not get_is_extend_in_batch()


def counts_are_rows() -> bool:
    """Whether the DP-synchronized per-rank counts are row counts on every rank. Not on a speculative server while
    some rank extends: the scheduler gathers tokens from an extending rank and requests from a decoding one, and
    each rank scales the whole list by its own batch's rows per request."""
    from sglang.srt.layers.dp_attention import get_is_extend_in_batch

    return not (_speculative() and get_is_extend_in_batch())


def drop_padding_rows(topk_ids: torch.Tensor) -> None:
    """Expert id -1 (not dispatched) on the rows that only pad this rank's batch to a captured size."""
    from sglang.srt.layers.moe.topk import _mask_topk_ids_padded_region
    from sglang.srt.model_executor.forward_context import (
        get_forward_context,
        has_forward_context,
    )

    if not has_forward_context():
        return
    bound = get_forward_context().moe_num_token_non_padded
    if bound is not None:
        _mask_topk_ids_padded_region(topk_ids, bound())


def _capturing(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and torch.cuda.is_current_stream_capturing()


class Received(NamedTuple):
    """The rows dispatched to this rank: fixed-size views, the first ``total`` rows live."""

    rows: torch.Tensor  # bf16 [cap, expert hidden]
    scores: torch.Tensor  # fp32 [cap, top-k] routing weights of the rows' experts
    # int32 [cap, top-k] global expert ids, -1 on the rows past ``total``
    ids: torch.Tensor
    total: torch.Tensor  # int32 [1] on the device


def receive_rows(local_rows: int, physical: int, total: torch.Tensor) -> int:
    """Rows of the receive view worth running, rounded up: at most the rows all ranks dispatch, known from the
    DP-synchronized per-rank counts without a device read. Where those are not row counts (``counts_are_rows``)
    the step runs eagerly and the live count ``total`` is read. The whole view when neither is at hand."""
    from sglang.srt.layers.dp_attention import get_dp_global_num_tokens

    parallel = get_parallel()
    if counts_are_rows():
        counts = get_dp_global_num_tokens()
        if (
            not isinstance(counts, (list, tuple))
            or len(counts) != parallel.attn_dp_size
            or not all(isinstance(count, int) and count >= 0 for count in counts)
            or local_rows > counts[parallel.attn_dp_rank]
        ):
            return physical
        bound = sum(counts)
    elif _capturing(total):
        return physical
    else:
        bound = int(total.reshape(-1)[0].item())
    return min(physical, -(-max(bound, 1) // _ROW_ALIGN) * _ROW_ALIGN)


class ExpertDispatch:
    """MORI-EP dispatch / combine of one process, shared by its MoE layers (one dispatch-combine pair at a time)."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        from sglang.srt.layers.moe.token_dispatcher.moriep import MoriEPDispatcher
        from sglang.srt.layers.moe.utils import get_deepep_mode

        parallel = get_parallel()
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.max_rows = envs.SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        self._origin = None
        self._prepared = False
        self._logged: set = set()
        self.dispatcher = MoriEPDispatcher(
            group=parallel.tp_group,
            router_topk=top_k,
            permute_fusion=True,
            num_experts=num_experts,
            num_local_experts=num_experts // parallel.moe_ep_size,
            hidden_size=hidden_size,
            params_dtype=torch.bfloat16,
            deepep_mode=get_deepep_mode(),
            async_finish=True,
            return_recv_hook=True,
        )

    def prepare(self) -> None:
        """Build the MORI op (collective, allocates its buffers): after loading, before the KV pool is sized and
        before any graph is captured. The rows travel as BF16 both ways."""
        if self._prepared:
            return
        self._prepared = True
        from sglang.srt.layers.moe.token_dispatcher.moriep import (
            CombineDtype,
            DispatchDtype,
        )

        impl = getattr(self.dispatcher, "_normal_dispatcher", None)
        if impl is None or hasattr(self.dispatcher, "_low_latency_dispatcher"):
            raise NotImplementedError(
                "Muse Spark attention DP runs the MORI-EP normal kernels: use --deepep-mode normal"
            )
        self.dispatcher.set_quant_config({})
        if (
            impl.dispatch_dtype != DispatchDtype.bf16
            or impl.combine_dtype != CombineDtype.bf16
        ):
            raise NotImplementedError(
                "Muse Spark attention DP moves BF16 rows through MORI-EP; unset SGLANG_MORI_DISPATCH_DTYPE / "
                f"SGLANG_MORI_COMBINE_DTYPE (resolved {impl.dispatch_dtype} / {impl.combine_dtype})"
            )
        # Reading the op builds it.
        impl.mori_op
        parallel = get_parallel()
        logger.info(
            "Muse DP + EP: MORI-EP %s op ready on rank %d / %d (%d rows per rank, BF16 dispatch and combine)",
            "v2" if impl._is_epv2 else "v1",
            parallel.moe_ep_rank,
            parallel.moe_ep_size,
            self.max_rows,
        )

    def dispatch(
        self, expert_input: torch.Tensor, scores: torch.Tensor, ids: torch.Tensor
    ) -> Received:
        """Send this rank's rows [n, expert hidden] (``ids`` int32 [n, top-k] global expert ids, -1 on rows not to
        send; ``scores`` their routing weights) and return what this rank received. ``n`` may be 0."""
        from sglang.kernels.ops.moe.fill_padded_rows import _fill_padded_rows
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        rows = expert_input.shape[0]
        if rows > self.max_rows:
            raise RuntimeError(
                f"Muse DP + EP: {rows} rows on one rank exceed SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="
                f"{self.max_rows}; raise it to the largest per-rank batch (prefill chunk or decode / verify rows)"
            )
        out = self.dispatcher.dispatch(
            expert_input, StandardTopKOutput(scores, ids, None)
        )
        if out.hidden_states_scale is not None:
            raise NotImplementedError(
                "Muse DP + EP expects BF16 rows from the MORI-EP dispatch"
            )
        self._origin = (out.origin_topk_ids, out.origin_topk_weights)
        physical = out.hidden_states.shape[0]
        total = out.num_recv_tokens_per_expert
        cap = receive_rows(rows, physical, total)
        if len(self._logged) < _LOGGED_VIEWS and cap not in self._logged:
            self._logged.add(cap)
            logger.info(
                "Muse DP + EP: %d local rows, receive view %d of %d rows",
                rows,
                cap,
                physical,
            )
        recv_ids = out.topk_ids[:cap]
        # Rows past the live count hold what an earlier dispatch left: no expert of this rank may run them.
        _fill_padded_rows(recv_ids, total, -1)
        return Received(
            out.hidden_states[:cap], out.topk_weights[:cap], recv_ids, total
        )

    def combine(self, partial_sums: torch.Tensor) -> torch.Tensor:
        """Return the received rows' partial sums [cap, expert hidden] to their sources; the sums of this rank's
        own rows [n, expert hidden] come back (a view the next combine overwrites)."""
        from sglang.srt.layers.moe.token_dispatcher.moriep import (
            MoriEPNormalCombineInput,
        )

        origin_ids, origin_scores = self._origin
        self._origin = None
        return self.dispatcher.combine(
            MoriEPNormalCombineInput(partial_sums, origin_ids, origin_scores)
        )


_DISPATCH: ExpertDispatch | None = None


def expert_dispatch(hidden_size: int, num_experts: int, top_k: int) -> ExpertDispatch:
    """The process's dispatch, built by its first MoE layer."""
    global _DISPATCH
    if _DISPATCH is None:
        _DISPATCH = ExpertDispatch(hidden_size, num_experts, top_k)
    elif (_DISPATCH.hidden_size, _DISPATCH.top_k) != (hidden_size, top_k):
        raise ValueError("Muse DP + EP: MoE layers of different widths")
    return _DISPATCH
