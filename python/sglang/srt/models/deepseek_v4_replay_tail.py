"""DeepSeek-V4.1's decoder-replay tail layers, run from row-bucket graphs.

Eager prefill steps replay them directly. A prefill-graph shape of at least
``SGLANG_DSV4_DECODER_TAIL_GRAPH_MIN_PREFILL_TOKENS`` tokens replays them from one
eager break, through bridge buffers; smaller shapes keep the tail captured.

Under DP attention every rank replays the largest rank's bucket. In an eager step
MegaMoE agrees on it through a host gather; the prefill-graph break reads every
rank's rows from the pre-forward exchange.

Interleave CP prefills keep the prefix in BCG and execute the exact CP tail
at one eager break. A fixed full-shard bridge preserves the outer graph
and DSpark output coordinates. Eligible CP tails replay row-bucket graphs;
capture warmup and unsupported layouts keep the eager fallback. Tail graphs
are warmed and captured at worker startup; serving never captures a missing
bucket or variant.
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from sglang.srt.distributed.utils import all_gather_single
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.cp.interleave import (
    InterleaveContextParallelMetadata,
    attn_cp_interleave_gather,
)
from sglang.srt.layers.dp_attention import (
    dp_gather_replicate,
    note_rank_barrier_in_prefill_graph,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_context import (
    forward_context,
    get_attn_backend,
    get_forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
)
from sglang.srt.model_executor.runner.row_bucket_graph_runner import (
    RowBucketGraphRunner,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg_module,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
    get_tc_piecewise_forward_context,
    set_tc_piecewise_forward_context,
)
from sglang.srt.model_executor.runner_utils import capture_mode
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.model_executor.runner_utils.prefill_graph import (
    is_in_prefill_graph_scope,
)
from sglang.srt.models import deepseek_v4_boundary as hc_boundary
from sglang.srt.models.deepseek_common.decoder_swa_bounded_replay import (
    enter_late_layer_dp,
    exit_late_layer_dp,
)
from sglang.srt.models.deepseek_v4_mhc import HcState
from sglang.srt.models.deepseek_v4_tail_graph_startup import capture_decoder_tail_graphs
from sglang.srt.runtime_context import get_exec, get_flags, get_forward, get_parallel
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_in_replay_tail = False


def in_replay_tail_graph() -> bool:
    """True while the tail layers capture or replay a decoder tail graph."""
    return _in_replay_tail


@contextmanager
def _replay_tail_scope():
    global _in_replay_tail
    _in_replay_tail = True
    try:
        yield
    finally:
        _in_replay_tail = False


def scattered_dp_counts(
    num_tokens: int, num_rows: int, tail_graph_compatible: bool
) -> Tuple[int, List[int], bool]:
    """Gather sizes and replay eligibility when every rank runs the step eagerly."""
    group = get_parallel().tp_group.cpu_group
    counts = torch.empty(
        3 * torch.distributed.get_world_size(group=group), dtype=torch.int64
    )
    all_gather_single(
        counts,
        torch.tensor([num_tokens, num_rows, tail_graph_compatible], dtype=torch.int64),
        group=group,
    )
    tokens, rows, tail_graph_votes = counts.view(-1, 3).T.tolist()
    return max(tokens), rows, all(tail_graph_votes)


def _tail_kv_store(attention, x, positions, qkv_a) -> None:
    # The SWA store writes the step's live tail slots, so a tail graph breaks here.
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    num_rows = forward_batch.global_num_token_non_padded_cpu
    if num_rows == 0:
        return
    attention._compute_kv_to_cache(
        x[:num_rows],
        positions[:num_rows],
        forward_batch,
        get_attn_backend(),
        qkv_a=None if qkv_a is None else qkv_a[:num_rows],
    )


bcg_tail_kv_store = eager_on_graph(True)(_tail_kv_store)


def gather_cp_tail_ids(
    input_ids: torch.Tensor, vl_routing: bool
) -> Optional[torch.Tensor]:
    """Match the rank-major CP MoE operands, including each rank's padding."""
    if not vl_routing:
        return None
    return attn_cp_interleave_gather(input_ids[:, None]).squeeze(-1)


def _flatten_state(state: HcState):
    """The state's row tensors, whether ``pre`` is among them, and the inverse.
    The normed input and mix stats are dropped (the first tail layer rebuilds
    them), so every step keys the same graphs."""
    assert isinstance(state.streams, torch.Tensor), "tail graphs are CUDA only"
    has_pre = state.pre is not None
    leaves = [state.streams, state.pre] if has_pre else [state.streams]

    def rebuild(tensors: List[torch.Tensor]) -> HcState:
        return HcState(tensors[0], tensors[1] if has_pre else None)

    return leaves, has_pre, rebuild


class DecoderReplayTail:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        layer_ids: range,
        row_step: int,
        max_rows: int,
    ) -> None:
        self._model = model
        self._layer_ids = layer_ids
        self._layers = None
        self.runner = RowBucketGraphRunner(
            row_step=row_step,
            max_rows=max_rows,
            label="Decoder tail graph",
            log_capture=False,
        )
        self._startup_capture = False
        self._startup_prepared = False
        self._startup_warming = False
        self._last_run_replayed = False
        self._static_ids_global: dict[torch.dtype, torch.Tensor] = {}
        # Older, smaller sets stay alive for the graphs that write and read them.
        self._bridge_buffers: dict[tuple, List[List[Optional[torch.Tensor]]]] = {}

    def capture_at_startup(
        self, model_runner, eager_runner, prefill_runner=None
    ) -> tuple[float, float]:
        """Warm and capture before the worker is ready; returns GiB and seconds."""
        return capture_decoder_tail_graphs(
            self, model_runner, eager_runner, prefill_runner
        )

    @classmethod
    def maybe_create(cls, model: torch.nn.Module) -> Optional[DecoderReplayTail]:
        # Keep the CP BCG bridge when dedicated tail graphs are not requested.
        max_rows = (
            envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS.get()
            if get_exec().graph.enable_decoder_tail_cuda_graph
            else 0
        )
        row_graphs = max_rows > 0 and not envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.get()
        if not (model.tail_writes_source_kv and is_cuda()):
            return None
        if not row_graphs and not (
            get_parallel().attn_cp_size > 1
            and check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE)
        ):
            return None
        from sglang.srt.layers.attention.deepseek_v4_backend import SWA_WINDOW

        # The budget is in global tail rows; CP owns a fraction on each rank.
        cp_size = get_parallel().attn_cp_size
        row_step = max(1, (SWA_WINDOW + cp_size - 1) // cp_size)
        return cls(
            model=model,
            layer_ids=range(model.late_layer_start, model.end_layer),
            row_step=row_step,
            # A zero-capacity runner still owns CP BCG's full-shard bridge.
            max_rows=max(0, max_rows // cp_size),
        )

    @staticmethod
    def _can_capture_moe_bound(forward_batch: ForwardBatch) -> bool:
        # LOCAL, attention-TP GLOBAL and CP-local counts have graph-owned buffers.
        bound = forward_batch.moe_num_token_non_padded()
        return (
            bound is None
            or bound is forward_batch.num_token_non_padded
            or (
                isinstance(
                    forward_batch.attn_cp_metadata, InterleaveContextParallelMetadata
                )
                and bound is forward_batch.attn_cp_metadata.moe_local_token_count
            )
            or (
                forward_batch.attn_tp_sequence_sharded
                and bound is forward_batch.global_num_token_non_padded
            )
        )

    @staticmethod
    def _can_capture_cp_tail(tail: LateLayerTail) -> bool:
        # NONE uses fixed-shape rank-major CP gather/reduce-scatter; MegaMoE
        # keeps the local rows. Both leave live attention metadata in eager breaks.
        cp = tail.cp_metadata
        parallel = get_parallel()
        return (
            isinstance(cp, InterleaveContextParallelMetadata)
            and cp.local_index is not None
            and cp.gather_index is not None
            and parallel.attn_cp_size > 1
            and parallel.attn_tp_size == 1
            and parallel.attn_dp_size == 1
            and (get_moe_a2a_backend().is_megamoe() or get_moe_a2a_backend().is_none())
            and cp.per_rank_logical_token is not None
            and len(cp.per_rank_logical_token) == parallel.attn_cp_size
            and cp.per_rank_actual_token is not None
            and len(cp.per_rank_actual_token) == parallel.attn_cp_size
            and len(set(cp.per_rank_actual_token)) == 1
            and all(
                0 <= n <= cp.per_rank_actual_token[0] for n in cp.per_rank_logical_token
            )
            and cp.per_rank_actual_token[0] == tail.positions.shape[0]
        )

    def plan(
        self,
        tail: Optional[LateLayerTail],
        *,
        forward_batch: ForwardBatch,
        num_tokens: int,
        tail_in_graph: bool,
    ) -> Tuple[bool, Optional[List[int]]]:
        """Whether this step uses a tail graph and, under DP attention, every rank's
        tail rows; also feeds the warm gate."""
        from sglang.srt.layers.attention.dsv4.unified_kv_gate import (
            is_unified_kv_triton,
        )

        dp = get_parallel().attn_dp_size > 1
        if not (tail is not None or (dp and forward_batch.is_extend_in_batch)):
            return False, None
        if forward_batch.dp_spec_prefill_coordination_applied:
            return False, None
        if (
            tail is not None
            and tail.cp_metadata is not None
            and get_flags().dp.capturing_prefill_graph
        ):
            # Startup's padded outer buckets are not live prefill warmup.
            return False, None
        scattered = self._scattered_dp(
            tail,
            tail_in_graph=tail_in_graph,
            forward_batch=forward_batch,
            num_tokens=num_tokens,
        )
        if scattered is not None:
            self.runner.ready(scattered[0])
        else:
            counts = forward_batch.global_num_tokens_cpu
            warm_tokens = max(counts) if dp and counts else num_tokens
            if tail is not None and tail.cp_metadata is not None:
                # Model arguments are already CP-local; the warm gate is global.
                warm_tokens *= get_parallel().attn_cp_size
            self.runner.ready(warm_tokens)
        if (
            tail_in_graph
            or self._startup_warming
            or capture_mode.is_capture_mode
            or bcg_module._current_capture_var.get() is not None
            or envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.get()
            or is_unified_kv_triton()
            or get_global_expert_distribution_recorder().recording
            or not self.runner.warm
            or (
                tail is not None
                and tail.cp_metadata is not None
                and not self._can_capture_cp_tail(tail)
            )
            or not self._can_capture_moe_bound(forward_batch)
        ):
            return False, None
        if not dp:
            return (
                tail is not None
                and self.runner.bucket_rows(tail.positions.shape[0]) is not None,
                None,
            )
        if scattered is not None:
            _, dp_rows, can_replay = scattered
        elif get_moe_a2a_backend().is_none() and forward_batch.is_extend_in_batch:
            dp_rows = forward_batch.global_late_layer_num_tokens_cpu
            can_replay = forward_batch.global_late_layer_tail_graph_compatible
        else:
            return False, None
        # A live decode rank has no prefill tail. Fall back on every rank so
        # that it keeps its rows and the tail collectives still agree.
        if (
            not can_replay
            or dp_rows is None
            or self.runner.bucket_rows(max(dp_rows)) is None
        ):
            return False, None
        return True, dp_rows

    def _scattered_dp(
        self,
        tail: Optional[LateLayerTail],
        *,
        tail_in_graph: bool,
        forward_batch: ForwardBatch,
        num_tokens: int,
    ) -> Optional[Tuple[int, List[int], bool]]:
        """`scattered_dp_counts` for a MoE routing each rank's own rows. Such a MoE
        lets ranks pick prefill graph or eager independently, so it needs the
        prefill graph disabled."""
        if (
            get_parallel().attn_dp_size == 1
            or get_moe_a2a_backend().is_none()
            or not check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)
            or not forward_batch.is_extend_in_batch
            or tail_in_graph
            or get_is_capture_mode()
        ):
            return None
        return scattered_dp_counts(
            num_tokens,
            0 if tail is None else tail.positions.shape[0],
            tail is not None or forward_batch.forward_mode.is_idle(),
        )

    def breaks_out_of_prefill_graph(
        self, tail: LateLayerTail, *, num_tokens: int
    ) -> bool:
        from sglang.srt.layers.attention.dsv4.unified_kv_gate import (
            is_unified_kv_triton,
        )

        return (
            num_tokens >= envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MIN_PREFILL_TOKENS.get()
            and tail.cp_metadata is None
            and not is_unified_kv_triton()
            # Encoder replay's folded rows are not checked against the exact tail.
            and get_attn_backend().token_to_kv_pool.request_window is None
        )

    def run(
        self,
        *,
        state: HcState,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_global: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        capture_dspark: bool,
        dp_rows: Optional[List[int]] = None,
        out_rows: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """Returns views of graph buffers valid until the next replay, cut to
        ``out_rows`` rows. Under DP every rank replays the bucket of the largest
        ``dp_rows``."""
        replay_batch = forward_batch
        replay_batch.tail_cuda_graph_replayed = False
        if forward_batch.forward_mode.is_idle():
            # An idle DP rank runs the extend's graph, with no rows of its own.
            forward_batch = copy.copy(forward_batch)
            forward_batch.forward_mode = ForwardMode.EXTEND
        cp_metadata = forward_batch.attn_cp_metadata
        cp_tail = (
            isinstance(cp_metadata, InterleaveContextParallelMetadata)
            and cp_metadata.local_index is not None
        )
        local_num_rows = None
        cp_size = None
        if cp_tail:
            assert dp_rows is None, "CP tail graphs do not support attention DP"
            # Keep collective padding as physical rows, but mask it out of MoE.
            cp_size = len(cp_metadata.per_rank_actual_token)
            local_num_rows = cp_metadata.per_rank_logical_token[
                get_parallel().attn_cp_rank
            ]
            forward_batch = copy.copy(forward_batch)
            if forward_batch.num_token_non_padded is None:
                forward_batch.num_token_non_padded = torch.zeros(
                    (), dtype=torch.int32, device=positions.device
                )
        cp_gather = cp_tail and get_moe_a2a_backend().is_none()
        gather = dp_rows is not None and get_moe_a2a_backend().is_none()
        # Ranks in the DP layout the MoE sees: all of them when it gathers.
        layout = None if dp_rows is None else (len(dp_rows) if gather else 1)
        gather_ids = (gather or cp_gather) and self._model.late_layer_vl_routing
        if input_ids_global is None:
            input_ids_global = input_ids
        leaves, has_pre, rebuild = _flatten_state(state)
        num_state = len(leaves)
        live = [*leaves, positions, input_ids]
        if not (gather or cp_gather):
            live.append(input_ids_global)
        # Eager and outer-BCG forwards can choose different fused producers.
        # Keep their recordings separate even when tensor shapes are identical.
        key = (
            capture_dspark,
            has_pre,
            layout,
            gather_ids,
            cp_size,
            cp_gather,
            get_forward().is_extend_in_batch,
            is_in_prefill_graph_scope(),
        )
        global_num_rows = (
            positions.shape[0]
            if forward_batch.attn_tp_sequence_sharded
            and forward_batch.global_num_token_non_padded is not None
            and forward_batch.moe_num_token_non_padded()
            is forward_batch.global_num_token_non_padded
            else None
        )
        self._last_run_replayed = False
        if not self._startup_capture and not self.runner.can_replay(
            key=key,
            live=live,
            forward_batch=forward_batch,
            bucket_from=None if dp_rows is None else max(dp_rows),
            global_num_rows=global_num_rows,
        ):
            return self._run_layers_eagerly(
                state=state,
                positions=positions,
                input_ids=input_ids,
                input_ids_global=input_ids_global,
                forward_batch=forward_batch,
                capture_dspark=capture_dspark,
                dp_rows=dp_rows,
            )
        ids_global = (
            self._ids_global_buffer(input_ids.dtype, len(dp_rows))
            if gather and gather_ids
            else None
        )

        def body(inputs: List[torch.Tensor], batch: ForwardBatch):
            rows = inputs[0].shape[0]
            if cp_gather:
                # Gather bucket-sized IDs inside the graph, just like MoE rows.
                # Live global IDs have cp_size * live_rows, not bucket_rows.
                ids = gather_cp_tail_ids(inputs[num_state + 1], gather_ids)
            elif not gather:
                ids = inputs[num_state + 2]
            else:
                ids = None if ids_global is None else ids_global[: rows * layout]
            residual, pre, aux = self._layer_stack(
                rebuild(inputs[:num_state]),
                positions=inputs[num_state],
                input_ids=inputs[num_state + 1],
                input_ids_global=ids,
                forward_batch=batch,
                capture_dspark=capture_dspark,
            )
            return [residual, pre, *aux]

        @contextmanager
        def dp_layout(batch: ForwardBatch, inputs: List[torch.Tensor]):
            if cp_tail:
                # CP KV gathers/indexers read the live batch in eager breaks.
                # Captured MoE kernels read a graph-owned local count.
                saved_cp = batch.attn_cp_metadata
                if cp_gather:
                    logical = saved_cp.per_rank_logical_token[
                        get_parallel().attn_cp_rank
                    ]
                    # NONE routes every gathered row; clear stale bucket padding.
                    for value in inputs:
                        value[logical:].zero_()
                batch.attn_cp_metadata = copy.copy(saved_cp)
                batch.attn_cp_metadata.moe_local_token_count = (
                    batch.num_token_non_padded
                )
                try:
                    yield
                finally:
                    batch.attn_cp_metadata = saved_cp
                return
            if layout is None:
                yield
                return
            rows = inputs[0].shape[0]
            graph_count = batch.num_token_non_padded
            saved = enter_late_layer_dp(batch, [rows] * layout)
            if layout == 1 and graph_count is not None:
                # A local MoE masks pad rows by the graph's count, not the bucket.
                batch.global_num_token_non_padded = graph_count
                batch.num_token_non_padded = graph_count
            try:
                if ids_global is not None:
                    dp_gather_replicate(
                        ids_global[: batch.global_dp_buffer_len, None],
                        inputs[num_state + 1][:, None].clone(),
                        batch,
                    )
                yield
            finally:
                exit_late_layer_dp(batch, saved)

        residual, pre, *aux = self.runner.run(
            key=key,
            live=live,
            forward_batch=forward_batch,
            body=body,
            context=self._break_context,
            batch_scope=dp_layout,
            bucket_from=None if dp_rows is None else max(dp_rows),
            out_rows=out_rows,
            local_num_rows=local_num_rows,
            # mHC tail inputs contain the whole attention-DP sequence before
            # any MoE transport, including when attention TP shards its inputs.
            global_num_rows=global_num_rows,
            allow_capture=self._startup_capture,
            capture_all_buckets=False,
        )
        self._last_run_replayed = True
        replay_batch.tail_cuda_graph_replayed = True
        return residual, pre, aux

    def run_idle(
        self,
        *,
        state: HcState,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_global: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        capture_dspark: bool,
        dp_rows: List[int],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """An idle DP rank replays the shared bucket with no rows of its own."""
        return self.run(
            state=state.take_rows(lambda t: t[:0]),
            positions=positions[:0],
            input_ids=input_ids[:0],
            input_ids_global=(
                None if input_ids_global is None else input_ids_global[:0]
            ),
            forward_batch=forward_batch,
            capture_dspark=capture_dspark,
            dp_rows=dp_rows,
            out_rows=state.residual.shape[0],
        )

    def _run_layers_eagerly(
        self,
        *,
        state: HcState,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_global: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        capture_dspark: bool,
        dp_rows: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """For a prefill-graph step whose tail has no graph to replay."""
        if forward_batch.forward_mode.is_idle():
            forward_batch = copy.copy(forward_batch)
            forward_batch.forward_mode = ForwardMode.EXTEND
        tail_batch = copy.copy(forward_batch)
        tail_batch.global_num_token_non_padded_cpu = positions.shape[0]
        gather = dp_rows is not None and get_moe_a2a_backend().is_none()
        if input_ids_global is None or gather:
            input_ids_global = input_ids if not gather else None
        saved_dp = None
        if dp_rows is not None:
            saved_dp = enter_late_layer_dp(tail_batch, dp_rows)
        try:
            if (
                isinstance(
                    tail_batch.attn_cp_metadata, InterleaveContextParallelMetadata
                )
                and tail_batch.attn_cp_metadata.local_index is not None
                and get_moe_a2a_backend().is_none()
            ):
                input_ids_global = gather_cp_tail_ids(
                    input_ids, self._model.late_layer_vl_routing
                )
            if gather and self._model.late_layer_vl_routing:
                gathered = input_ids.new_empty((sum(dp_rows), 1))
                dp_gather_replicate(gathered, input_ids[:, None].clone(), tail_batch)
                input_ids_global = gathered.squeeze(-1)
            with (
                (
                    forward_context(get_forward_context(), forward_batch=tail_batch)
                    if has_forward_context()
                    else nullcontext()
                ),
                self._break_context(tail_batch),
            ):
                return self._layer_stack(
                    state,
                    positions=positions,
                    input_ids=input_ids,
                    input_ids_global=input_ids_global,
                    forward_batch=tail_batch,
                    capture_dspark=capture_dspark,
                )
        finally:
            if saved_dp is not None:
                exit_late_layer_dp(tail_batch, saved_dp)

    def run_in_prefill_graph(
        self,
        *,
        state: HcState,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_global: Optional[torch.Tensor],
        capture_dspark: bool,
        block_rows: int,
        source_kv_layer_id: Optional[int],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """Runs the tail in one break of the outer prefill graph and returns views of
        the bridge over its ``block_rows`` block, which the break fills."""
        leaves, has_pre, rebuild = _flatten_state(state)
        bridge_key = (capture_dspark, has_pre)
        bridge = self._bridge(bridge_key, block_rows)
        # A capture records the bridge it is given; the warmup before it allocates one.
        assert bridge is not None or bcg_module._current_capture_var.get() is None, (
            "tail bridge not allocated before capture"
        )
        bcg_tail_layers_in_prefill_graph(
            self,
            rebuild,
            capture_dspark,
            block_rows,
            source_kv_layer_id,
            get_parallel().attn_dp_size > 1,
            bridge_key,
            bridge,
            leaves,
            positions,
            input_ids,
            input_ids if input_ids_global is None else input_ids_global,
        )
        if bridge is None:  # first run: the break allocated the bridge
            bridge = self._bridge(bridge_key, block_rows)
        residual, pre, *aux = bridge
        return residual, pre, aux

    def run_cp_in_prefill_graph(
        self,
        *,
        state: HcState,
        input_ids: torch.Tensor,
        capture_dspark: bool,
        source_kv_layer_id: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """Run a live CP tail at one break; expose fixed full-shard outputs.

        Early DSpark captures and the outer CP gather use the original shard
        coordinates. Scatter the exact tail back to those coordinates inside
        the break, so captured segments never depend on live CP tail lengths.
        """
        leaves, has_pre, rebuild = _flatten_state(state)
        key = (capture_dspark, has_pre)
        block_rows = input_ids.shape[0]
        bridge = self._bridge(key, block_rows)
        assert bridge is not None or bcg_module._current_capture_var.get() is None, (
            "CP tail bridge not allocated before capture"
        )
        bcg_cp_tail_layers_in_prefill_graph(
            self,
            rebuild,
            capture_dspark,
            source_kv_layer_id,
            key,
            bridge,
            leaves,
            input_ids,
        )
        if bridge is None:
            bridge = self._bridge(key, block_rows)
        residual, pre, *aux = bridge
        return residual, pre, aux

    def _bridge(
        self, key: Tuple[bool, bool], block_rows: int
    ) -> Optional[List[Optional[torch.Tensor]]]:
        """The newest bridge buffers for ``key`` cut to ``block_rows`` rows, or None
        when there are none yet or they are too small."""
        sets = self._bridge_buffers.get(key)
        if not sets or sets[0][0].shape[0] < block_rows:
            return None
        return [None if buf is None else buf[:block_rows] for buf in sets[0]]

    def _write_bridge(
        self,
        key: Tuple[bool, bool],
        block_rows: int,
        bridge: Optional[List[Optional[torch.Tensor]]],
        outputs: List[Optional[torch.Tensor]],
    ) -> None:
        """Copies the tail rows into ``bridge``, allocating a new one for ``key`` when
        it is None. Rows past the tail keep stale values, which flow only into rows
        that _scatter_tail_rows and the real_rows_count cut drop."""
        if bridge is None:
            bridge = [
                None if t is None else t.new_zeros((block_rows, *t.shape[1:]))
                for t in outputs
            ]
            self._bridge_buffers.setdefault(key, []).insert(0, bridge)
        for buf, t in zip(bridge, outputs, strict=True):
            if t is None:
                assert buf is None, "tail outputs differ within one bridge key"
                continue
            assert (buf.dtype, buf.shape[1:]) == (t.dtype, t.shape[1:]), (
                (buf.dtype, buf.shape),
                (t.dtype, t.shape),
            )
            buf[: t.shape[0]].copy_(t)

    def _ids_global_buffer(self, dtype: torch.dtype, dp_size: int) -> torch.Tensor:
        buf = self._static_ids_global.get(dtype)
        if buf is None:
            buf = self._static_ids_global[dtype] = torch.zeros(
                (self.runner.max_rows * dp_size,), dtype=dtype, device="cuda"
            )
        return buf

    @contextmanager
    def _break_context(self, tail_batch: ForwardBatch):
        outer = get_tc_piecewise_forward_context()
        if outer is not None:
            # Nested in a prefill-graph break: swap only the batch the breaks read.
            saved_batch = outer.forward_batch
            outer.forward_batch = tail_batch
            try:
                with _replay_tail_scope():
                    yield
            finally:
                outer.forward_batch = saved_batch
            return
        if self._layers is None:
            self._layers = compute_attention_and_moe_layers(self._model)
        layers = self._layers
        with (
            set_tc_piecewise_forward_context(
                tail_batch,
                layers.attention_layers,
                None,
                layers.moe_layers,
                layers.moe_fusions,
                dsa_indexers=layers.dsa_indexers,
                mha_companion_layers=layers.mha_companion_layers,
            ),
            enable_breakable_cuda_graph(),
            _replay_tail_scope(),
        ):
            yield

    def _layer_stack(
        self,
        state: HcState,
        *,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_global: torch.Tensor,
        forward_batch: ForwardBatch,
        capture_dspark: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """The graphs' body; mirrors the model loop past the tail switch."""
        model = self._model
        aux: List[torch.Tensor] = []
        residual_batch.start(forward_batch)
        hidden_states = residual_batch.set_written(state, forward_batch)
        try:
            for i in self._layer_ids:
                layer = model.layers[i]
                assert layer.engram is None
                if capture_dspark and i in model.dspark_layers_to_capture:
                    call = hc_boundary.LayerCall(input_ids, input_ids_global)
                    next_boundary = (
                        hc_boundary.HcRead(layer, True)
                        .bind(forward_batch, call)
                        .next_boundary
                    )
                    hidden_states = state = hc_boundary.materialize(
                        hidden_states, forward_batch, next=next_boundary
                    )
                    from sglang.kernels.ops.layernorm.mhc import mhc_stream_mean

                    # The draft head reads the attention input of its target layers.
                    aux.append(mhc_stream_mean(state.residual))
                ctx = (
                    nullcontext()
                    if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    hidden_states = layer(
                        positions=positions,
                        hidden_states=hidden_states,
                        input_ids=input_ids,
                        forward_batch=forward_batch,
                        input_ids_global=input_ids_global,
                    )
            state = residual_batch.read_output(
                hidden_states, forward_batch, hc_boundary.FinalRead(), None
            )
        finally:
            residual_batch.clear(forward_batch)
        state = state.materialized(model.layers[self._layer_ids[-1]].hc_cfg)
        return state.residual, state.pre, aux


def _unpadded_batch(forward_batch: ForwardBatch) -> ForwardBatch:
    """The batch cut to its live requests and tokens (the graph pads both)."""
    extend_lens = forward_batch.extend_seq_lens_cpu
    if extend_lens is None:
        return forward_batch
    bs, num_tokens = len(extend_lens), int(sum(extend_lens))
    view = copy.copy(forward_batch)
    view.batch_size = bs
    for name in (
        "req_pool_indices",
        "seq_lens",
        "seq_lens_cpu",
        "extend_seq_lens",
        "extend_prefix_lens",
        "extend_start_loc",
    ):
        value = getattr(forward_batch, name, None)
        if value is not None:
            setattr(view, name, value[:bs])
    for name in ("positions", "out_cache_loc", "input_ids"):
        value = getattr(forward_batch, name, None)
        if value is not None:
            setattr(view, name, value[:num_tokens])
    return view


def _enter_eager_tail_in_prefill_graph(
    backend, graph_batch: ForwardBatch, source_kv_layer_id: Optional[int]
) -> Tuple[tuple, ForwardBatch]:
    """Switches onto exact-row tail metadata; returns the exit state and the
    batch the tail runs on (the dummy batch during the graph warmup)."""
    forward_batch = _unpadded_batch(backend.prefill_graph_replay_batch or graph_batch)
    held = backend.tail_forward_metadata
    backend.tail_forward_metadata = backend._build_late_layer_tail_metadata(
        forward_batch
    )
    try:
        saved = backend.enter_late_layer_tail(forward_batch)
    finally:
        backend.tail_forward_metadata = held
    backend.tail_source_kv_layer_id = source_kv_layer_id
    return saved, forward_batch


def _tail_layers_in_prefill_graph(
    tail_layers: DecoderReplayTail,
    rebuild: Callable[[List[torch.Tensor]], HcState],
    capture_dspark: bool,
    block_rows: int,
    source_kv_layer_id: Optional[int],
    dp: bool,
    bridge_key: Tuple[bool, bool],
    bridge: Optional[List[Optional[torch.Tensor]]],
    leaves: List[torch.Tensor],
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    input_ids_global: torch.Tensor,
) -> None:
    # The tail captures its row-bucket graphs on first use.
    note_rank_barrier_in_prefill_graph()
    backend = get_attn_backend()
    graph_batch = get_tc_piecewise_forward_context().forward_batch
    saved, forward_batch = _enter_eager_tail_in_prefill_graph(
        backend, graph_batch, source_kv_layer_id
    )
    try:
        tail = backend.forward_metadata.late_layer_tail
        # The graph's rows are padded to its shape; the tail indexes live rows.
        num_tokens = forward_batch.input_ids.shape[0]

        def rows(t: torch.Tensor) -> torch.Tensor:
            return tail.rows(t[:num_tokens])

        num_rows = tail.positions.shape[0]
        dp_rows = None
        if dp:
            # Exchanged before the forward, except on the dummy batches the prefill
            # graphs warm up on (every rank together), when it gathers here.
            dp_rows = forward_batch.global_late_layer_num_tokens_cpu
            if dp_rows is None:
                counts = torch.empty(get_parallel().tp_size, dtype=torch.int64)
                all_gather_single(
                    counts,
                    torch.tensor([num_rows], dtype=torch.int64),
                    group=get_parallel().tp_group.cpu_group,
                )
                dp_rows = counts.tolist()
            assert dp_rows[get_parallel().attn_dp_rank] == num_rows, (
                dp_rows,
                num_rows,
            )
        tail_ids = rows(input_ids)
        # A TP MoE under DP gathers every rank's IDs inside the tail layers.
        if dp and get_moe_a2a_backend().is_none():
            ids_global = None
        else:
            ids_global = rows(input_ids_global)
        kwargs = dict(
            state=rebuild([rows(t) for t in leaves]),
            positions=tail.positions,
            input_ids=tail_ids,
            input_ids_global=ids_global,
            forward_batch=forward_batch,
            capture_dspark=capture_dspark,
            dp_rows=dp_rows,
        )
        # Capture stream and pool are busy while the prefill graphs capture.
        runner = tail_layers.runner
        use_tail_graph = (
            not get_flags().dp.capturing_prefill_graph
            and runner.warm
            and tail_layers._can_capture_moe_bound(forward_batch)
            and runner.bucket_rows(max(dp_rows) if dp else num_rows) is not None
        )
        if use_tail_graph:
            residual, pre, aux = tail_layers.run(**kwargs)
        else:
            residual, pre, aux = tail_layers._run_layers_eagerly(**kwargs)
        graph_batch.tail_cuda_graph_replayed = (
            use_tail_graph and tail_layers._last_run_replayed
        )
        tail_layers._write_bridge(
            bridge_key, block_rows, bridge, outputs=[residual, pre, *aux]
        )
    finally:
        backend.exit_late_layer_tail(saved, forward_batch)
        backend.prefill_graph_replay_batch = None


bcg_tail_layers_in_prefill_graph = eager_on_graph(True)(_tail_layers_in_prefill_graph)


def _cp_tail_layers_in_prefill_graph(
    tail_layers: DecoderReplayTail,
    rebuild: Callable[[list[torch.Tensor]], HcState],
    capture_dspark: bool,
    source_kv_layer_id: int | None,
    bridge_key: tuple[bool, bool],
    bridge: list[torch.Tensor | None] | None,
    leaves: list[torch.Tensor],
    input_ids: torch.Tensor,
) -> None:
    backend = get_attn_backend()
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    tail = backend.tail_forward_metadata.late_layer_tail
    assert tail.cp_metadata is not None and tail.global_token_indices is not None
    saved = backend.enter_late_layer_tail(forward_batch)
    backend.tail_source_kv_layer_id = source_kv_layer_id
    try:
        tail_ids = tail.rows(input_ids)
        if get_moe_a2a_backend().is_none():
            # The tail gathers local IDs in the same rank-major order as MoE.
            ids_global = None
        else:
            ids_global = tail_ids
        use_tail_graph = False
        if not get_flags().dp.capturing_prefill_graph:
            # The outer inputs have capture capacity. Feed the warm gate the
            # live prefill size instead, then reuse the normal tail eligibility.
            cp_size = get_parallel().attn_cp_size
            local_tokens = (
                sum(forward_batch.extend_seq_lens_cpu) + cp_size - 1
            ) // cp_size
            use_tail_graph, _ = tail_layers.plan(
                tail,
                forward_batch=forward_batch,
                num_tokens=local_tokens,
                tail_in_graph=False,
            )
        run_layers = (
            tail_layers.run if use_tail_graph else tail_layers._run_layers_eagerly
        )
        residual, pre, aux = run_layers(
            state=rebuild([tail.rows(t) for t in leaves]),
            positions=tail.positions,
            input_ids=tail_ids,
            input_ids_global=ids_global,
            forward_batch=forward_batch,
            capture_dspark=capture_dspark,
        )

        use_tail_graph = use_tail_graph and tail_layers._last_run_replayed
        forward_batch.tail_cuda_graph_replayed = use_tail_graph

        def scatter(value):
            if value is None:
                return None
            full = value.new_zeros((input_ids.shape[0], *value.shape[1:]))
            full.index_copy_(
                0, tail.token_indices, value[: tail.token_indices.shape[0]]
            )
            return full

        tail_layers._write_bridge(
            bridge_key,
            input_ids.shape[0],
            bridge,
            outputs=[scatter(t) for t in (residual, pre, *aux)],
        )
    finally:
        backend.exit_late_layer_tail(saved, forward_batch)


bcg_cp_tail_layers_in_prefill_graph = eager_on_graph(True)(
    _cp_tail_layers_in_prefill_graph
)
