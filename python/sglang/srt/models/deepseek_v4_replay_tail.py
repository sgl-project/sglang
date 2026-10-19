"""DeepSeek-V4.1's decoder-replay tail layers (``late_layer_start`` on), replayed
from `RowBucketGraphRunner` graphs on eager prefill steps. The SWA KV store, the
attention and the low-ratio sources stay eager breaks of those graphs.

A prefill-graph step runs them from one eager break of its own graph, on
exact-row tail metadata, and writes the tail rows into bridge buffers that the
later captured segments read.

Under DP attention every rank replays the bucket of the largest rank's tail: a
TP MoE gathers that many rows from every rank; MegaMoE sees only the local
bucket, and the ranks agree on it through a host gather. An idle rank takes part
with no rows of its own.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from sglang.srt.distributed.utils import all_gather_single
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.dp_attention import dp_gather_replicate
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend
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
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.models import deepseek_v4_boundary as hc_boundary
from sglang.srt.models.deepseek_common.decoder_swa_bounded_replay import (
    enter_late_layer_dp,
    exit_late_layer_dp,
)
from sglang.srt.models.deepseek_v4_mhc import HcState
from sglang.srt.runtime_context import get_flags, get_parallel
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

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


def scattered_dp_counts(num_tokens: int, num_rows: int) -> Tuple[int, List[int]]:
    """Under attention DP with a MoE that routes each rank's own rows, the largest
    step token count and every rank's tail rows, gathered on the host: those
    ranks' MoE kernels exchange tokens, so they must capture and replay together.
    The pre-forward exchange skips such a MoE (its token counts are local), so
    this gathers in the forward; it is only safe when every rank runs the step
    eagerly."""
    group = get_parallel().tp_group.cpu_group
    counts = torch.empty(
        2 * torch.distributed.get_world_size(group=group), dtype=torch.int64
    )
    all_gather_single(
        counts, torch.tensor([num_tokens, num_rows], dtype=torch.int64), group=group
    )
    tokens, rows = counts.view(-1, 2).T.tolist()
    return max(tokens), rows


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
            row_step=row_step, max_rows=max_rows, label="Decoder tail graph"
        )
        # Every rank's padded tail IDs under DP attention, for MoE routing on IDs.
        self._static_ids_global: dict[torch.dtype, torch.Tensor] = {}
        # Bridge outputs of prefill-graph steps, one set per output structure,
        # sized by the first (largest) captured shape; older sets stay alive for
        # the graphs that read them.
        self._bridge: dict[tuple, List[List[Optional[torch.Tensor]]]] = {}

    @classmethod
    def maybe_create(cls, model: torch.nn.Module) -> Optional[DecoderReplayTail]:
        max_rows = envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS.get()
        if not (
            model.tail_writes_source_kv
            and is_cuda()
            and max_rows > 0
            # Borrowed graph-pool storage forbids capturing mid-serving.
            and not envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.get()
        ):
            return None
        from sglang.srt.layers.attention.deepseek_v4_backend import SWA_WINDOW

        return cls(
            model=model,
            layer_ids=range(model.late_layer_start, model.end_layer),
            row_step=SWA_WINDOW,
            max_rows=max_rows,
        )

    def plan(
        self,
        tail: Optional[LateLayerTail],
        *,
        forward_batch: ForwardBatch,
        num_tokens: int,
        tail_in_graph: bool,
        late_dp_counts: Optional[List[int]],
    ) -> Tuple[bool, Optional[List[int]]]:
        """Whether this step runs its tail layers from a graph, and under DP
        attention every rank's tail rows; also reports the step's token count to
        the warm gate. Under DP every condition is rank-uniform, idle ranks
        included."""
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        dp = get_parallel().attn_dp_size > 1
        if not (tail is not None or (dp and forward_batch.is_extend_in_batch)):
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
            self.runner.ready(max(counts) if dp and counts else num_tokens)
        if (
            tail_in_graph
            or get_is_capture_mode()
            or is_unified_kv_triton()
            or get_global_expert_distribution_recorder().recording
            or not self.runner.warm
            or (tail is not None and tail.cp_metadata is not None)
        ):
            return False, None
        if not dp:
            return (
                tail is not None
                and self.runner.bucket_rows(tail.positions.shape[0]) is not None,
                None,
            )
        dp_rows = scattered[1] if scattered is not None else None
        if dp_rows is None:
            dp_rows = self._gathered_dp_rows(late_dp_counts, forward_batch)
        if dp_rows is None or self.runner.bucket_rows(max(dp_rows)) is None:
            return False, None
        return True, dp_rows

    def _scattered_dp(
        self,
        tail: Optional[LateLayerTail],
        *,
        tail_in_graph: bool,
        forward_batch: ForwardBatch,
        num_tokens: int,
    ) -> Optional[Tuple[int, List[int]]]:
        """Under attention DP with a MoE that routes each rank's own rows, the
        rank-uniform step tokens and every rank's tail rows; None elsewhere.
        Every condition here is rank-uniform, since the gather is collective. Such
        a MoE lets each rank pick a prefill graph or eager on its own, so the
        tail graphs serve it only with the prefill graph disabled."""
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
            num_tokens, 0 if tail is None else tail.positions.shape[0]
        )

    @staticmethod
    def _gathered_dp_rows(
        late_dp_counts: Optional[List[int]], forward_batch: ForwardBatch
    ) -> Optional[List[int]]:
        """Every attention-DP rank's tail rows for a TP MoE, which gathers them,
        or None when the DP layout cannot be made rank-uniform."""
        if not get_moe_a2a_backend().is_none() or not forward_batch.is_extend_in_batch:
            return None
        # Exchanged before the forward (ForwardBatch.prepare_mlp_sync_batch).
        if forward_batch.global_late_layer_num_tokens_cpu is not None:
            return list(forward_batch.global_late_layer_num_tokens_cpu)
        if late_dp_counts is not None:
            return late_dp_counts
        # No re-layout was needed: every rank's tail is its whole extend.
        early = forward_batch.global_num_tokens_padded_cpu
        if early is None:
            early = forward_batch.global_num_tokens_cpu
        if early is None or len(early) != get_parallel().attn_dp_size:
            return None
        return list(early)

    def in_prefill_graph(self, tail: LateLayerTail, *, num_tokens: int) -> bool:
        """Whether a prefill-graph capture of ``num_tokens`` hands the tail layers
        to the tail graphs' break instead of capturing them over the block."""
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        return (
            num_tokens >= envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MIN_PREFILL_TOKENS.get()
            and (get_parallel().attn_dp_size == 1 or get_moe_a2a_backend().is_none())
            and tail.cp_metadata is None
            and not is_unified_kv_triton()
            # Encoder replay folds window rows into the graph's extend; the
            # break's exact-row tail has not been checked against them.
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
        """Returns the tail rows' materialized residual and pre and the DSpark
        auxiliary states, as views of graph buffers valid until the next replay
        (``out_rows`` rows; default the tail's own). Under DP every rank replays
        the bucket of the largest in ``dp_rows``: a TP MoE gathers every rank's
        rows, and their padded tail IDs when it routes on them; MegaMoE reads
        this rank's own."""
        if forward_batch.forward_mode.is_idle():
            # An idle DP rank runs the extend's graph, with no rows of its own.
            forward_batch = copy.copy(forward_batch)
            forward_batch.forward_mode = ForwardMode.EXTEND
        gather = dp_rows is not None and get_moe_a2a_backend().is_none()
        # Ranks in the DP layout the MoE sees: all of them when it gathers.
        layout = None if dp_rows is None else (len(dp_rows) if gather else 1)
        gather_ids = gather and self._model.late_layer_vl_routing
        if input_ids_global is None:
            input_ids_global = input_ids
        leaves, has_pre, rebuild = _flatten_state(state)
        num_state = len(leaves)
        live = [*leaves, positions, input_ids]
        if not gather:
            live.append(input_ids_global)
        ids_global = (
            self._ids_global_buffer(input_ids.dtype, len(dp_rows))
            if gather_ids
            else None
        )

        def body(inputs: List[torch.Tensor], batch: ForwardBatch):
            rows = inputs[0].shape[0]
            if not gather:
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
            if layout is None:
                yield
                return
            rows = inputs[0].shape[0]
            graph_count = batch.num_token_non_padded
            saved = enter_late_layer_dp(batch, [rows] * layout)
            if layout == 1 and graph_count is not None:
                # A local MoE masks this rank's pad rows by the graph's count,
                # not by the bucket bound the layout clamps it to.
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
            key=(capture_dspark, has_pre, layout, gather_ids),
            live=live,
            forward_batch=forward_batch,
            body=body,
            context=self._break_context,
            batch_scope=dp_layout,
            bucket_from=None if dp_rows is None else max(dp_rows),
            out_rows=out_rows,
        )
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
        """An idle DP rank's part: the same bucket with no rows of its own, its
        outputs cut to its own padded rows."""
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

    def run_eager(
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
        """The graphs' body run eagerly on the live rows, for a prefill-graph
        step whose tail has no graph to replay; under DP attention in the DP
        layout of every rank's real rows."""
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
            if gather and self._model.late_layer_vl_routing:
                gathered = input_ids.new_empty((sum(dp_rows), 1))
                dp_gather_replicate(gathered, input_ids[:, None].clone(), tail_batch)
                input_ids_global = gathered.squeeze(-1)
            with self._break_context(tail_batch):
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
        """A prefill-graph step's tail: one break of the outer graph runs the tail
        layers on the live rows and bridges them back into its ``block_rows``
        block. Returns the bridged residual, pre and DSpark auxiliary states."""
        leaves, _, rebuild = _flatten_state(state)
        residual, pre, *aux = bcg_tail_layers_in_prefill_graph(
            self,
            rebuild,
            capture_dspark,
            block_rows,
            source_kv_layer_id,
            get_parallel().attn_dp_size > 1,
            leaves,
            positions,
            input_ids,
            input_ids if input_ids_global is None else input_ids_global,
        )
        return residual, pre, aux

    def bridge(
        self, block_rows: int, outputs: List[Optional[torch.Tensor]]
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Copy the tail rows into the bridge buffers of a prefill-graph step and
        return their block-sized views; the rows past the tail are zeroed."""
        key = tuple(None if t is None else (t.dtype, t.shape[1:]) for t in outputs)
        sets = self._bridge.setdefault(key, [])
        if not sets or sets[0][0].shape[0] < block_rows:
            sets.insert(
                0,
                [
                    None if t is None else t.new_zeros((block_rows, *t.shape[1:]))
                    for t in outputs
                ],
            )
        views = []
        for buf, t in zip(sets[0], outputs):
            if t is None:
                views.append(None)
                continue
            view = buf[:block_rows]
            view[: t.shape[0]].copy_(t)
            view[t.shape[0] :].zero_()
            views.append(view)
        return tuple(views)

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
            # Inside a prefill-graph step's break: the outer replay keeps its
            # context and breakable flag, so only the batch the breaks read swaps.
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
                    # The draft head reads the attention input of its target layers.
                    aux.append(state.residual.mean(dim=1))
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


def _tail_layers_in_prefill_graph(
    tail_layers: DecoderReplayTail,
    rebuild: Callable[[List[torch.Tensor]], HcState],
    capture_dspark: bool,
    block_rows: int,
    source_kv_layer_id: Optional[int],
    dp: bool,
    leaves: List[torch.Tensor],
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    input_ids_global: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], ...]:
    # A prefill-graph step's tail: the break reads the live batch, runs the tail
    # graphs on its exact rows and bridges them back into the captured block.
    backend = get_attn_backend()
    saved, forward_batch = backend.enter_eager_tail_in_prefill_graph(
        get_tc_piecewise_forward_context().forward_batch, source_kv_layer_id
    )
    # This body runs between the outer graph's segments, also while it captures:
    # the breaks inside the tail layers must run directly, not end its segment.
    capture_token = bcg_module._current_capture_var.set(None)
    try:
        tail = backend.forward_metadata.late_layer_tail
        # The graph's rows are padded to its shape; the tail indexes live rows.
        num_tokens = forward_batch.input_ids.shape[0]

        def rows(t: torch.Tensor) -> torch.Tensor:
            return tail.rows(t[:num_tokens])

        num_rows = tail.positions.shape[0]
        dp_rows = None
        if dp:
            # Exchanged before the forward (ForwardBatch.prepare_mlp_sync_batch).
            # The prefill graphs' own capture has no exchange; every rank captures
            # together, so it gathers here.
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
        # The prefill graphs' own capture runs the tail eagerly: their capture
        # stream and pool are busy with the outer graph.
        runner = tail_layers.runner
        if (
            not get_flags().dp.capturing_prefill_graph
            and runner.warm
            and runner.bucket_rows(max(dp_rows) if dp else num_rows) is not None
        ):
            residual, pre, aux = tail_layers.run(**kwargs)
        else:
            residual, pre, aux = tail_layers.run_eager(**kwargs)
        return tail_layers.bridge(block_rows, [residual, pre, *aux])
    finally:
        bcg_module._current_capture_var.reset(capture_token)
        backend.exit_late_layer_tail(saved, forward_batch)


bcg_tail_layers_in_prefill_graph = eager_on_graph(True)(_tail_layers_in_prefill_graph)
