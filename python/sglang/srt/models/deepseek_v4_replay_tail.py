"""DeepSeek-V4.1's decoder-replay tail layers (``late_layer_start`` on), replayed
from `RowBucketGraphRunner` graphs on eager prefill steps. The SWA KV store, the
attention and the low-ratio sources stay eager breaks of those graphs.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
)
from sglang.srt.model_executor.runner.row_bucket_graph_runner import (
    RowBucketGraphRunner,
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
from sglang.srt.models.deepseek_v4_mhc import HcState
from sglang.srt.runtime_context import get_parallel
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
        self, tail: LateLayerTail, *, num_tokens: int, tail_in_graph: bool
    ) -> bool:
        """Whether this step runs its tail layers from a graph; also reports the
        step's token count to the warm gate."""
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        self.runner.ready(num_tokens)
        return not (
            tail_in_graph
            or get_parallel().attn_dp_size > 1
            or get_is_capture_mode()
            or is_unified_kv_triton()
            or get_global_expert_distribution_recorder().recording
            or not self.runner.warm
            or tail.cp_metadata is not None
            or self.runner.bucket_rows(tail.positions.shape[0]) is None
        )

    def run(
        self,
        *,
        state: HcState,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_global: torch.Tensor,
        forward_batch: ForwardBatch,
        capture_dspark: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """Returns the tail rows' materialized residual and pre and the DSpark
        auxiliary states, as views of graph buffers valid until the next replay."""
        leaves, has_pre, rebuild = _flatten_state(state)
        num_state = len(leaves)

        def body(inputs: List[torch.Tensor], batch: ForwardBatch):
            residual, pre, aux = self._layer_stack(
                rebuild(inputs[:num_state]),
                positions=inputs[num_state],
                input_ids=inputs[num_state + 1],
                input_ids_global=inputs[num_state + 2],
                forward_batch=batch,
                capture_dspark=capture_dspark,
            )
            return [residual, pre, *aux]

        residual, pre, *aux = self.runner.run(
            key=(capture_dspark, has_pre),
            live=[*leaves, positions, input_ids, input_ids_global],
            forward_batch=forward_batch,
            body=body,
            context=self._break_context,
        )
        return residual, pre, aux

    @contextmanager
    def _break_context(self, tail_batch: ForwardBatch):
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
