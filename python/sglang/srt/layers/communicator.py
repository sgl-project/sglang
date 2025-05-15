from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Tuple

import torch
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.dp_attention import dp_gather_partial, dp_scatter
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class ScatterMode(Enum):
    SCATTERED = auto()
    TP_ATTN_FULL = auto()
    FULL = auto()


_IsLayerSparseCallable = Callable[[int], bool]


@dataclass
class _LayerModeComputationContext:
    num_layers: int
    is_layer_sparse: _IsLayerSparseCallable


@dataclass
class LayerScatterModes:
    layer_input_mode: ScatterMode
    attn_mode: ScatterMode
    # Can be further split into e.g. ffn_input_mode and ffn_output_mode if needed
    ffn_mode: ScatterMode
    layer_output_mode: ScatterMode

    @classmethod
    def init_new(cls, layer_id: int, num_layers: int, is_layer_sparse: _IsLayerSparseCallable):
        context = _LayerModeComputationContext(num_layers=num_layers, is_layer_sparse=is_layer_sparse)
        return cls(
            layer_input_mode=cls._compute_layer_input_mode(layer_id, context),
            attn_mode=ScatterMode.TP_ATTN_FULL,
            ffn_mode=cls._compute_ffn_mode(layer_id, context),
            layer_output_mode=cls._compute_layer_output_mode(layer_id, context),
        )

    @classmethod
    def _compute_layer_input_mode(cls, layer_id: int, context: _LayerModeComputationContext):
        if layer_id == 0:
            return ScatterMode.TP_ATTN_FULL
        return cls._compute_layer_output_mode(layer_id=layer_id - 1, context=context)

    @classmethod
    def _compute_ffn_mode(cls, layer_id: int, context: _LayerModeComputationContext):
        if context.is_layer_sparse(layer_id):
            return ScatterMode.SCATTERED if global_server_args_dict["enable_deepep_moe"] else ScatterMode.FULL
        else:
            return ScatterMode.SCATTERED if enable_moe_dense_fully_dp() else ScatterMode.FULL

    @classmethod
    def _compute_layer_output_mode(cls, layer_id: int, context: _LayerModeComputationContext):
        if layer_id == context.num_layers - 1:
            return ScatterMode.TP_ATTN_FULL
        return cls._compute_ffn_mode(layer_id, context)


def enable_moe_dense_fully_dp():
    return global_server_args_dict["moe_dense_tp_size"] == 1


class LayerCommunicator:
    def __init__(self, layer_scatter_modes: LayerScatterModes):
        self.layer_scatter_modes = layer_scatter_modes

    def forward_pre_attn(self):
        TODO

    def forward_pre_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if TODO_mode_forward_ffn_with_full_input:
            if get_tensor_model_parallel_world_size() > 1:
                # all gather and all reduce
                if self.local_dp_size != 1:
                    if self.attn_tp_rank == 0:
                        hidden_states += residual
                    hidden_states, local_hidden_states = (
                        forward_batch.gathered_buffer,
                        hidden_states,
                    )
                    dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
                    dp_scatter(residual, hidden_states, forward_batch)
                    hidden_states = self.post_attention_layernorm(hidden_states)
                else:
                    hidden_states = tensor_model_parallel_all_reduce(hidden_states)
                    hidden_states, residual = self.post_attention_layernorm(
                        hidden_states, residual
                    )
            else:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )

        return hidden_states, residual

    def forward_layer_end(self):
        TODO
