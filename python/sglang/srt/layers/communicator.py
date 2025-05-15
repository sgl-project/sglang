from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

from sglang.srt.managers.schedule_batch import global_server_args_dict


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
            return TODO
        else:
            return TODO

        return (
            _FFNInputMode.SCATTERED
            if (global_server_args_dict["enable_deepep_moe"] and is_sparse)
               or (DeepseekV2DecoderLayer._enable_moe_dense_fully_dp() and not is_sparse)
            else _FFNInputMode.FULL
        )

    @classmethod
    def _compute_layer_output_mode(cls, layer_id: int, context: _LayerModeComputationContext):
        if layer_id == context.num_layers - 1:
            return ScatterMode.TP_ATTN_FULL
        return cls._compute_ffn_mode(layer_id, context)


class LayerCommunicator:
    def __init__(self, layer_scatter_modes: LayerScatterModes):
        self.layer_scatter_modes = layer_scatter_modes

    def forward_pre_attn(self):
        TODO

    def forward_pre_mlp(self):
        TODO

    def forward_layer_end(self):
        TODO
