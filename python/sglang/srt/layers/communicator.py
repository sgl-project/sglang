from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable


class ScatterMode(Enum):
    SCATTERED = auto()
    TP_ATTN_FULL = auto()
    FULL = auto()


_IsLayerSparseCallable = Callable[[int], bool]


@dataclass
class LayerScatterModes:
    layer_input_mode: ScatterMode
    attn_mode: ScatterMode
    # Can be further split into e.g. ffn_input_mode and ffn_output_mode if needed
    ffn_mode: ScatterMode
    layer_output_mode: ScatterMode

    @classmethod
    def init_new(cls, layer_id: int, is_layer_sparse: _IsLayerSparseCallable):
        return cls(
            layer_input_mode=cls._compute_layer_input_mode(layer_id, is_layer_sparse),
            attn_mode=ScatterMode.TP_ATTN_FULL,
            ffn_mode=cls._compute_ffn_mode(layer_id, is_layer_sparse),
            layer_output_mode=cls._compute_layer_output_mode(layer_id, is_layer_sparse),
        )

    @classmethod
    def _compute_layer_input_mode(cls, layer_id: int, is_layer_sparse: _IsLayerSparseCallable):
        if layer_id == 0:
            return ScatterMode.TP_ATTN_FULL
        return TODO

    @classmethod
    def _compute_ffn_mode(cls, layer_id: int, is_layer_sparse: _IsLayerSparseCallable):
        if layer_id == num_layers - 1:
            return ScatterMode.TP_ATTN_FULL
        return TODO

    @classmethod
    def _compute_layer_output_mode(cls, layer_id: int, is_layer_sparse: _IsLayerSparseCallable):
        return TODO
