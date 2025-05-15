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

    @staticmethod
    def init_new(layer_id: int, is_layer_sparse: _IsLayerSparseCallable):
        return LayerScatterModes(
            layer_input_mode=TODO,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            ffn_mode=TODO,
            layer_output_mode=TODO,
        )
