from dataclasses import dataclass
from enum import Enum, auto


class ScatterMode(Enum):
    SCATTERED = auto()
    TP_ATTN_FULL = auto()
    FULL = auto()


@dataclass
class LayerScatterModes:
    layer_input_mode: ScatterMode
    attn_mode: ScatterMode
    ffn_mode: ScatterMode
    layer_output_mode: ScatterMode
