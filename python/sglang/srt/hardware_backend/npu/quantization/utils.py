from typing import TYPE_CHECKING, List, Optional

from sglang.srt.layers.quantization.base_config import LinearMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

class _NPULinearMethodBase(LinearMethodBase):

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        super().__init__()
        self.quant_config = quant_config
