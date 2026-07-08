# SPDX-License-Identifier: Apache-2.0

import logging
from fractions import Fraction
from typing import Any, Optional, Union

import torch

from sglang.srt.layers.quantization.base_config import QuantizationConfig

logger = logging.getLogger(__name__)


class AutoRoundConfig(QuantizationConfig):
    """Config class for AutoRound.
    Reference: https://arxiv.org/pdf/2309.05516
    """

    SUPPORTED_BITS = {2, 3, 4, 8}
    SUPPORTED_DTYPES = {"int"}
    SUPPORTED_FORMATS = {"auto_round:auto_gptq", "auto_round:auto_awq"}
    SUPPORTED_BACKENDS = {"auto", "gptq", "gptq:marlin", "awq", "awq:marlin", "marlin"}

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        sym: bool = True,
        packing_format: str = "auto_round:auto_gptq",
        block_name_to_quantize: Optional[Union[str, list[str]]] = None,
        extra_config: Optional[dict[str, Any]] = None,
        data_type: str = "int",
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if weight_bits not in self.SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported weight_bits: {weight_bits}, "
                f"currently only support  {self.SUPPORTED_BITS}"
            )
        if data_type not in self.SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported data_type: {data_type},"
                f" currently only support  {self.SUPPORTED_DTYPES}"
            )
        if packing_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported packing_format: {packing_format}, "
                f"currently only support  {self.SUPPORTED_FORMATS}"
            )
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend},  "
                f"currently only support  {self.SUPPORTED_BACKENDS}"
            )

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.sym = sym
        self.packing_format = packing_format
        self.block_name_to_quantize = (
            block_name_to_quantize.split(",")
            if isinstance(block_name_to_quantize, str)
            else block_name_to_quantize
        )
        self.extra_config = extra_config
        self.data_type = data_type
        self.backend = backend
        self.pack_factor = Fraction(32, weight_bits)

    def __repr__(self) -> str:
        return (
            f"AutoRoundConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, sym={self.sym})"
        )

    @classmethod
    def get_name(cls):
        return "auto-round"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AutoRoundConfig":
        return cls(
            weight_bits=cls.get_from_keys(config, ["bits"]),
            group_size=cls.get_from_keys(config, ["group_size"]),
            sym=cls.get_from_keys(config, ["sym"]),
            packing_format=cls.get_from_keys_or(
                config,
                ["packing_format"],
                "auto_round:auto_gptq",
            ),
            block_name_to_quantize=cls.get_from_keys_or(
                config, ["block_name_to_quantize", "to_quant_block_names"], None
            ),
            extra_config=cls.get_from_keys_or(config, ["extra_config"], None),
            data_type=cls.get_from_keys_or(config, ["data_type"], "int"),
            backend=cls.get_from_keys_or(
                config, ["backend", "vllm_backend", "sglang_backend"], "auto"
            ),
        )

    def get_scaled_act_names(self) -> list[str]:
        """Returns the activation function names that should be post-scaled.

        For now, this is only used by AWQ.
        """
        raise NotImplementedError

    def check_quantized(self, weight_bits: int) -> bool:
        return weight_bits < 16

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from sglang.srt.layers.quantization.auto_round import routing

        return routing.get_quant_method(self, layer, prefix)
