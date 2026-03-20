# SPDX-License-Identifier: Apache-2.0

import logging
import re
from fractions import Fraction
from typing import Any, Optional, Union

import torch

logger = logging.getLogger(__name__)

from sglang.srt.layers.quantization.utils import get_scalar_types

ScalarType, scalar_types = get_scalar_types()

from sglang.srt.layers.quantization.base_config import LinearMethodBase, QuantizationConfig
from sglang.srt.utils import is_cpu, is_npu

_is_cpu = is_cpu()
_is_npu = is_npu()


def _has_int4_cpu_kernels() -> bool:
    sgl_kernel_ops = getattr(torch.ops, "sgl_kernel", None)
    return (
        sgl_kernel_ops is not None
        and hasattr(sgl_kernel_ops, "int4_scaled_mm_cpu")
        and hasattr(sgl_kernel_ops, "convert_weight_packed_scale_zp")
    )


class AutoRoundAWQCPULinearMethod(LinearMethodBase):
    """CPU fallback for AutoRound AWQ using sgl-kernel int4 GEMM."""

    def __init__(self, quant_config):
        from sglang.srt.layers.quantization.awq import AWQLinearMethod

        self.quant_config = quant_config
        self._delegate = AWQLinearMethod(quant_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self._delegate.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _has_int4_cpu_kernels():
            raise RuntimeError(
                "AutoRound AWQ CPU requires int4 CPU kernels. "
                "Please build sgl-kernel CPU backend first."
            )

        qweight, qzeros, scales = torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
            layer.qweight.data,
            layer.qzeros.data,
            layer.scales.data,
        )
        layer.qweight = torch.nn.Parameter(qweight, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(qzeros, requires_grad=False)
        layer.scales = torch.nn.Parameter(scales, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        output = torch.ops.sgl_kernel.int4_scaled_mm_cpu(
            x_2d,
            layer.qweight,
            layer.qzeros,
            layer.scales,
            bias,
        )
        return output.reshape(x.shape[:-1] + (output.shape[-1],))


class AutoRoundGPTQCPULinearMethod(LinearMethodBase):
    """CPU fallback for AutoRound GPTQ by dequantizing weights on-the-fly."""

    def __init__(self, quant_config):
        from sglang.srt.layers.quantization.gptq import GPTQLinearMethod

        self.quant_config = quant_config
        self._delegate = GPTQLinearMethod(quant_config)
        self.use_v2_format = quant_config.checkpoint_format == "gptq_v2"

    @staticmethod
    def _unpack_int32(
        packed: torch.Tensor, num_bits: int, packed_dim: int
    ) -> torch.Tensor:
        pack_factor = 32 // num_bits
        mask = (1 << num_bits) - 1
        shifts = (
            torch.arange(pack_factor, device=packed.device, dtype=torch.int32)
            * num_bits
        )
        packed_i32 = packed.to(torch.int32)

        if packed_dim == 0:
            # [K // pack_factor, N] -> [K, N]
            return (
                (packed_i32.unsqueeze(1) >> shifts.view(1, -1, 1)) & mask
            ).reshape(-1, packed.shape[1])
        if packed_dim == 1:
            # [G, N // pack_factor] -> [G, N]
            return (
                (packed_i32.unsqueeze(-1) >> shifts.view(1, 1, -1)) & mask
            ).reshape(packed.shape[0], -1)
        raise ValueError(f"Unsupported packed_dim={packed_dim}")

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self._delegate.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Keep tensors in loaded order for CPU fallback dequantization.
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.g_idx = torch.nn.Parameter(layer.g_idx.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        num_bits = self.quant_config.weight_bits

        qweight = self._unpack_int32(layer.qweight, num_bits=num_bits, packed_dim=0).to(
            layer.scales.dtype
        )
        qzeros = self._unpack_int32(layer.qzeros, num_bits=num_bits, packed_dim=1).to(
            layer.scales.dtype
        )
        if not self.use_v2_format:
            qzeros = qzeros + 1

        use_default_gidx = layer.g_idx.numel() == 0 or (
            self.quant_config.group_size == -1 and torch.any(layer.g_idx < 0)
        )
        if use_default_gidx:
            if self.quant_config.group_size == -1:
                g_idx = torch.zeros(
                    (qweight.shape[0],), dtype=torch.long, device=layer.scales.device
                )
            else:
                g_idx = torch.arange(
                    qweight.shape[0], dtype=torch.long, device=layer.scales.device
                ) // self.quant_config.group_size
        else:
            g_idx = layer.g_idx.to(dtype=torch.long)

        if g_idx.numel() != qweight.shape[0]:
            raise ValueError(
                "AutoRound GPTQ CPU fallback expects g_idx rows to match qweight rows, "
                f"but got g_idx={g_idx.numel()} and qweight_rows={qweight.shape[0]}."
            )

        scale_zeros = qzeros * layer.scales
        dequant_weight = qweight * layer.scales[g_idx] - scale_zeros[g_idx]
        output = torch.matmul(x_2d, dequant_weight)
        if bias is not None:
            output.add_(bias)
        return output.reshape(x.shape[:-1] + (dequant_weight.shape[-1],))


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

    def get_layer_config(self, layer, layer_name: str):
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        def get_config(name: str, quantized: bool = True):
            if not self.extra_config:
                return (
                    self.weight_bits if quantized else 16,
                    self.group_size if quantized else -1,
                    self.sym if quantized else True,
                )

            # Exact match first
            if name in self.extra_config:
                cfg = self.extra_config[name]
                return (
                    cfg.get("bits", self.weight_bits if quantized else 16),
                    cfg.get("group_size", self.group_size if quantized else -1),
                    cfg.get("sym", self.sym if quantized else True),
                )

            REGEX_SPECIAL_CHARS = set(r"*+?^$()[]{}|\\")
            for pattern, cfg in self.extra_config.items():
                if not isinstance(pattern, str) or not any(
                    c in REGEX_SPECIAL_CHARS for c in pattern
                ):
                    continue

                try:
                    if re.fullmatch(pattern, name):
                        return (
                            cfg.get("bits", self.weight_bits if quantized else 16),
                            cfg.get("group_size", self.group_size if quantized else -1),
                            cfg.get("sym", self.sym if quantized else True),
                        )
                except re.error:
                    # Invalid regex, ignore.
                    continue

            return (
                self.weight_bits if quantized else 16,
                self.group_size if quantized else -1,
                self.sym if quantized else True,
            )

        # 1. Exact match from config
        if self.extra_config and layer_name in self.extra_config:
            return get_config(layer_name)

        # 2. Determine whether layer should be quantized
        quantized = not isinstance(layer, ParallelLMHead)
        if self.block_name_to_quantize:
            quantized = any(
                layer_name.startswith(name) for name in self.block_name_to_quantize
            )

        # 3. Handle fused MoE
        if self.extra_config and "fusedmoe" in layer.__class__.__name__.lower():
            moe_configs = [
                get_config(name, quantized)
                for name in self.extra_config
                if name.startswith(layer_name)
            ]
            if moe_configs:
                if len(set(moe_configs)) == 1:
                    return moe_configs[0]
                raise ValueError(
                    f"Fused MoE layer '{layer_name}' requires "
                    f"consistent quant config for all sub-layers"
                )

        # 4. Handle fused QKV or other patterns
        if self.extra_config:
            for fusion_key, sub_keys in self.packed_modules_mapping.items():
                if fusion_key in layer_name and layer_name.count(fusion_key) == 1:
                    sub_names = [
                        layer_name.replace(fusion_key, sub_key) for sub_key in sub_keys
                    ]
                    sub_configs = [get_config(name, quantized) for name in sub_names]
                    if len(set(sub_configs)) == 1:
                        return sub_configs[0]
                    raise ValueError(
                        f"Fused module '{layer_name}' requires "
                        f"consistent quant config for {sub_names}"
                    )

        # 5. Fallback or try a regular expression match
        return get_config(layer_name, quantized)

    def check_quantized(self, weight_bits: int) -> bool:
        return weight_bits < 16

    def apply_awq_quant_layer(self, layer, prefix: str, backend: str = "auto"):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.marlin_utils import (
            check_marlin_supported,
            check_moe_marlin_supports_layer,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None
        logger.debug(
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
            prefix,
            layer.__class__.__name__,
            weight_bits,
            group_size,
            sym,
        )
        if _is_cpu:
            from sglang.srt.layers.quantization.awq import AWQConfig

            if isinstance(layer, FusedMoE):
                raise NotImplementedError(
                    "AutoRound AWQ MoE on CPU is not supported yet."
                )
            quant_args = AWQConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                zero_point=not sym,
            )
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return AutoRoundAWQCPULinearMethod(quant_args)
            return None

        if backend == "auto" or "marlin" in backend:
            AWQ_TYPE_MAP = {
                4: scalar_types.uint4,
                8: scalar_types.uint8,
            }
            use_marlin = (weight_bits in AWQ_TYPE_MAP) and check_marlin_supported(
                AWQ_TYPE_MAP[weight_bits], group_size, not sym
            )
            if isinstance(layer, FusedMoE):
                use_marlin = use_marlin and check_moe_marlin_supports_layer(
                    layer, group_size
                )

        else:
            use_marlin = False
        if use_marlin:
            from sglang.srt.layers.quantization.awq import (
                AWQMarlinConfig,
                AWQMarlinLinearMethod,
                AWQMoEMethod,
            )

            quant_args_marlin = AWQMarlinConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                zero_point=not sym,
                lm_head_quantized=False,
                full_config={},
                modules_to_not_convert=[],
            )
        else:
            from sglang.srt.layers.quantization.awq import AWQConfig, AWQLinearMethod

            quant_args = AWQConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                zero_point=not sym,
            )

        if isinstance(layer, FusedMoE):
            if use_marlin:
                return AWQMoEMethod(quant_args_marlin)
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config

            config = {
                "quant_method": "awq",
                "bits": weight_bits,
                "group_size": group_size,
                "zero_point": not sym,
                "lm_head": False,
            }
            return MoeWNA16Config.from_config(config).get_quant_method(layer, prefix)

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if use_marlin:
                return AWQMarlinLinearMethod(quant_args_marlin)
            else:
                return AWQLinearMethod(quant_args)
        return None

    def apply_gptq_quant_layer(self, layer, prefix: str, backend: str = "auto"):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.gptq import (
            GPTQConfig,
            GPTQLinearAscendMethod,
            GPTQMoEAscendMethod,
        )
        from sglang.srt.layers.quantization.marlin_utils import (
            check_marlin_supported,
            check_moe_marlin_supports_layer,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None

        logger.debug(
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
            prefix,
            layer.__class__.__name__,
            weight_bits,
            group_size,
            sym,
        )
        if _is_cpu:
            quant_args = GPTQConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                lm_head_quantized=False,
                desc_act=False,
                dynamic={},
            )
            if isinstance(layer, FusedMoE):
                raise NotImplementedError(
                    "AutoRound GPTQ MoE on CPU is not supported yet."
                )
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return AutoRoundGPTQCPULinearMethod(quant_args)
            return None

        if _is_npu:
            quant_args = GPTQConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                lm_head_quantized=False,
                desc_act=False,
                dynamic={},
            )
            quant_args.sym = sym

            if isinstance(layer, FusedMoE):
                return GPTQMoEAscendMethod(quant_args)

            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return GPTQLinearAscendMethod(quant_args)

            return None

        if backend == "auto" or "marlin" in backend:
            GPTQ_TYPE_MAP = {
                (4, True): scalar_types.uint4b8,
                (8, True): scalar_types.uint8b128,
            }
            use_marlin = (weight_bits, sym) in GPTQ_TYPE_MAP and check_marlin_supported(
                GPTQ_TYPE_MAP[(weight_bits, sym)], group_size, has_zp=not sym
            )
            if isinstance(layer, FusedMoE):
                use_marlin = use_marlin and check_moe_marlin_supports_layer(
                    layer, group_size
                )
        else:
            use_marlin = False
        if use_marlin:
            from sglang.srt.layers.quantization.gptq import (
                GPTQMarlinConfig,
                GPTQMarlinLinearMethod,
                GPTQMarlinMoEMethod,
            )

            quant_args_marlin = GPTQMarlinConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                is_sym=sym,
                lm_head_quantized=False,
                desc_act=False,
                dynamic={},
                full_config={},
            )
        else:
            from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQLinearMethod

            quant_args = GPTQConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                lm_head_quantized=False,
                desc_act=False,
                dynamic={},
            )

        if isinstance(layer, FusedMoE):
            if use_marlin:
                return GPTQMarlinMoEMethod(quant_args_marlin)
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config

            config = {
                "quant_method": "gptq",
                "bits": weight_bits,
                "group_size": group_size,
                "sym": sym,
                "lm_head": False,
            }
            return MoeWNA16Config.from_config(config).get_quant_method(layer, prefix)
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if use_marlin:
                return GPTQMarlinLinearMethod(quant_args_marlin)
            else:
                return GPTQLinearMethod(quant_args)

        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if "gptq" in self.packing_format or "gptq" in self.backend:
            return self.apply_gptq_quant_layer(layer, prefix)
        if "awq" in self.packing_format or "awq" in self.backend:
            return self.apply_awq_quant_layer(layer, prefix)
        return None
