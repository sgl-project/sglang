# SPDX-License-Identifier: Apache-2.0

"""Per-layer config resolution and AWQ/GPTQ dispatch logic for AutoRound.

This module isolates the routing concerns from the configuration definition in
``auto_round.py``. ``AutoRoundConfig`` delegates ``get_quant_method`` here.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.auto_round.auto_round import AutoRoundConfig

logger = logging.getLogger(__name__)

ScalarType, scalar_types = get_scalar_types()

_is_npu = is_npu()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()

# AutoRound GPTQ-format checkpoints are always packed with desc_act=False, so
# activation reordering is never applied. This assumption is shared by every
# GPTQ backend below (Ascend, CPU, Marlin, and the generic path).
_AUTOROUND_GPTQ_DESC_ACT = False


def get_layer_config(config: "AutoRoundConfig", layer, layer_name: str):
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

    def get_config(name: str, quantized: bool = True):
        if not config.extra_config:
            return (
                config.weight_bits if quantized else 16,
                config.group_size if quantized else -1,
                config.sym if quantized else True,
            )

        # Exact match first
        if name in config.extra_config:
            cfg = config.extra_config[name]
            return (
                cfg.get("bits", config.weight_bits if quantized else 16),
                cfg.get("group_size", config.group_size if quantized else -1),
                cfg.get("sym", config.sym if quantized else True),
            )

        REGEX_SPECIAL_CHARS = set(r"*+?^$()[]{}|\\")
        for pattern, cfg in config.extra_config.items():
            if not isinstance(pattern, str) or not any(
                c in REGEX_SPECIAL_CHARS for c in pattern
            ):
                continue

            try:
                if re.fullmatch(pattern, name):
                    return (
                        cfg.get("bits", config.weight_bits if quantized else 16),
                        cfg.get("group_size", config.group_size if quantized else -1),
                        cfg.get("sym", config.sym if quantized else True),
                    )
            except re.error:
                # Invalid regex, ignore.
                continue

        return (
            config.weight_bits if quantized else 16,
            config.group_size if quantized else -1,
            config.sym if quantized else True,
        )

    # 1. Exact match from config
    if config.extra_config and layer_name in config.extra_config:
        return get_config(layer_name)

    # 2. Determine whether layer should be quantized
    quantized = not isinstance(layer, ParallelLMHead)
    if config.block_name_to_quantize:
        quantized = any(
            layer_name.startswith(name) for name in config.block_name_to_quantize
        )

    # 3. Handle fused MoE
    if config.extra_config and "fusedmoe" in layer.__class__.__name__.lower():
        moe_configs = [
            get_config(name, quantized)
            for name in config.extra_config
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
    if config.extra_config:
        for fusion_key, sub_keys in config.packed_modules_mapping.items():
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


def apply_awq_quant_layer(
    config: "AutoRoundConfig", layer, prefix: str, backend: str = "auto"
):
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.quantization.marlin_utils import (
        check_marlin_supported,
        check_moe_marlin_supports_layer,
    )
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

    weight_bits, group_size, sym = get_layer_config(config, layer, prefix)
    if not config.check_quantized(weight_bits):
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
        from sglang.srt.layers.quantization.awq import (
            AWQCPUConfig,
            AWQLinearMethod,
            AWQMoEMethod,
        )

        quant_args = AWQCPUConfig(
            weight_bits=weight_bits,
            group_size=group_size,
            zero_point=not sym,
        )
        if isinstance(layer, FusedMoE):
            assert (
                _is_cpu_amx_available
            ), "AutoRound AWQ MoE on CPU requires an x86 CPU with AMX support."
            layer.scheme = quant_args.get_moe_scheme(layer)
            return AWQMoEMethod(quant_args)
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            layer.scheme = quant_args.get_linear_scheme(layer)
            return AWQLinearMethod(quant_args)
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
            AWQLinearMethod,
            AWQMarlinConfig,
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
            layer.scheme = quant_args_marlin.get_moe_scheme(layer)
            return AWQMoEMethod(quant_args_marlin)
        from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config

        moe_wna16_config = {
            "quant_method": "awq",
            "bits": weight_bits,
            "group_size": group_size,
            "zero_point": not sym,
            "lm_head": False,
        }
        return MoeWNA16Config.from_config(moe_wna16_config).get_quant_method(
            layer, prefix
        )

    if isinstance(layer, (LinearBase, ParallelLMHead)):
        if use_marlin:
            layer.scheme = quant_args_marlin.get_linear_scheme(layer)
            return AWQLinearMethod(quant_args_marlin)
        else:
            layer.scheme = quant_args.get_linear_scheme(layer)
            return AWQLinearMethod(quant_args)
    return None


def apply_gptq_quant_layer(
    config: "AutoRoundConfig", layer, prefix: str, backend: str = "auto"
):
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.quantization.gptq import (
        GPTQAscendConfig,
        GPTQLinearMethod,
        GPTQMoEMethod,
    )
    from sglang.srt.layers.quantization.marlin_utils import (
        check_marlin_supported,
        check_moe_marlin_supports_layer,
    )
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

    weight_bits, group_size, sym = get_layer_config(config, layer, prefix)
    if not config.check_quantized(weight_bits):
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
    if _is_npu:
        quant_args = GPTQAscendConfig(
            weight_bits=weight_bits,
            group_size=group_size,
            lm_head_quantized=False,
            desc_act=_AUTOROUND_GPTQ_DESC_ACT,
            dynamic={},
        )
        quant_args.sym = sym

        if isinstance(layer, FusedMoE):
            layer.scheme = quant_args.get_moe_scheme(layer)
            return GPTQMoEMethod(quant_args)

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            layer.scheme = quant_args.get_linear_scheme(layer)
            return GPTQLinearMethod(quant_args)

        return None

    if _is_cpu:
        from sglang.srt.layers.quantization.gptq import CPUGPTQConfig

        quant_args = CPUGPTQConfig(
            weight_bits=weight_bits,
            group_size=group_size,
            lm_head_quantized=False,
            desc_act=_AUTOROUND_GPTQ_DESC_ACT,
            dynamic={},
        )
        quant_args.sym = sym

        if isinstance(layer, FusedMoE):
            assert (
                _is_cpu_amx_available
            ), "AutoRound GPTQ MoE on CPU requires an x86 CPU with AMX support."
            layer.scheme = quant_args.get_moe_scheme(layer)
            return GPTQMoEMethod(quant_args)

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            layer.scheme = quant_args.get_linear_scheme(layer)
            return GPTQLinearMethod(quant_args)

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
            desc_act=_AUTOROUND_GPTQ_DESC_ACT,
            dynamic={},
            full_config={},
        )
    else:
        from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQLinearMethod

        quant_args = GPTQConfig(
            weight_bits=weight_bits,
            group_size=group_size,
            lm_head_quantized=False,
            desc_act=_AUTOROUND_GPTQ_DESC_ACT,
            dynamic={},
        )

    if isinstance(layer, FusedMoE):
        if use_marlin:
            return GPTQMarlinMoEMethod(quant_args_marlin)
        from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config

        moe_wna16_config = {
            "quant_method": "gptq",
            "bits": weight_bits,
            "group_size": group_size,
            "sym": sym,
            "lm_head": False,
        }
        return MoeWNA16Config.from_config(moe_wna16_config).get_quant_method(
            layer, prefix
        )

    if isinstance(layer, (LinearBase, ParallelLMHead)):
        if use_marlin:
            return GPTQMarlinLinearMethod(quant_args_marlin)
        else:
            return GPTQLinearMethod(quant_args)

    return None


def get_quant_method(config: "AutoRoundConfig", layer: torch.nn.Module, prefix: str):
    if "gptq" in config.packing_format or "gptq" in config.backend:
        return apply_gptq_quant_layer(config, layer, prefix, config.backend)
    if "awq" in config.packing_format or "awq" in config.backend:
        return apply_awq_quant_layer(config, layer, prefix, config.backend)
