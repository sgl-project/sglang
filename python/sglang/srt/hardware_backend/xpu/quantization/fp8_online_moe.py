# SPDX-License-Identifier: Apache-2.0
"""Online FP8 MoE on Intel XPU: quantize the experts while they load.

``--quantization fp8`` on an unquantized (BF16) checkpoint allocates the experts
in the checkpoint dtype and quantizes them after loading, so loading peaks at
the full BF16 expert size. Here the experts start on the meta device and each
layer's w13 / w2 is materialized, filled by the original FusedMoE loader and
quantized as soon as this rank's copy is complete, the way the online MXFP4
(Quark) and NVFP4 MoE methods load. The quantization is the per-expert
``scaled_fp8_quant`` of ``Fp8MoEMethod.process_weights_after_loading_online_quant``,
so the weights and scales are the same.

``Fp8Config.get_quant_method`` selects ``XPUOnlineFp8MoEMethod`` on XPU when the
checkpoint is not FP8.
"""

from __future__ import annotations

import threading
from typing import Callable

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, scaled_fp8_quant
from sglang.srt.layers.quantization.dequantization import copy_missing_attrs
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.online_quantization import CopyNumelCounter

_NAMES = ("w13", "w2")


class XPUOnlineFp8MoEMethod(Fp8MoEMethod):
    """``Fp8MoEMethod`` for an unquantized checkpoint on XPU: the experts are
    quantized while they load instead of after loading."""

    def create_weights(self, layer: torch.nn.Module, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        _install_online_loader(self, layer)

    def process_weights_after_loading_online_quant(
        self, layer: torch.nn.Module
    ) -> None:
        if not getattr(layer, "xpu_online_fp8_moe", False):
            # The loader was not installed (padded experts).
            super().process_weights_after_loading_online_quant(layer)
            return
        # Quantize what was not quantized while loading (an incomplete weight,
        # or ``--load-format dummy``, which never calls the loader).
        for name in _NAMES:
            if getattr(layer, f"{name}_weight").dtype != fp8_dtype:
                _quantize(layer, name)


def _install_online_loader(method: Fp8MoEMethod, layer: torch.nn.Module) -> None:
    """Move w13 / w2 to the meta device and wrap their weight loaders.

    Block-quantized, FP4 and padded expert weights keep quantizing after
    loading. Sets ``layer.xpu_online_fp8_moe`` when the loaders are installed.
    """
    if (
        method.block_quant
        or method.is_fp4_expert
        or getattr(layer.w13_weight, "weight_padded", False)
    ):
        return
    load_device = layer.w13_weight.device
    lock = threading.Lock()
    for name in _NAMES:
        weight = getattr(layer, f"{name}_weight")
        meta = torch.nn.Parameter(
            torch.empty_like(weight, device="meta"), requires_grad=False
        )
        copy_missing_attrs(weight, meta)
        meta.weight_loader = _online_weight_loader(
            layer, name, weight.weight_loader, weight.dtype, load_device, lock
        )
        setattr(layer, f"{name}_weight", meta)
    layer.xpu_online_fp8_moe_load_device = load_device
    layer.xpu_online_fp8_moe = True


def _online_weight_loader(
    layer: torch.nn.Module,
    name: str,
    original_weight_loader: Callable,
    source_dtype: torch.dtype,
    load_device: torch.device,
    lock: threading.Lock,
) -> Callable:
    loaded_numel = 0

    def online_fp8_moe_weight_loader(
        param, loaded_weight, weight_name, shard_id, expert_id=None
    ):
        nonlocal loaded_numel
        with lock:
            weight = getattr(layer, f"{name}_weight")
            if expert_id is not None and not tuple(
                layer._weight_loader_expert_ids(weight, expert_id)
            ):
                # Nonlocal experts must not allocate or start a reload: with EP
                # their callbacks can arrive after this rank's weight is quantized.
                return
            if weight.is_meta or weight.dtype != source_dtype:
                # First shard, or a reload after quantization.
                source = torch.nn.Parameter(
                    torch.empty_like(weight, dtype=source_dtype, device=load_device),
                    requires_grad=False,
                )
                copy_missing_attrs(weight, source)
                setattr(layer, f"{name}_weight", source)
                weight, loaded_numel = source, 0

            # The original loader narrows for TP and maps experts, so count what
            # it copies rather than what arrives.
            counter = CopyNumelCounter()
            with counter:
                original_weight_loader(
                    weight, loaded_weight, weight_name, shard_id, expert_id
                )
            loaded_numel += counter.copied_numel
            if loaded_numel == weight.numel():
                _quantize(layer, name)

    return online_fp8_moe_weight_loader


def _quantize(layer: torch.nn.Module, name: str) -> None:
    """Quantize ``{name}_weight`` per expert, one per-tensor scale per expert
    (for w13 over the merged gate and up projections)."""
    weight = getattr(layer, f"{name}_weight")
    source = weight
    if weight.is_meta:
        # Never loaded: random values, as initialize_dummy_weights gives the
        # unquantized weights.
        source = torch.empty_like(
            weight, device=layer.xpu_online_fp8_moe_load_device
        ).uniform_(-1e-3, 1e-3)
    qweight = torch.empty_like(source.data, dtype=fp8_dtype)
    scale = torch.ones(
        layer.num_local_experts, dtype=torch.float32, device=qweight.device
    )
    for expert in range(layer.num_local_experts):
        qweight[expert, :, :], scale[expert] = scaled_fp8_quant(
            source.data[expert, :, :]
        )
    qparam = torch.nn.Parameter(qweight, requires_grad=False)
    # Keep the wrapped loader so a weight reload quantizes again.
    copy_missing_attrs(weight, qparam)
    setattr(layer, f"{name}_weight", qparam)
    setattr(
        layer,
        f"{name}_weight_scale",
        torch.nn.Parameter(scale, requires_grad=False),
    )
