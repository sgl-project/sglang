from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW4A8MXFP4MoEMethod,
)
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_runner_backend
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

__all__ = ["NPUCompressedTensorsW4A8mxfp4MoE"]

logger = logging.getLogger(__name__)


class NPUCompressedTensorsW4A8mxfp4MoE(CompressedTensorsMoEScheme):
    """Compressed-tensors MXFP4 MoE scheme for Ascend NPU.

    Follows the same structure as the other NPU MoE schemes: the MXFP4
    payload / scale layout transforms live in ``NPUW4A8MXFP4MoEMethod``
    (shared with the ModelSlim path), and the runner drives w13 -> activation
    -> w2 through the Ascend ``MoeRunner``.
    """

    def __init__(self):
        self.group_size = 32
        self.w13_kernel = NPUW4A8MXFP4MoEMethod()
        self.w2_kernel = NPUW4A8MXFP4MoEMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        layer.params_dtype = params_dtype

        # Weights are stored as packed FP4 (two FP4 items per byte), so the
        # K dimension is halved. The compressed-tensors loader writes the
        # payload under the `_packed` suffix; process_weights_after_loading
        # renames it to the kernel's `w{13,2}_weight`.
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight scales: one e8m0 block scale per 32-value group.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The compressed-tensors MXFP4 loader stores the packed FP4 payloads
        # under the `_packed` suffix; rename them to the kernel's expected
        # names before delegating the NZ layout / scale transform.
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        self.w13_kernel.process_weights_after_loading(layer, "w13")
        self.w2_kernel.process_weights_after_loading(layer, "w2")

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        layer.w13_kernel = self.w13_kernel
        layer.w2_kernel = self.w2_kernel
        moe_runner_config.layer = layer
        self.moe_runner_config = moe_runner_config
        backend = get_moe_runner_backend()
        if backend.is_auto():
            backend = MoeRunnerBackend.ASCEND
        self.runner = MoeRunner(backend, moe_runner_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        quant_info = AscendQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight_scale=layer.w2_weight_scale,
        )
        return self.runner.run(dispatch_output, quant_info)
