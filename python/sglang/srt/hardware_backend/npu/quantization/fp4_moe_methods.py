"""MXFP4 routed-expert MoE method for Ascend A5 (Ascend 950).

DeepSeek-V4's FP4 expert checkpoint stores block-32 MXFP4 weights with E8M0
scales. This module adapts those checkpoint weights to the shared Ascend MoE
runner and A5 grouped-matmul kernels.
"""

from typing import TYPE_CHECKING

import torch

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW4A8MXFP4MoEMethod,
    prepare_w4a8_mxfp_weight,
)
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutput

# MXFP4 group size, fixed at 32 by the msmodelslim export format.
MXFP4_BLOCK_SIZE = 32


def _wrap_mxfp4_scale_weight_loader(weight_loader):
    def load_scale(param, loaded_weight, *args, **kwargs):
        if param.dtype == torch.uint8 and loaded_weight.dtype == torch.float8_e8m0fnu:
            loaded_weight = loaded_weight.view(torch.uint8)
        return weight_loader(param, loaded_weight, *args, **kwargs)

    return load_scale


class NPUW4A8MXFP4FusedMoEMethod(FusedMoEMethodBase):
    """DeepSeek-V4 routed experts on Ascend A5: W4A8 MXFP weights.

    The checkpoint-specific loading remains here while execution is delegated
    to the shared Ascend MoE runner.
    """

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        # ``None`` selects the full MX dynamic-quant defaults used by the
        # original DeepSeek-V4 path; the shared ModelSlim path keeps its
        # historical explicit ``dst_type`` behavior.
        self.w13_kernel = NPUW4A8MXFP4MoEMethod(dynamic_quant_kwargs=None)
        self.w2_kernel = NPUW4A8MXFP4MoEMethod(dynamic_quant_kwargs=None)
        self.runner = None

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

        # Two FP4 values per stored byte, hence the // 2 on the K dimension.
        w13_weight = torch.nn.Parameter(
            torch.empty(
                (num_experts, 2 * intermediate_size_per_partition, hidden_size // 2),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                (num_experts, hidden_size, intermediate_size_per_partition // 2),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        if weight_loader := scale_attrs.get("weight_loader"):
            scale_attrs["weight_loader"] = _wrap_mxfp4_scale_weight_loader(
                weight_loader
            )
        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                (
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // MXFP4_BLOCK_SIZE,
                ),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                (
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // MXFP4_BLOCK_SIZE,
                ),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        # Scales ship as raw E8M0 exponent bytes; no ue8m0 requantization here.
        w13_weight_scale.format_ue8m0 = False
        w2_weight_scale.format_ue8m0 = False
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config):
        from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
        from sglang.srt.layers.moe.utils import (
            MoeRunnerBackend,
            get_moe_runner_backend,
        )

        backend = get_moe_runner_backend()
        if backend.is_auto():
            backend = MoeRunnerBackend.ASCEND
        if not backend.is_ascend():
            raise ValueError(
                "NPU W4A8 MXFP4 requires the Ascend MoE runner, " f"got {backend.value}"
            )

        layer.w13_kernel = self.w13_kernel
        layer.w2_kernel = self.w2_kernel
        moe_runner_config.layer = layer
        self.runner = MoeRunner(backend, moe_runner_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sglang.srt.hardware_backend.npu.utils import NPUACLFormat

        if layer.w13_weight_scale_inv.data.max() == 0:
            raise RuntimeError(
                f"FP4 expert weight scales are all zero (never loaded) for "
                f"prefix={self.prefix!r}; the checkpoint scale names likely did "
                "not match w13_weight_scale_inv."
            )
        if layer.w2_weight_scale_inv.data.max() == 0:
            raise RuntimeError(
                f"FP4 expert weight scales are all zero (never loaded) for "
                f"prefix={self.prefix!r}; the checkpoint scale names likely did "
                "not match w2_weight_scale_inv."
            )

        nz_format = NPUACLFormat.ACL_FORMAT_FRACTAL_NZ
        layer.w13_weight.data, w13_scale = prepare_w4a8_mxfp_weight(
            layer.w13_weight.data.view(torch.uint8),
            layer.w13_weight_scale_inv.data,
            npu_format=nz_format,
        )
        layer.w2_weight.data, w2_scale = prepare_w4a8_mxfp_weight(
            layer.w2_weight.data.view(torch.uint8),
            layer.w2_weight_scale_inv.data,
            npu_format=nz_format,
        )
        layer.w13_weight_scale_inv = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale_inv = torch.nn.Parameter(w2_scale, requires_grad=False)

        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})

    def apply(self, layer: torch.nn.Module, dispatch_output: "DispatchOutput"):
        from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

        if self.runner is None:
            raise RuntimeError("The NPU FP4 MoE runner has not been initialized")

        quant_info = AscendQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            w13_weight_scale=layer.w13_weight_scale_inv,
            w2_weight_scale=layer.w2_weight_scale_inv,
        )
        return self.runner.run(dispatch_output, quant_info)
