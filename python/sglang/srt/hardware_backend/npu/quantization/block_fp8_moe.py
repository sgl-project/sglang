"""Ascend block-FP8 MoE with native G-B or opt-in MXFP8 execution."""

from types import SimpleNamespace

import torch

from sglang.kernels.ops.gemm import npu_w8a8_block_fp8_grouped_matmul
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.moe.standard_dispatch import set_dispatch_dtype
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUMXFP8MoEMethod,
    _pair_pack_mxfp_act_scale,
)
from sglang.srt.hardware_backend.npu.quantization.w8a8_mxfp8 import (
    process_npu_arch35_mxfp8_linear_weights,
)
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.moe.moe_runner import MoeRunner
from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    get_moe_runner_backend,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.runtime_context import get_exec


class _NPUBlockFp8MoEKernel:
    """One native G-B grouped GEMM; the runner supplies the activation."""

    def apply(
        self,
        quant_info,
        hidden_states,
        expert_tokens,
        pertoken_scale,
        output_dtype,
        weight_prefix,
        group_list_type,
    ):
        return npu_w8a8_block_fp8_grouped_matmul(
            hidden_states,
            getattr(quant_info, f"{weight_prefix}_weight"),
            getattr(quant_info, f"{weight_prefix}_weight_scale"),
            expert_tokens,
            group_list_type,
            output_dtype,
        )


class NPUBlockFp8MoEMethod(Fp8MoEMethod):
    """Reuse checkpoint loading and select the Ascend execution format."""

    DISPATCHER_QUANT_CONFIG = {
        "dispatcher_output_dtype": "mxfp8",  # Ascend Ops prefill / transport fallback
        "normal_dispatcher_output_dtype": "bf16",
        "low_latency_dispatcher_output_dtype": "mxfp8",
    }

    def __init__(self, quant_config: Fp8Config):
        backend = get_moe_runner_backend()
        if not (backend.is_auto() or backend.is_ascend()):
            raise ValueError(
                "Block FP8 MoE on Ascend requires --moe-runner-backend auto "
                f"or ascend, got {backend.value!r}"
            )
        dispatcher_dtype = get_exec().moe.deepep_dispatcher_output_dtype
        if dispatcher_dtype not in ("auto", "bf16"):
            raise ValueError(
                "--deepep-dispatcher-output-dtype must be auto or bf16, "
                f"got {dispatcher_dtype!r}"
            )
        super().__init__(quant_config)
        self.requant_block_fp8 = envs.SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8.get()
        self._weights_prepared = False
        self._force_bf16_dispatch = (
            dispatcher_dtype == "bf16" or envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        )

    def create_weights(self, layer, *args, with_bias=False, **kwargs):
        if with_bias:
            raise ValueError("Ascend block FP8 MoE does not support expert bias")
        super().create_weights(layer, *args, with_bias=False, **kwargs)

    def create_moe_runner(self, layer, moe_runner_config):
        if moe_runner_config.activation != "silu" or not moe_runner_config.is_gated:
            raise NotImplementedError(
                "Ascend block FP8 MoE requires gated SiLU (SwiGLU)"
            )
        if (
            moe_runner_config.gemm1_alpha is not None
            or moe_runner_config.gemm1_beta is not None
            or moe_runner_config.gemm1_clamp_limit is not None
        ):
            raise NotImplementedError(
                "Ascend block FP8 MoE does not support gemm1_alpha, "
                "gemm1_beta or gemm1_clamp_limit; swiglu_limit is supported"
            )
        if self.requant_block_fp8:
            layer.w13_kernel = NPUMXFP8MoEMethod("w13")
            layer.w2_kernel = NPUMXFP8MoEMethod("w2")
        else:
            layer.w13_kernel = _NPUBlockFp8MoEKernel()
            layer.w2_kernel = _NPUBlockFp8MoEKernel()
        self.moe_runner_config = moe_runner_config
        moe_runner_config.layer = layer
        self.runner = MoeRunner(MoeRunnerBackend.ASCEND, moe_runner_config)
        self._owns_moe_runner = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        set_dispatch_dtype(layer, "mxfp8" if self.requant_block_fp8 else "bf16")
        if not self.requant_block_fp8:
            # Native G-B retains checkpoint shapes and FP32 block scales.
            layer.dispatcher.set_quant_config(
                {
                    "dispatcher_output_dtype": "bf16",
                    "normal_dispatcher_output_dtype": "bf16",
                    "low_latency_dispatcher_output_dtype": "bf16",
                }
            )
            return
        if self._weights_prepared:
            return
        for prefix in ("w13", "w2"):
            weight_param = getattr(layer, f"{prefix}_weight")
            scale_param = getattr(layer, f"{prefix}_weight_scale_inv")
            weight = weight_param.data
            if not weight.is_npu:
                weight = weight.to(f"npu:{torch.npu.current_device()}")
            scale = scale_param.data.to(weight.device)
            experts, rows, cols = weight.shape
            quantized = torch.empty_like(weight)
            mx_scale = torch.empty(
                experts, rows, cols // 64, 2, device=weight.device, dtype=torch.uint8
            )
            for expert in range(experts):
                expert_layer = SimpleNamespace(
                    weight=weight[expert], weight_scale_inv=scale[expert]
                )
                process_npu_arch35_mxfp8_linear_weights(
                    expert_layer, self.weight_block_size, self.quant_config.scale_fmt
                )
                quantized[expert].view(torch.uint8).copy_(
                    expert_layer.weight.T.view(torch.uint8)
                )
                mx_scale[expert].copy_(
                    expert_layer.weight_scale_inv.transpose(0, 1).view(torch.uint8)
                )
            weight_param.data = npu_format_cast(quantized).transpose(1, 2)
            scale_param.data = mx_scale.transpose(1, 2)
        self._weights_prepared = True

        layer.dispatcher.set_quant_config(dict(self.DISPATCHER_QUANT_CONFIG))

    def _prepare_dispatch_output(self, dispatch_output):
        if not self.requant_block_fp8 or not dispatch_output.format.is_deepep_ll():
            return dispatch_output
        scale = dispatch_output.hidden_states_scale
        if scale is None:
            if (
                self._force_bf16_dispatch
                and dispatch_output.hidden_states.dtype == torch.bfloat16
            ):
                return dispatch_output
            raise RuntimeError(
                "DeepEP low-latency MXFP8 dispatch returned no activation scales. "
                "Use a DeepEP-Ascend build supporting use_mxfp8, or explicitly "
                "set --deepep-dispatcher-output-dtype bf16."
            )
        x = dispatch_output.hidden_states
        if x.dtype != torch.float8_e4m3fn or scale.dtype not in (
            torch.uint8,
            torch.float8_e8m0fnu,
        ):
            raise ValueError(
                "DeepEP low-latency MXFP8 requires E4M3 activations and E8M0 "
                "scales (or raw uint8 bytes); ordinary FP8 with FP32 scales "
                "cannot be passed to the MXFP8 GEMM."
            )
        # DeepEP may return flat or rowwise scales, while NPU MX quantization
        # produces [M,K/64,2]. Normalize logical order, including strided/padded
        # rows, as bytes so no numerical conversion or requantization occurs.
        rows, cols = x.shape
        scale = scale.view(torch.uint8).reshape(rows, cols // 32).contiguous()
        scale = _pair_pack_mxfp_act_scale(scale)
        return dispatch_output._replace(hidden_states_scale=scale)

    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

        return self.runner.run(
            self._prepare_dispatch_output(dispatch_output),
            AscendQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_weight_scale=layer.w13_weight_scale_inv,
                w2_weight_scale=layer.w2_weight_scale_inv,
            ),
        )
