from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
    from sglang.srt.layers.moe.moe_runner.base import MoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher.base import CombineInput
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput


def _get_inductor_groupedmm_compile_options():
    compile_options = {
        "max_autotune_gemm_backends": "TRITON",
        "max_autotune_gemm": True,
        "trace.enabled": False,
    }
    if hasattr(torch._inductor.config, "combo_kernels"):
        compile_options["combo_kernels"] = True
    if hasattr(torch._inductor.config, "triton"):
        triton_cfg = torch._inductor.config.triton
        if hasattr(triton_cfg, "enable_pdl"):
            compile_options["triton.enable_pdl"] = True
    return compile_options


INDUCTOR_GROUPEDMM_COMPILE_OPTIONS = _get_inductor_groupedmm_compile_options()


class InductorGroupedMMRunner:
    def __init__(self, default_runner: MoeRunner, config: MoeRunnerConfig):
        self.default_runner = default_runner
        self.config = config
        self.native_activation_fn = self._create_native_activation_fn(config)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.INDUCTOR_GROUPEDMM

    @property
    def default_runner_backend(self) -> MoeRunnerBackend:
        return self.default_runner.runner_backend

    @staticmethod
    def create_native_activation_fn_args(x, alpha, limit) -> tuple:
        if alpha is None or limit is None:
            return (x,)
        return x, alpha, limit

    @staticmethod
    def _create_native_activation_fn(config: MoeRunnerConfig) -> Callable:
        from sglang.srt.layers.activation import GeluAndMul, SiluAndMul
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
            _swiglu_gpt_oss_sigmoid_alpha,
        )

        if (
            config.activation == "silu"
            and config.gemm1_alpha is not None
            and config.gemm1_clamp_limit is not None
        ):
            return _swiglu_gpt_oss_sigmoid_alpha
        if (
            config.activation == "gelu"
            and config.gemm1_alpha is not None
            and config.gemm1_clamp_limit is not None
        ):
            raise ValueError(
                "Gelu activation with gemm1_alpha and gemm1_clamp_limit is not supported"
            )
        if (
            config.activation == "silu"
            and config.gemm1_alpha is None
            and config.gemm1_clamp_limit is None
        ):
            return SiluAndMul()
        if (
            config.activation == "gelu"
            and config.gemm1_alpha is None
            and config.gemm1_clamp_limit is None
        ):
            return GeluAndMul()
        raise ValueError(
            "Unsupported activation combination: "
            f"{config.activation=} "
            f"with gemm1_alpha={config.gemm1_alpha} "
            f"and gemm1_clamp_limit={config.gemm1_clamp_limit}"
        )

    def _should_use_inductor_groupedmm(self, is_decode: bool) -> bool:
        return self.native_activation_fn is not None and is_decode

    def _maybe_run_inductor_groupedmm(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
        is_decode: bool,
    ) -> CombineInput | None:
        if not self._should_use_inductor_groupedmm(is_decode):
            return None
        return self._forward_cuda_grouped_mm(layer, dispatch_output)

    @torch.compile(dynamic=True, options=INDUCTOR_GROUPEDMM_COMPILE_OPTIONS)
    def _forward_cuda_grouped_mm(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.fused_moe_native import (
            fused_moe_forward_native_grouped_mm,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        output = fused_moe_forward_native_grouped_mm(
            layer=layer,
            hidden_states=dispatch_output.hidden_states,
            topk_output=dispatch_output.topk_output,
            moe_runner_config=self.config,
            activation_fn=self.native_activation_fn,
            activation_fn_args=InductorGroupedMMRunner.create_native_activation_fn_args,
        )
        return StandardCombineInput(hidden_states=output)

    def run(
        self,
        dispatch_output: StandardDispatchOutput,
        quant_info: MoeQuantInfo,
        *,
        layer: torch.nn.Module,
        is_decode: bool = False,
    ) -> CombineInput:
        groupedmm_output = self._maybe_run_inductor_groupedmm(
            layer, dispatch_output, is_decode
        )
        if groupedmm_output is not None:
            return groupedmm_output
        return self.default_runner.run(dispatch_output, quant_info)

    def set_overlap_args(
        self, down_gemm_overlap_args: DownGemmOverlapArgs, meta_overlap_args: dict
    ) -> None:
        self.default_runner.set_overlap_args(down_gemm_overlap_args, meta_overlap_args)

    def clear_overlap_args(self) -> None:
        self.default_runner.clear_overlap_args()
