from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from sglang.srt.layers.moe.moe_runner.base import (
    FusedOpPool,
    MoeRunnerConfig,
    PermuteMethodPool,
)
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmRunnerCore
from sglang.srt.layers.moe.moe_runner.triton import TritonRunnerCore
from sglang.srt.layers.moe.moe_runner.triton_kernels import TritonKernelsRunnerCore
from sglang.srt.layers.moe.utils import get_moe_a2a_backend

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
    from sglang.srt.layers.moe.moe_runner.base import MoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher.base import CombineInput, DispatchOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

logger = logging.getLogger(__name__)


class MoeRunner:

    def __init__(self, runner_backend: MoeRunnerBackend, config: MoeRunnerConfig):
        self.runner_backend = runner_backend
        self.config = config

        self.fused_func = None

        if runner_backend.is_triton():
            self.runner_core = TritonRunnerCore(config)
        elif runner_backend.is_triton_kernels():
            self.runner_core = TritonKernelsRunnerCore(config)
        elif runner_backend.is_deep_gemm():
            self.runner_core = DeepGemmRunnerCore(config)
        elif runner_backend.is_marlin():
            self.runner_core = None  # Marlin only supports fused path
        else:
            raise NotImplementedError(f"Unsupported runner backend: {runner_backend}")

        a2a_backend_name = get_moe_a2a_backend().value
        runner_backend_name = runner_backend.value

        # TODO(cwan): add a server argument to disable fused func
        self.fused_func = FusedOpPool.get_fused_func(
            a2a_backend_name, runner_backend_name
        )

        self.down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

        SGLANG_CI_DISABLE_MOE_FUSED_FUNC = os.environ.get(
            "SGLANG_CI_DISABLE_MOE_FUSED_FUNC", "0"
        )
        if SGLANG_CI_DISABLE_MOE_FUSED_FUNC == "1":
            logger.info(
                "SGLANG_CI_DISABLE_MOE_FUSED_FUNC is set to 1, disabling fused func"
            )
            self.fused_func = None

    def run(
        self, dispatch_output: DispatchOutput, quant_info: MoeQuantInfo
    ) -> CombineInput:

        if self.fused_func is not None:
            return self.fused_func(dispatch_output, quant_info, self.config)

        dispatch_format = dispatch_output.format.value
        runner_format = self.runner_core.runner_backend.value
        self.pre_permute_func = PermuteMethodPool.get_pre_permute(
            dispatch_format, runner_format
        )

        running_state = {}
        if self.down_gemm_overlap_args is not None:
            running_state["down_gemm_overlap_args"] = self.down_gemm_overlap_args
        if self.meta_overlap_args is not None:
            running_state["meta_overlap_args"] = self.meta_overlap_args

        runner_input = self.pre_permute_func(
            dispatch_output, quant_info, self.config, running_state
        )
        runner_output = self.runner_core.run(runner_input, quant_info, running_state)

        runner_format = self.runner_core.runner_backend.value
        combine_format = dispatch_output.format.value
        self.post_permute_func = PermuteMethodPool.get_post_permute(
            runner_format, combine_format
        )
        combine_input = self.post_permute_func(
            runner_output, quant_info, self.config, running_state
        )

        return combine_input

    def set_overlap_args(
        self, down_gemm_overlap_args: DownGemmOverlapArgs, meta_overlap_args: dict
    ):
        assert self.fused_func is None, "Fused func is not supported for overlap args"
        self.down_gemm_overlap_args = down_gemm_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        assert self.fused_func is None, "Fused func is not supported for overlap args"
        self.down_gemm_overlap_args = None
        self.meta_overlap_args = None
