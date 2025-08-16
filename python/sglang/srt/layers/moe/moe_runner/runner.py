from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig, PermuteMethodPool
from sglang.srt.layers.moe.token_dispatcher.base import CombineInput, CombineInputFormat, DispatchOutput
from sglang.srt.layers.moe.moe_runner.triton import TritonRunnerCore


if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import MoeQuantInfo
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

class MoeRunner:

    def __init__(self, runner_backend: MoeRunnerBackend, config: MoeRunnerConfig):
        self.runner_backend = runner_backend
        self.config = config

        if runner_backend.is_triton():
            self.runner_core = TritonRunnerCore(config)
        else:
            raise NotImplementedError(f"Unsupported runner backend: {runner_backend}")

    def run(self, dispatch_output: DispatchOutput, quant_info: MoeQuantInfo) -> CombineInput:

        # TODO: we may cache the pre_permute_func and post_permute_func

        dispatch_output_format = dispatch_output.format
        runner_input_format = self.runner_core.input_format
        pre_permute_func = PermuteMethodPool.get_pre_permute(dispatch_output_format, runner_input_format)

        runner_input = pre_permute_func(dispatch_output, self.config)
        runner_output = self.runner_core.run(runner_input, quant_info)

        runner_output_format = self.runner_core.output_format
        combine_input_format = CombineInputFormat.STANDARD
        post_permute_func = PermuteMethodPool.get_post_permute(runner_output_format, combine_input_format)
        combine_input = post_permute_func(runner_output, self.config)

        return combine_input
