from sglang.srt.layers.moe.moe_runner.base import (
    MoeRunnerConfig,
    MoeRunnerCore,
    PermuteMethodPool,
    RunnerInput,
    RunnerInputFormat,
    RunnerOutput,
    RunnerOutputFormat,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.quantization.base_config import MoeQuantInfo
from sglang.srt.layers.moe.token_dispatcher import (
    CombineInputFormat,
    DispatchOutputFormat,
    StandardCombineInput,
    StandardDispatchOutput,
)

permute_pool = PermuteMethodPool()


class TritonRunnerInput(RunnerInput):
    def get_format(self) -> RunnerInputFormat:
        return RunnerInputFormat.TRITON


class TritonRunnerOutput(RunnerOutput):
    def get_format(self) -> RunnerOutputFormat:
        return RunnerOutputFormat.TRITON


class TritonRunnerCore(MoeRunnerCore):

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    def run(self, runner_input: TritonRunnerInput, quant_info: MoeQuantInfo) -> TritonRunnerOutput:
        pass

    @classmethod
    def get_runner_input_format(cls) -> RunnerInputFormat:
        return RunnerInputFormat.TRITON

    @classmethod
    def get_runner_output_format(cls) -> RunnerOutputFormat:
        return RunnerOutputFormat.TRITON


@register_pre_permute(DispatchOutputFormat.STANDARD, RunnerInputFormat.TRITON)
def pre_permute_standard_to_triton(
    dispatch_output: StandardDispatchOutput,
    runner_config: MoeRunnerConfig,
) -> TritonRunnerInput:
    hidden_states, topk_output = dispatch_output


@register_post_permute(RunnerOutputFormat.TRITON, CombineInputFormat.STANDARD)
def post_permute_triton_to_standard(
    runner_output: TritonRunnerOutput,
) -> StandardDispatchOutput:
    pass
