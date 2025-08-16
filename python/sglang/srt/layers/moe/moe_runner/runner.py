from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig


class MoeRunner:
    def __init__(self, config: MoeRunnerConfig):
        self.config = config
