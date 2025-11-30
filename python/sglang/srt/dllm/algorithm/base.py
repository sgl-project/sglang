from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.server_args import ServerArgs


class DllmAlgorithm:

    def __init__(
        self,
        config: DllmConfig,
    ):
        self.block_size = config.block_size
        self.mask_id = config.mask_id

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        config = DllmConfig.from_server_args(server_args)
        return get_algorithm(config)
