from sglang.srt.diffusion.algorithm import get_algorithm
from sglang.srt.diffusion.config import DiffusionConfig
from sglang.srt.server_args import ServerArgs


class DiffusionAlgorithm:

    def __init__(
        self,
        config: DiffusionConfig,
    ):
        self.block_size = config.block_size
        self.mask_id = config.mask_id

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        config = DiffusionConfig.from_server_args(server_args)
        return get_algorithm(config)
