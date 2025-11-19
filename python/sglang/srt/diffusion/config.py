from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


class DiffusionConfig:
    def __init__(
        self,
        mask_id: int,
        block_size: int,
        algorithm: str,
    ):
        self.algorithm = algorithm
        self.block_size = block_size
        self.mask_id = mask_id

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):
        if server_args.diffusion_algorithm is None:
            return None

        config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )

        if config.hf_config.architectures[0] == "LLaDA2MoeModelLM":
            mask_id = 156895
        else:
            raise RuntimeError(
                f"Unknown diffusion model: {config.hf_config.architectures[0]}"
            )

        return DiffusionConfig(
            algorithm=server_args.diffusion_algorithm,
            block_size=server_args.diffusion_block_size,
            mask_id=mask_id,
        )
