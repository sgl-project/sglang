import json
from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        mask_id: int,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):
        if server_args.dllm_algorithm is None:
            return None

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )

        if model_config.hf_config.architectures[0] == "LLaDA2MoeModelLM":
            block_size = 32
            mask_id = 156895
        else:
            raise RuntimeError(
                f"Unknown diffusion LLM: {model_config.hf_config.architectures[0]}"
            )

        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            try:
                algorithm_config = json.loads(server_args.dllm_algorithm_config)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Invalid algorithm config: {server_args.dllm_algorithm_config}. Load config failed: {e}"
                )

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
        )
