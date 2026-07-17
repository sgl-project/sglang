from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


def _validate_multi_block_prefill_backend(
    *, block_size: int, prefill_block_size: int, prefill_attention_backend: str
) -> None:
    if prefill_block_size > block_size and prefill_attention_backend != "flashinfer":
        raise ValueError(
            "dLLM multi-block prefill currently requires the FlashInfer "
            "prefill attention backend: "
            f"{prefill_block_size=}, {block_size=}, "
            f"{prefill_attention_backend=}"
        )


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        prefill_block_size: int,
        mask_id: int,
        max_running_requests: int,
        first_done_first_out_mode: bool = False,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.prefill_block_size = prefill_block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        self.first_done_first_out_mode = first_done_first_out_mode

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
        DLLM_PARAMS = {
            "LLaDA2MoeModelLM": {"block_size": 32, "mask_id": 156895},
            "SDARForCausalLM": {"block_size": 4, "mask_id": 151669},
            "SDARMoeForCausalLM": {"block_size": 4, "mask_id": 151669},
        }

        arch = model_config.hf_config.architectures[0]
        if arch in DLLM_PARAMS:
            params = DLLM_PARAMS[arch]
            block_size = params["block_size"]
            mask_id = params["mask_id"]
        else:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")

        max_running_requests = (
            1
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )

        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(server_args.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f) or {}

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)

        # Preserve the previous fixed-block behavior unless the user explicitly
        # opts into larger prefill chunks.
        prefill_block_size = algorithm_config.get("prefill_block_size", block_size)
        if server_args.dllm_prefill_block_size is not None:
            prefill_block_size = server_args.dllm_prefill_block_size
        if prefill_block_size < block_size or prefill_block_size % block_size != 0:
            raise ValueError(
                "dllm prefill_block_size must be a positive multiple of block_size "
                f"and no smaller than it: {prefill_block_size=}, {block_size=}"
            )
        prefill_attention_backend, _ = server_args.get_attention_backends()
        _validate_multi_block_prefill_backend(
            block_size=block_size,
            prefill_block_size=prefill_block_size,
            prefill_attention_backend=prefill_attention_backend,
        )

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            prefill_block_size=prefill_block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            first_done_first_out_mode=server_args.dllm_fdfo,
        )
