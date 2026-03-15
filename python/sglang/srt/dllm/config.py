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
        max_running_requests: int,
        needs_full_prefill: bool = False,
        pad_full_generation: bool = False,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        # Bidirectional models (e.g. Dream) cannot use prefix KV cache;
        # they must recompute all tokens every forward pass.
        self.needs_full_prefill = needs_full_prefill
        # Full-attention algorithms pad ALL max_new_tokens masks at once
        # instead of adding block_size masks per round.
        self.pad_full_generation = pad_full_generation

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
        needs_full_prefill = False
        arch = model_config.hf_config.architectures[0]
        if arch == "LLaDA2MoeModelLM":
            block_size = 32
            mask_id = 156895
        elif arch == "Qwen3ForCausalLM":
            block_size = 32
            mask_id = 151670
        elif arch == "DreamModel":
            block_size = 32
            mask_id = 151666
            needs_full_prefill = True
        elif arch == "LLaDAModelLM":
            block_size = 32
            mask_id = 126336
            needs_full_prefill = True
        elif arch in {"SDARForCausalLM", "SDARMoeForCausalLM"}:
            block_size = 4
            mask_id = 151669
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
                algorithm_config = yaml.safe_load(f)

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)

        # Full-attention algorithms pad ALL max_new_tokens masks at once
        _FULL_GEN_PAD_ALGORITHMS = {"FullAttnMultiBlock"}
        pad_full_generation = (
            needs_full_prefill
            and server_args.dllm_algorithm in _FULL_GEN_PAD_ALGORITHMS
        )

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            needs_full_prefill=needs_full_prefill,
            pad_full_generation=pad_full_generation,
        )
