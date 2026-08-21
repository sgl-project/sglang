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
        first_done_first_out_mode: bool = False,
        requires_separate_context_encoding: bool = False,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        self.first_done_first_out_mode = first_done_first_out_mode
        self.requires_separate_context_encoding = requires_separate_context_encoding

    def validate_request(self, req) -> str | None:
        from sglang.srt.dllm.algorithm import get_algorithm_cls

        return get_algorithm_cls(self.algorithm).validate_request(req)

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
            "DiffusionGemmaForBlockDiffusion": {
                "block_size": getattr(model_config.hf_config, "canvas_length", 256),
                "mask_id": -1,
                "algorithm": "Gemma4Renoise",
            },
        }

        architectures = getattr(model_config.hf_config, "architectures", None) or []
        if not architectures:
            raise RuntimeError("The model config does not declare an architecture")
        arch = architectures[0]
        if arch in DLLM_PARAMS:
            params = DLLM_PARAMS[arch]
            block_size = params["block_size"]
            mask_id = params["mask_id"]
        else:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")

        from sglang.srt.dllm.algorithm import get_algorithm_cls

        algorithm_cls = get_algorithm_cls(server_args.dllm_algorithm)
        required_algorithm = params.get("algorithm")
        if (
            required_algorithm is not None
            and required_algorithm != server_args.dllm_algorithm
        ):
            raise ValueError(
                f"{arch} requires the {required_algorithm} diffusion algorithm"
            )
        if (
            algorithm_cls.supported_architectures
            and arch not in algorithm_cls.supported_architectures
        ):
            raise ValueError(
                f"{server_args.dllm_algorithm} does not support model architecture {arch}"
            )

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

            if not isinstance(algorithm_config, dict):
                raise ValueError("The dLLM algorithm config must be a YAML mapping")

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            first_done_first_out_mode=server_args.dllm_fdfo,
            requires_separate_context_encoding=(
                algorithm_cls.requires_separate_context_encoding
            ),
        )
