from typing import Any

from sglang.srt.arg_groups.overrides import resolving_view
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
        delete_token_id: int | None = None,
        split_token_id: int | None = None,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        self.first_done_first_out_mode = first_done_first_out_mode
        self.delete_token_id = delete_token_id
        self.split_token_id = split_token_id

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):

        cfg = resolving_view(server_args)
        if cfg.dllm_algorithm is None:
            return None

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=cfg.model_path,
            model_revision=cfg.revision,
        )
        DLLM_PARAMS = {
            "LLaDA2MoeModelLM": {
                "block_size": 32,
                "mask_id": 156895,
            },
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

        delete_token_id = None
        split_token_id = None
        if cfg.dllm_algorithm == "JointThresholdInDel":
            delete_token_id = getattr(model_config.hf_config, "delete_token_id", None)
            split_token_id = getattr(model_config.hf_config, "split_token_id", None)

            if delete_token_id is None or split_token_id is None:
                override_example = (
                    '{"delete_token_id": 156930, "split_token_id": 156931}'
                )
                raise RuntimeError(
                    "JointThresholdInDel is not supported for checkpoint "
                    f"{cfg.model_path!r}: the checkpoint must declare both "
                    "delete_token_id and split_token_id. Use a checkpoint with "
                    "explicit Insert/Delete support. If you are certain this "
                    "checkpoint was trained for Insert/Delete decoding, declare the "
                    "correct token IDs in config.json or pass them with "
                    "`--json-model-override-args "
                    f"'{override_example}'`. "
                    "Use the token IDs defined by your checkpoint."
                )

            vocab_size = model_config.vocab_size
            for token_name, token_id in (
                ("delete_token_id", delete_token_id),
                ("split_token_id", split_token_id),
            ):
                if isinstance(token_id, bool) or not isinstance(token_id, int):
                    raise ValueError(
                        f"{token_name} must be an integer token ID, got {token_id!r}"
                    )
                if not 0 <= token_id < vocab_size:
                    raise ValueError(
                        f"{token_name} must be within the model vocabulary "
                        f"[0, {vocab_size}), got {token_id}"
                    )

            if len({mask_id, delete_token_id, split_token_id}) != 3:
                raise ValueError(
                    "JointThresholdInDel token IDs must be distinct, got "
                    f"mask_id={mask_id}, delete_token_id={delete_token_id}, "
                    f"split_token_id={split_token_id}"
                )

        max_running_requests = (
            1 if cfg.max_running_requests is None else cfg.max_running_requests
        )

        algorithm_config = {}
        if cfg.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(cfg.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f)

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)

        return DllmConfig(
            algorithm=cfg.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            first_done_first_out_mode=cfg.dllm_fdfo,
            delete_token_id=delete_token_id,
            split_token_id=split_token_id,
        )
