import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def load_dllm_algorithm_config(algorithm_config_path: str | None) -> dict[str, Any]:
    if algorithm_config_path is None:
        return {}

    try:
        import yaml
    except ImportError:
        raise ImportError(
            "Please install PyYAML to use YAML config files. " "`pip install pyyaml`"
        )
    with open(algorithm_config_path, "r") as f:
        return yaml.safe_load(f) or {}


def should_defer_cuda_graph_capture(server_args: "ServerArgs") -> bool:
    if server_args.dllm_algorithm != "LinearSpec":
        return False

    algorithm_config = load_dllm_algorithm_config(server_args.dllm_algorithm_config)
    return bool(algorithm_config.get("lora_path"))


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        mask_id: int,
        max_running_requests: int,
        max_steps: int,
        causal_context: bool = False,
        block_size_tiers: list[dict[str, int]] | None = None,
        first_done_first_out_mode: bool = False,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        self.max_steps = max_steps
        self.first_done_first_out_mode = first_done_first_out_mode
        # Causal prefix KV for Nemotron diffusion-style instruct models.
        # Off for bidirectional-prefix models such as LLaDA2.
        self.causal_context = causal_context
        self.block_size_tiers: list[dict[str, int]] | None = None
        if block_size_tiers:
            if algorithm != "LinearSpec":
                raise ValueError(
                    "block_size_tiers are only supported with LinearSpec; "
                    f"got dllm_algorithm={algorithm}."
                )
            tiers = sorted(
                (
                    {
                        "max_running_bs": int(t["max_running_bs"]),
                        "block_size": int(t["block_size"]),
                    }
                    for t in block_size_tiers
                ),
                key=lambda t: t["max_running_bs"],
            )
            for i in range(1, len(tiers)):
                if tiers[i]["max_running_bs"] <= tiers[i - 1]["max_running_bs"]:
                    raise ValueError(
                        "block_size_tiers max_running_bs must be strictly "
                        f"ascending; got {[t['max_running_bs'] for t in tiers]}"
                    )
            if tiers[-1]["max_running_bs"] < max_running_requests:
                raise ValueError(
                    f"block_size_tiers last tier max_running_bs="
                    f"{tiers[-1]['max_running_bs']} is below "
                    f"max_running_requests={max_running_requests}; add a "
                    f"catch-all tier (e.g. max_running_bs: 9999)."
                )
            self.block_size_tiers = tiers
            max_tier_block = max(t["block_size"] for t in tiers)
            if self.block_size != max_tier_block:
                logger.warning(
                    "DllmConfig: overriding static block_size=%d with "
                    "max(block_size_tiers.block_size)=%d for KV pool sizing.",
                    self.block_size,
                    max_tier_block,
                )
                self.block_size = max_tier_block

    def select_block_size(self, running_bs: int) -> int:
        """Pick the block size for the current running batch size."""
        if not self.block_size_tiers:
            return self.block_size
        for tier in self.block_size_tiers:
            if running_bs <= tier["max_running_bs"]:
                return tier["block_size"]
        return self.block_size_tiers[-1]["block_size"]

    @staticmethod
    def from_server_args(
        server_args: "ServerArgs",
    ):
        if server_args.dllm_algorithm is None:
            return None

        from sglang.srt.configs.model_config import ModelConfig

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )
        DLLM_PARAMS = {
            "LLaDA2MoeModelLM": {"block_size": 32, "mask_id": 156895},
            "SDARForCausalLM": {"block_size": 4, "mask_id": 151669},
            "SDARMoeForCausalLM": {"block_size": 4, "mask_id": 151669},
            "DiffEncoderModel": {"block_size": 32, "mask_id": 151662},
            "NemotronLabsDiffusionModel": {"block_size": 32, "mask_id": 100},
        }

        arch = model_config.hf_config.architectures[0]
        if arch in DLLM_PARAMS:
            params = DLLM_PARAMS[arch]
            block_size = params["block_size"]
            mask_id = params["mask_id"]
        else:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")

        algorithm_config = load_dllm_algorithm_config(server_args.dllm_algorithm_config)
        block_size = algorithm_config.get("block_size", block_size)
        max_steps = algorithm_config.get("max_steps", block_size)
        causal_context = algorithm_config.get("causal_context", False)
        block_size_tiers = algorithm_config.get("block_size_tiers", None)
        # yaml override wins over server_args default; sync-only algorithms
        # like FastDiffuser / LinearSpec opt out via the yaml.
        first_done_first_out_mode = algorithm_config.get(
            "first_done_first_out_mode", server_args.dllm_fdfo
        )

        if server_args.max_running_requests is not None:
            max_running_requests = server_args.max_running_requests
        else:
            max_running_requests = 1

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            max_steps=max_steps,
            causal_context=causal_context,
            block_size_tiers=block_size_tiers,
            first_done_first_out_mode=first_done_first_out_mode,
        )
