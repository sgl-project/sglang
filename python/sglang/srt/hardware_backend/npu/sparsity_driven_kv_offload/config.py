"""Configuration and validation for sparsity-driven KV offload."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

_ENABLE_ENV_VAR = "SGLANG_ENABLE_SPARSITY_DRIVEN_KV_OFFLOAD"


def is_sparsity_driven_kv_offload_requested() -> bool:
    return get_bool_env_var(_ENABLE_ENV_VAR)


def is_sparsity_driven_kv_offload_enabled(
    *,
    model_config: ModelConfig,
    server_args: ServerArgs,
    use_mla_backend: bool,
) -> bool:
    if not is_sparsity_driven_kv_offload_requested():
        return False

    if not (
        is_npu()
        and server_args.attention_backend == "ascend"
        and use_mla_backend
        and is_deepseek_dsa(model_config.hf_config)
    ):
        raise ValueError(
            f"{_ENABLE_ENV_VAR} requires an NPU DeepSeek DSA model using "
            "the Ascend MLA attention backend."
        )
    if server_args.max_running_requests is None:
        raise ValueError(
            f"{_ENABLE_ENV_VAR} requires an explicit "
            "--max-running-requests to bound the per-process host KV allocation."
        )
    return True


def get_sparsity_driven_kv_offload_cell_size(
    *,
    model_config: ModelConfig,
    server_args: ServerArgs,
    use_mla_backend: bool,
    num_layers: int,
    element_size: int,
) -> Optional[int]:
    if not is_sparsity_driven_kv_offload_enabled(
        model_config=model_config,
        server_args=server_args,
        use_mla_backend=use_mla_backend,
    ):
        return None

    index_head_dim = model_config.index_head_dim
    if index_head_dim is None:
        raise ValueError("Sparsity-driven KV offload requires an index KV cache.")
    return index_head_dim * num_layers * element_size
