import json
import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_dsa import DeepSeekDSAAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    DSABackendAdaptor,
    FlashAttentionAdaptor,
)
from sglang.srt.mem_cache.sparsity.backend.visibility_adaptor import (
    FlashAttentionVisibilityAdaptor,
    HBMResidentPlacement,
)
from sglang.srt.mem_cache.sparsity.config import KVSparsityConfig
from sglang.srt.mem_cache.sparsity.core.kv_sparsity_controller import (
    KVSparsityController,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    SparseConfig,
    SparseCoordinator,
)
from sglang.srt.mem_cache.sparsity.policies.streaming_llm import StreamingLLMPolicy

logger = logging.getLogger(__name__)

_KV_SPARSITY_POLICIES = {
    "streaming_llm": StreamingLLMPolicy,
}
_KV_SPARSITY_BACKEND_ALIASES = {
    "flashattention": "fa3",
}
_KV_SPARSITY_CONFIG_FIELDS = {
    "policy",
    "backend",
    "page_size",
    "min_sparse_tokens",
    "start_layer",
    "end_layer",
    "policy_config",
}

_global_sparse_coordinator: Optional[SparseCoordinator] = None

_ALGORITHM_REGISTRY = {
    "quest": lambda config, device, **kw: QuestAlgorithm(config, device, **kw),
    "deepseek_dsa": lambda config, device, **kw: DeepSeekDSAAlgorithm(
        config, device, **kw
    ),
}


def _create_sparse_algorithm(
    config: SparseConfig,
    device: torch.device,
    **kwargs,
) -> BaseSparseAlgorithm:
    algorithm_name = config.algorithm.lower()
    factory = _ALGORITHM_REGISTRY.get(algorithm_name)

    if factory is None:
        raise ValueError(f"Unknown sparse algorithm: {algorithm_name}")

    return factory(config, device, **kwargs)


def _create_backend_adaptor(
    backend: str,
    device: torch.device,
    sparse_algorithm: BaseSparseAlgorithm,
    req_to_token_pool,
):
    """Create backend adaptor."""
    if isinstance(sparse_algorithm, DeepSeekDSAAlgorithm):
        return DSABackendAdaptor(device, req_to_token_pool)

    if backend in ["fa3", "flashattention"]:
        return FlashAttentionAdaptor(device)

    raise ValueError(f"Unknown attention backend: {backend}")


def _parse_sparse_config(server_args) -> SparseConfig:
    """Parse hierarchical sparse config from JSON string.

    Required fields with defaults: top_k (2048), device_buffer_size (2*top_k),
    host_to_device_ratio (2), swap_in_block_size (960).
    Optional fields (default None): algorithm, backend, min_sparse_prompt_len,
    page_size. All remaining fields go to sparse_extra_config.
    """
    extra_config_str = server_args.hisparse_config
    if extra_config_str is not None:
        try:
            extra_config = json.loads(extra_config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse hisparse_config: {e}") from e
    else:
        extra_config = {}

    top_k = extra_config.pop("top_k", 2048)
    device_buffer_size = extra_config.pop("device_buffer_size", 2 * top_k)
    host_to_device_ratio = extra_config.pop("host_to_device_ratio", 2)
    swap_in_block_size = extra_config.pop("swap_in_block_size", 960)

    if device_buffer_size < top_k:
        raise ValueError(
            f"device_buffer_size ({device_buffer_size}) must be no smaller than top_k ({top_k})"
        )
    if not isinstance(swap_in_block_size, int) or isinstance(swap_in_block_size, bool):
        raise ValueError(
            f"swap_in_block_size must be an integer, got {swap_in_block_size!r}"
        )
    if swap_in_block_size <= 0 or swap_in_block_size > 1024:
        raise ValueError(
            f"swap_in_block_size ({swap_in_block_size}) must be in the range [1, 1024]"
        )

    algorithm = extra_config.pop("algorithm", None)
    backend = extra_config.pop("backend", None)
    min_sparse_prompt_len = extra_config.pop("min_sparse_prompt_len", None)
    page_size = extra_config.pop("page_size", None)

    return SparseConfig(
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        host_to_device_ratio=host_to_device_ratio,
        swap_in_block_size=swap_in_block_size,
        algorithm=algorithm,
        backend=backend,
        page_size=page_size,
        min_sparse_prompt_len=min_sparse_prompt_len,
        sparse_extra_config=extra_config,
    )


def parse_hisparse_config(server_args) -> SparseConfig:
    """Parse hisparse config from server_args, returning defaults if no config provided."""
    return _parse_sparse_config(server_args)


def parse_kv_sparsity_config(server_args) -> KVSparsityConfig:
    """Parse the independent HBM-resident sparse-attention configuration."""
    raw_config = getattr(server_args, "kv_cache_sparsity_config", None)
    if raw_config:
        try:
            values = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse kv_cache_sparsity_config: {exc}"
            ) from exc
        if not isinstance(values, dict):
            raise ValueError("kv_cache_sparsity_config must be a JSON object")
    else:
        values = {}

    unknown = sorted(set(values) - _KV_SPARSITY_CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"Unknown KV sparsity config field(s): {unknown}")

    runtime_page_size = getattr(server_args, "page_size", None)
    configured_page_size = values.get("page_size", runtime_page_size)
    if configured_page_size is None:
        configured_page_size = 1
    if runtime_page_size is not None and configured_page_size != runtime_page_size:
        raise ValueError(
            "KV sparsity page_size must match the runtime --page-size "
            f"({runtime_page_size}), got {configured_page_size}"
        )

    policy = values.get("policy", "streaming_llm")
    backend = values.get("backend", "fa3")
    if not isinstance(policy, str) or not policy.strip():
        raise ValueError("KV sparsity policy must be a non-empty string")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("KV sparsity backend must be a non-empty string")
    policy = policy.strip().lower()
    backend = _KV_SPARSITY_BACKEND_ALIASES.get(
        backend.strip().lower(), backend.strip().lower()
    )

    return KVSparsityConfig(
        policy=policy,
        backend=backend,
        page_size=configured_page_size,
        min_sparse_tokens=values.get("min_sparse_tokens", 2048),
        start_layer=values.get("start_layer", 0),
        end_layer=values.get("end_layer", -1),
        policy_config=values.get("policy_config", {}),
    )


def create_kv_sparsity_controller(
    *,
    device: torch.device,
    req_to_token_pool,
    start_layer: int,
    end_layer: int,
    server_args,
) -> KVSparsityController:
    config = parse_kv_sparsity_config(server_args)
    policy_class = _KV_SPARSITY_POLICIES.get(config.policy)
    if policy_class is None:
        raise ValueError(
            f"Unknown KV sparsity policy {config.policy!r}; "
            f"available policies: {sorted(_KV_SPARSITY_POLICIES)}"
        )
    if config.backend != "fa3":
        raise ValueError("The initial KV sparsity runtime supports only FA3")

    policy = policy_class(config, device)
    placement = HBMResidentPlacement(
        req_to_token=req_to_token_pool.req_to_token,
        page_size=config.page_size,
    )
    adaptor = FlashAttentionVisibilityAdaptor(placement)
    return KVSparsityController(
        config=config,
        policy=policy,
        adaptor=adaptor,
        req_to_token_pool=req_to_token_pool,
        start_layer=start_layer,
        end_layer=end_layer,
    )


def create_sparse_coordinator(
    device: torch.device,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    server_args,
    **kwargs,
) -> SparseCoordinator:
    config = _parse_sparse_config(server_args)
    algorithm = _create_sparse_algorithm(config, device, **kwargs)
    backend_adaptor = _create_backend_adaptor(
        config.backend, device, algorithm, req_to_token_pool
    )

    coordinator = SparseCoordinator(
        config=config,
        algorithm=algorithm,
        backend_adaptor=backend_adaptor,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        start_layer=start_layer,
        end_layer=end_layer,
        device=device,
    )
    register_sparse_coordinator(coordinator)
    return coordinator


def register_sparse_coordinator(coordinator: SparseCoordinator) -> None:
    global _global_sparse_coordinator
    _global_sparse_coordinator = coordinator


def get_sparse_coordinator() -> Optional[SparseCoordinator]:
    return _global_sparse_coordinator
