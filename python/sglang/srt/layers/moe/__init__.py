from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    MoeA2ABackend,
    MoeRunnerBackend,
    get_deepep_config,
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
    get_tbo_token_distribution_threshold,
    initialize_moe_config,
    is_tbo_enabled,
    should_skip_post_experts_all_reduce,
    should_use_dp_reduce_scatterv,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)


def __getattr__(name: str):
    if name == "MoeRunner":
        from sglang.srt.layers.moe.moe_runner.runner import MoeRunner

        return MoeRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DeepEPMode",
    "MoeA2ABackend",
    "MoeRunner",
    "MoeRunnerConfig",
    "MoeRunnerBackend",
    "initialize_moe_config",
    "get_moe_a2a_backend",
    "get_moe_runner_backend",
    "get_deepep_mode",
    "should_skip_post_experts_all_reduce",
    "should_use_dp_reduce_scatterv",
    "should_use_flashinfer_cutlass_moe_fp4_allgather",
    "is_tbo_enabled",
    "get_tbo_token_distribution_threshold",
    "get_deepep_config",
]
