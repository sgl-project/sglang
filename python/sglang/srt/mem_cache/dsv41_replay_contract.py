"""Contracts for the CP>1 extension of upstream V4.1 encoder replay.

CP1 keeps the upstream folded extend, kernels, and cache identity. Resolve
parallelism before using these predicates; target and draft runners must not
share a process-global replay policy.
"""


def is_cp_encoder_replay(cfg) -> bool:
    return cfg.enable_encoder_swa_bounded_replay and cfg.attn_cp_size > 1


def validate_cp_encoder_replay(cfg, *, is_cuda: bool, is_sm90: bool) -> None:
    """Constrain the new path without changing any upstream CP1 recipe."""
    if not is_cp_encoder_replay(cfg):
        return
    if not is_cuda or not is_sm90:
        raise ValueError("CP encoder replay currently requires SM90 CUDA")
    if any(
        getattr(cfg, name) != 1
        for name in ("pp_size", "dp_size", "dcp_size", "attn_dp_size")
    ):
        raise ValueError("CP encoder replay requires PP=1, DP=1 and DCP=1")
    if not cfg.enable_prefill_cp or cfg.cp_strategy != "interleave":
        raise ValueError("CP encoder replay requires interleave prefill CP")
    if cfg.attn_cp_size != cfg.tp_size:
        raise ValueError("CP encoder replay requires attention TP=1")
    if cfg.cuda_graph_config.prefill.backend != "disabled":
        raise ValueError("CP encoder replay requires disabled prefill CUDA graphs")
    if cfg.disaggregation_mode == "decode":
        raise ValueError("encoder replay is a Prefill feature; Decode requires CP=1")
    if cfg.disaggregation_mode == "prefill" and (
        cfg.disaggregation_transfer_backend != "mooncake"
        or str(cfg.speculative_algorithm).upper() != "DSPARK"
    ):
        raise ValueError("CP encoder replay PD requires Mooncake and static DSpark")
    # The existing DSpark validator checks the static verify mode and wire layout.
    if cfg.moe_a2a_backend not in (None, "none"):
        raise ValueError("CP encoder replay currently requires native MoE dispatch")
    if cfg.enable_kv_cache_sharding:
        raise ValueError("CP encoder replay does not support KV cache sharding yet")
    if cfg.enable_hierarchical_cache:
        if cfg.hicache_host_memory_mode != "cache":
            raise ValueError("CP encoder replay requires cache host memory mode")
        if cfg.hicache_storage_backend not in (None, "mooncake"):
            raise ValueError("CP encoder replay L3 currently requires Mooncake Store")
