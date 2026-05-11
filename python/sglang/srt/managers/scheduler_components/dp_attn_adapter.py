from __future__ import annotations  # noqa: F401


class SchedulerDPAttnAdapter:
    """DP-attention batch synchronization adapter. Composition target on
    Scheduler (``self.dp_attn_adapter``). Owns no mutable state."""

    def __init__(
        self,
        *,
        tp_group,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        tree_cache,
        offload_tags,
        ps,
        server_args,
        model_config,
        enable_overlap: bool,
        spec_algorithm,
        require_mlp_sync: bool,
    ) -> None:
        self.tp_group = tp_group
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.tree_cache = tree_cache
        self.offload_tags = offload_tags
        self.ps = ps
        self.server_args = server_args
        self.model_config = model_config
        self.enable_overlap = enable_overlap
        self.spec_algorithm = spec_algorithm
        self.require_mlp_sync = require_mlp_sync
