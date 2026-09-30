"""Validation for opt-in GEN routing precedence; never changes CTX affinity.

The existing DPBudget refreshes from scheduler shared-memory snapshots. In PD
GEN, total_tokens counts used KV plus pending preallocation/retraction tokens;
waiting request counts also include transfer requests. This is a load heuristic,
not a prediction of remaining output length or a hard HBM admission guarantee.
"""


def validate_decode_workload_config(cfg):
    if not cfg.disaggregation_decode_workload_balancing:
        return
    if (
        cfg.disaggregation_mode != "decode"
        or not cfg.enable_dp_attention
        or cfg.dp_size <= 1
        or cfg.load_balance_method != "total_tokens"
    ):
        raise ValueError(
            "--disaggregation-decode-workload-balancing requires decode mode, "
            "DP attention with dp_size > 1, and --load-balance-method total_tokens"
        )
    if cfg.disaggregation_decode_enable_radix_cache:
        raise ValueError(
            "Decode workload routing must not be combined with GEN prefix affinity"
        )
    # DPBudget and its load snapshots are sized for the launch DP ranks.
    # Do not accept a configuration that can activate ranks outside that budget.
    if cfg.max_ep_size is not None and cfg.max_ep_size > cfg.dp_size:
        raise ValueError(
            "Decode workload routing does not support --max-ep-size above --dp-size"
        )


def is_fake_bootstrap_request(req):
    # The server's per-rank PD warmup uses this sentinel and has no CTX.
    # Match the transport sentinel, not request text or a user/session identifier.
    from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

    host = getattr(req, "bootstrap_host", None)
    return isinstance(host, str) and host == FAKE_BOOTSTRAP_HOST


def validate_prefill_routing(req):
    if is_fake_bootstrap_request(req):
        return  # Preserve built-in transport warmup; no real CTX transfer exists.
    # Native DP-aware PD supplies this separately from the GEN rank. Do not
    # silently fall back to bootstrap-room hashing when CTX uses cache affinity.
    for name in ("disagg_prefill_dp_rank", "bootstrap_room", "bootstrap_host"):
        value = getattr(req, name, None)
        values = value if isinstance(value, list) else [value]
        if not values or any(v is None for v in values):
            raise ValueError(f"Decode workload routing requires explicit {name}")
    # bootstrap_port=None is valid: native PD uses the server default port.
    ranks = req.disagg_prefill_dp_rank
    ranks = ranks if isinstance(ranks, list) else [ranks]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in ranks):
        raise ValueError("disagg_prefill_dp_rank must contain nonnegative integers")
