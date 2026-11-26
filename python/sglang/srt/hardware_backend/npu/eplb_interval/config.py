"""Resolve opt-in EPLB interval settings without importing a device runtime."""

import logging

from sglang.srt.arg_groups.model_override_base import model_config_of
from sglang.srt.arg_groups.overrides import declare_resolution

logger = logging.getLogger(__name__)


def _interval_replay_fallback(cfg):
    if cfg.enable_pd_role_switch:
        return "does not support switching PD roles at runtime"
    if cfg.device != "npu":
        return f"unsupported device {cfg.device}"
    if (
        cfg.pp_size != 1
        or cfg.nnodes != 1
        or cfg.dp_size != 1
        or cfg.tp_size != cfg.ep_size
        or cfg.ep_size <= 1
        or cfg.elastic_ep_backend is not None
    ):
        return "requires a single non-elastic TP=EP=WORLD group with PP=1"
    if (
        cfg.expert_distribution_recorder_mode != "stat"
        or cfg.ep_dispatch_algorithm != "static"
    ):
        return f"requires stat recording and static dispatch (got {cfg.expert_distribution_recorder_mode}, {cfg.ep_dispatch_algorithm})"
    if cfg.moe_a2a_backend not in ("ascend_ops", "ascend_fuseep", "deepep"):
        return f"unsupported dispatcher {cfg.moe_a2a_backend}"
    if cfg.eplb_algorithm not in ("auto", "deepseek", "deepseek_hierarchical"):
        return f"unsupported layout algorithm {cfg.eplb_algorithm}"
    if (
        cfg.eplb_rebalance_layers_per_chunk is not None
        or cfg.eplb_min_rebalancing_utilization_threshold != 1.0
    ):
        return "requires full-model updates and the default rebalance threshold"
    return None


def resolve_interval(server_args, cfg):
    source = "_handle_expert_distribution_metrics"
    if cfg.eplb_interval_policy not in ("recommend", "auto") or not cfg.enable_eplb:
        raise ValueError(
            "EPLB interval tuning requires --enable-eplb and recommend or auto."
        )
    fallback = _interval_replay_fallback(cfg)
    if fallback is not None:
        interval = cfg.eplb_rebalance_num_iterations
        if interval is None:
            interval = max(1000, cfg.expert_distribution_recorder_buffer_size or 0)
        if (cfg.expert_distribution_recorder_buffer_size or 0) > interval:
            raise ValueError(
                "This configuration requires fixed EPLB checks. Set the interval at least as large as the recorder window."
            )
        logger.warning(
            "EPLB interval replay %s. Using fixed interval %d.",
            fallback,
            interval,
        )
        declare_resolution(
            server_args,
            source,
            eplb_interval_policy="fixed",
            eplb_rebalance_num_iterations=interval,
        )
        return

    window = cfg.expert_distribution_recorder_buffer_size
    interval = cfg.eplb_rebalance_num_iterations
    if interval is None:
        interval = 1000
        if (
            cfg.disaggregation_mode in ("prefill", "decode")
            and cfg.ep_size == cfg.attn_dp_size == 16
            and cfg.ep_num_redundant_experts == 16
            and cfg.quantization == "modelslim"
        ):
            from sglang.srt.utils.device import get_device_name

            hf = model_config_of(server_args).hf_config
            if (
                "DeepseekV4ForCausalLM" in (hf.architectures or [])
                and getattr(hf, "n_routed_experts", None) == 256
                and (get_device_name() or "")
                .lower()
                .startswith(("ascend910_93", "ascend910c"))
            ):
                source = "ascend_pd_eplb_interval"
                interval, default_window = (
                    (50, 50) if cfg.disaggregation_mode == "prefill" else (4000, 1000)
                )
                if window is None:
                    window = default_window
    if window is None:
        window = interval
    if window <= 0 or not 0 < interval <= 2**31 - 1:
        raise ValueError(
            "EPLB interval tuning requires a positive recorder window and an interval in [1, 2147483647]."
        )
    declare_resolution(
        server_args,
        source,
        eplb_rebalance_num_iterations=interval,
        expert_distribution_recorder_buffer_size=window,
    )
    logger.info(
        "EPLB %s %s starts at interval=%d window=%d",
        cfg.disaggregation_mode,
        cfg.eplb_interval_policy,
        interval,
        window,
    )
