"""Argument checks for Ascend expert load balancing."""


def validate_async_eplb_args(cfg) -> None:
    if cfg.device != "npu" or not cfg.enable_eplb:
        raise ValueError(
            "--eplb-async-rebalance requires --enable-eplb and --device npu"
        )
    if cfg.ep_size != cfg.tp_size or cfg.tp_size <= 1:
        raise ValueError(
            "Async NPU EPLB requires resolved EP size = TP size > 1 "
            f"(got EP={cfg.ep_size}, TP={cfg.tp_size})"
        )
    if cfg.ep_dispatch_algorithm not in (None, "static", "dynamic"):
        raise ValueError("Async NPU EPLB supports static or dynamic dispatch")
    if cfg.weight_cache_mode != "off":
        raise ValueError(
            "--eplb-async-rebalance requires --weight-cache-mode off because "
            "expert migration writes to model weights in place"
        )
    if cfg.remote_instance_weight_loader_start_seed_via_transfer_engine:
        raise ValueError(
            "An async EPLB worker cannot serve as a transfer-engine weight seed: "
            "the registered weights can move to different expert slots"
        )
    if (
        cfg.pp_size != 1
        or cfg.elastic_ep_backend is not None
        or cfg.enable_paged_experts
        or cfg.enable_memory_saver
        or cfg.cpu_offload_gb
        or cfg.offload_group_size > 0
        or cfg.enable_two_batch_overlap
        or cfg.enable_single_batch_overlap
    ):
        raise ValueError(
            "Async NPU EPLB requires fixed PP=1 resident weights and single-batch execution"
        )
    chunk_size = cfg.eplb_rebalance_layers_per_chunk
    if chunk_size not in (None, 1):
        raise ValueError("Async NPU EPLB stages one layer per forward boundary")
    window = cfg.expert_distribution_recorder_buffer_size
    interval = cfg.eplb_rebalance_num_iterations
    if interval <= 0 or (window is not None and not 0 < window <= interval):
        raise ValueError(
            "Async NPU EPLB requires a positive recorder window no longer than "
            "--eplb-rebalance-num-iterations"
        )
    if cfg.eplb_algorithm.startswith("elasticity"):
        raise ValueError("Async NPU EPLB does not support elastic placement")


def validate_ascend_ops_eplb(config) -> None:
    if "fused_op" not in (config.prefill, config.decode):
        return

    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

    if not is_npu_arch35():
        raise ValueError(
            "Ascend arch22 fused_op keeps per-expert NZ weights outside the "
            "stacked parameters moved by EPLB. Use alltoallv or allgather for "
            "prefill and low_latency for decode when enabling EPLB."
        )
