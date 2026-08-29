# SPDX-License-Identifier: Apache-2.0
"""Server-argument validation that spans no single family."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from sglang.srt.arg_groups.overrides import (
    resolving_view,
)
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    parse_ib_device_config,
)
from sglang.srt.utils.common import is_hip, is_npu, torch_release
from sglang.srt.utils.runai_utils import is_runai_obj_uri

logger = logging.getLogger(__name__)


def check_server_args(server_args: Any):
    from sglang.srt.arg_groups.lora_hook import check_lora_server_args

    cfg = resolving_view(server_args)

    # Check parallel size constraints
    if cfg.ep_join_mode != "scale":
        assert (
            cfg.tp_size * cfg.pp_size
        ) % cfg.nnodes == 0, "tp_size must be divisible by number of nodes"

    assert cfg.pp_max_micro_batch_size is None or cfg.pp_max_micro_batch_size >= 1, (
        "pp_max_micro_batch_size must be a positive integer or None (for auto-compute). "
        f"Got: {cfg.pp_max_micro_batch_size}"
    )

    assert not (cfg.disable_cuda_graph_padding and cfg.enable_torch_compile), (
        "--disable-cuda-graph-padding is incompatible with --enable-torch-compile. "
        "With padding disabled, every distinct batch size gets its own torch.compile + "
        "Triton autotune cycle (O(max_batch_size) compilations) instead of the small fixed "
        "set of padded bucket sizes, causing engine initialisation to stall for many minutes. "
        "Remove --disable-cuda-graph-padding or --enable-torch-compile."
    )

    if cfg.pp_size > 1:
        assert (
            cfg.disable_overlap_schedule and cfg.speculative_algorithm is None
        ), "Pipeline parallelism is not compatible with overlap schedule, speculative decoding"
        assert cfg.min_free_slots_delay is None, (
            "--min-free-slots-delay is not supported with pipeline "
            "parallelism: allocatable slots per microbatch are bounded by "
            "pp-max-micro-batch-size, so the threshold may never be reached"
        )

    assert not (
        cfg.dp_size > 1 and cfg.nnodes != 1 and not cfg.enable_dp_attention
    ), "multi-node data parallel is not supported unless dp attention!"

    assert cfg.base_gpu_id >= 0, "base_gpu_id must be non-negative"
    assert cfg.gpu_id_step >= 1, "gpu_id_step must be positive"

    assert cfg.moe_dense_tp_size in (
        None,
        1,
        cfg.tp_size,
    ), "moe_dense_tp_size only supports None, 1, or tp_size currently"

    # Check served model name to not have colon as it is reserved for LoRA adapter syntax
    if not is_runai_obj_uri(cfg.served_model_name):
        assert ":" not in cfg.served_model_name, (
            "served_model_name cannot contain a colon (':') character. "
            "The colon is reserved for the 'model:adapter' syntax used in LoRA adapter specification. "
            f"Invalid value: '{cfg.served_model_name}'"
        )

    # Check LoRA
    check_lora_server_args(server_args)

    # Check speculative decoding
    if cfg.speculative_algorithm is not None:
        assert (
            not cfg.enable_mixed_chunk
        ), "enable_mixed_chunk is required for speculative decoding"

    # Check chunked prefill
    # Skip validation if chunked prefill is disabled (i.e., size <= 0).
    # Skip validation if disaggregation mode is decode.
    if cfg.chunked_prefill_size > 0 and cfg.disaggregation_mode != "decode":
        assert (
            cfg.chunked_prefill_size % cfg.page_size == 0
        ), "chunked_prefill_size must be divisible by page_size"

    # Check pdmux
    if cfg.enable_pdmux:
        assert (
            cfg.pp_size == 1
        ), "PD-Multiplexing is only supported with pipeline parallelism disabled (pp_size=1)."
        assert (
            cfg.chunked_prefill_size == -1
        ), "PD-Multiplexing is not compatible with chunked prefill."
        assert (
            cfg.disaggregation_mode == "null"
        ), "PD-Multiplexing is not compatible with disaggregation mode."
        assert (
            cfg.disable_overlap_schedule
        ), "PD-Multiplexing is not compatible with overlap schedule."

        # NOTE: CUDA Green Context may encounter potential issues with CudaGraph on torch 2.7.x – 2.8.x, leading to performance degradation.
        import torch

        if torch_release >= (2, 7):
            logger.warning(
                "WARNING: PD-Multiplexing may experience performance degradation with torch versions > 2.6.x.\n"
                f"  Current torch version is {torch.__version__}.\n"
                "  Please manually install torch 2.6.x."
            )

    assert cfg.tokenizer_worker_num > 0, "Tokenizer worker num must >= 1"
    assert cfg.detokenizer_worker_num > 0, "Detokenizer worker num must >= 1"
    assert cfg.mm_processor_worker_num >= 0, "Multimodal processor worker num must >= 0"
    assert cfg.mm_io_worker_num >= 0, "Multimodal I/O worker num must >= 0"
    validate_buckets_rule(
        server_args, "--prompt-tokens-buckets", cfg.prompt_tokens_buckets
    )
    validate_buckets_rule(
        server_args, "--generation-tokens-buckets", cfg.generation_tokens_buckets
    )

    # Check scheduling policy
    if cfg.enable_priority_scheduling:
        assert cfg.schedule_policy in [
            "fcfs",
            "lof",
        ], f"To use priority scheduling, schedule_policy must be 'fcfs' or 'lof'. '{cfg.schedule_policy}' is not supported."
        if cfg.default_priority_value is None:
            logger.warning(
                "--default-priority-value is not set while --enable-priority-scheduling is enabled. "
                "Requests without explicit priority will have priority=None, "
                "resulting in priority='None' string labels in Prometheus metrics."
            )
    else:
        if cfg.disable_priority_preemption:
            logger.warning(
                "--disable-priority-preemption has no effect without --enable-priority-scheduling"
            )
        if cfg.default_priority_value is not None:
            logger.warning(
                "--default-priority-value has no effect without --enable-priority-scheduling"
            )
    if cfg.retraction_policy == "priority" and not cfg.enable_priority_scheduling:
        raise ValueError(
            "--retraction-policy priority requires --enable-priority-scheduling"
        )

    # Check hisparse
    # Moved to the resolution pipeline (arg_groups/overrides.py:
    # _hisparse_validation), invoked here at its legacy slot.
    from sglang.srt.arg_groups.overrides import (
        _hisparse_validation,
        run_post_process_pass,
    )

    run_post_process_pass(server_args, _hisparse_validation)

    assert (
        cfg.schedule_conservativeness >= 0
    ), "schedule_conservativeness must be non-negative"

    if cfg.model_impl == "mindspore":
        assert is_npu(), "MindSpore model impl is only supported on Ascend npu."

    # Check metrics labels
    if (
        not cfg.tokenizer_metrics_custom_labels_header
        and cfg.tokenizer_metrics_allowed_custom_labels
    ):
        raise ValueError(
            "Please set --tokenizer-metrics-custom-labels-header when setting --tokenizer-metrics-allowed-custom-labels."
        )

    # Check metrics exporters
    if cfg.export_metrics_to_file and cfg.export_metrics_to_file_dir is None:
        raise ValueError(
            "--export-metrics-to-file-dir is required when --export-metrics-to-file is enabled"
        )

    # Check two batch overlap backend requirement.
    check_two_batch_overlap(server_args)

    # Check communications compression
    if cfg.enable_quant_communications and cfg.tp_size == 1:
        raise ValueError("Communications quantization is only used with tp_size != 1")

    if cfg.enable_quant_communications and cfg.device != "npu":
        raise ValueError("Communications quantization is only supported for NPU device")

    # grpc_port is None for HTTP-only launches, so the == comparison is
    # already False there; no explicit None check needed.
    if not (cfg.smg_grpc_mode or cfg.grpc_mode) and cfg.grpc_port == cfg.port:
        raise ValueError(
            f"--grpc-port ({cfg.grpc_port}) must differ from --port ({cfg.port})"
        )

    # TODO: Also validate grpc_port != metrics_http_port and grpc_port != nccl_port
    # to avoid opaque bind errors at runtime. Deferred because metrics_http_port
    # and nccl_port have dynamic defaults that may not be resolved yet here.

    if cfg.gc_threshold:
        if not (1 <= len(cfg.gc_threshold) <= 3):
            raise ValueError(
                "When setting gc_threshold, it must contain 1 to 3 integers."
            )

    if cfg.kv_canary_sweep_interval > 0 and cfg.kv_canary == "none":
        raise ValueError(
            "--kv-canary-sweep-interval requires --kv-canary in {log, raise}"
        )

    check_load_publish_args(server_args)


def validate_buckets_rule(server_args: Any, arg_name: str, buckets_rule: List[str]):
    if not buckets_rule:
        return

    assert len(buckets_rule) > 0, f"{arg_name} cannot be empty list"
    rule = buckets_rule[0]
    assert rule in [
        "tse",
        "default",
        "custom",
    ], f"Unsupported {arg_name} rule type: '{rule}'. Must be one of: 'tse', 'default', 'custom'"

    if rule == "tse":
        assert (
            len(buckets_rule) == 4
        ), f"{arg_name} TSE rule requires exactly 4 parameters: ['tse', middle, base, count], got {len(buckets_rule)}"
        try:
            middle = float(buckets_rule[1])
            base = float(buckets_rule[2])
            count = int(buckets_rule[3])
        except (ValueError, IndexError):
            assert (
                False
            ), f"{arg_name} TSE rule parameters must be: ['tse', <float:middle>, <float:base>, <int:count>]"
        assert base > 1, f"{arg_name} TSE base must be larger than 1, got: {base}"
        assert count > 0, f"{arg_name} TSE count must be positive, got: {count}"
        assert middle > 0, f"{arg_name} TSE middle must be positive, got: {middle}"

    elif rule == "default":
        assert (
            len(buckets_rule) == 1
        ), f"{arg_name} default rule should only have one parameter: ['default'], got {len(buckets_rule)}"

    elif rule == "custom":
        assert (
            len(buckets_rule) >= 2
        ), f"{arg_name} custom rule requires at least one bucket value: ['custom', value1, ...]"
        try:
            bucket_values = [float(x) for x in buckets_rule[1:]]
        except ValueError:
            assert False, f"{arg_name} custom rule bucket values must be numeric"
        assert len(set(bucket_values)) == len(
            bucket_values
        ), f"{arg_name} custom rule bucket values should not contain duplicates"
        assert all(
            val >= 0 for val in bucket_values
        ), f"{arg_name} custom rule bucket values should be non-negative"


def check_load_publish_args(server_args: Any):
    """Fail fast at the entrypoint on a --load-publish-endpoint the
    scheduler would decline (no active kv-events publisher to advertise
    through, unbindable, overlapping the KV range, u16 overflow) rather
    than only warning — or silently doing nothing — from a scheduler
    subprocess. Routes through the same resolver the scheduler binds and
    /server_info advertises with."""
    server_cfg = resolving_view(server_args)
    mode = (server_cfg.load_publish_endpoint or "").strip()
    if not mode or mode.lower() == "off":
        return  # disabled; nothing to validate

    from sglang.srt.disaggregation.kv_events import (
        KVEventsConfig,
        resolve_load_pub_range,
    )

    if not server_cfg.kv_events_config:
        raise ValueError(
            "--load-publish-endpoint requires --kv-events-config: routers"
            " discover the load range through /server_info's kv_events"
            " block, absent without a publisher."
        )
    try:
        cfg = KVEventsConfig.from_cli(server_cfg.kv_events_config)
    except Exception as e:
        raise ValueError(f"--kv-events-config is not parseable: {e}")
    if cfg.publisher == "null" or not cfg.endpoint:
        raise ValueError(
            "--load-publish-endpoint needs an active --kv-events-config"
            " publisher; got publisher='null' or an empty endpoint."
        )
    _, reason = resolve_load_pub_range(
        kv_endpoint=cfg.endpoint,
        replay_endpoint=cfg.replay_endpoint,
        dp_size=server_cfg.dp_size,
        load_publish_endpoint=mode,
    )
    if reason:
        raise ValueError(reason)


def validate_ib_devices(server_args: Any, device_str: Optional[str]) -> Optional[str]:
    """
    Validate IB devices before passing to mooncake.

    Args:
        device_str: Comma-separated IB device names, a per-GPU JSON mapping,
            or a path to a JSON file containing that mapping.

    Returns:
        A normalized comma-separated string or per-GPU JSON mapping string, or None if input is None.
    """
    if device_str is None:
        logger.warning(
            "No IB devices specified for Mooncake backend, falling back to auto discovery."
        )
        return None

    def _normalize_device_group(raw_value: str, context: str) -> str:
        if not isinstance(raw_value, str):
            raise ValueError(
                f"Invalid IB device format for {context}: expected a string. "
                f"Got {type(raw_value)}"
            )
        devices = [d.strip() for d in raw_value.split(",") if d.strip()]
        if not devices:
            raise ValueError(f"No valid IB devices specified for {context}")
        unique_devices = list(dict.fromkeys(devices))
        if len(unique_devices) != len(devices):
            logger.warning(
                "Duplicate IB devices specified for %s: %s. Deduplicating to: %s",
                context,
                raw_value,
                ",".join(unique_devices),
            )
        invalid_devices = [d for d in unique_devices if d not in available_devices]
        if len(invalid_devices) != 0:
            raise ValueError(
                f"Invalid IB devices specified for {context}: {invalid_devices}. "
                f"Available devices: {sorted(available_devices)}"
            )
        return ",".join(unique_devices)

    normalized_input = device_str.strip()
    if not normalized_input:
        raise ValueError("No valid IB devices specified")

    # Get available IB devices from sysfs
    ib_sysfs_path = "/sys/class/infiniband"
    if not os.path.isdir(ib_sysfs_path):
        raise RuntimeError(
            f"InfiniBand sysfs path not found: {ib_sysfs_path}. "
            "Please ensure InfiniBand drivers are installed."
        )

    available_devices = set(os.listdir(ib_sysfs_path))
    if len(available_devices) == 0:
        raise RuntimeError(f"No IB devices found in {ib_sysfs_path}")

    parsed_config = parse_ib_device_config(normalized_input)
    if isinstance(parsed_config, str):
        return _normalize_device_group(normalized_input, "all GPUs")
    assert parsed_config is not None

    normalized_mapping: Dict[str, str] = {}
    for gpu_key, gpu_devices in parsed_config.items():
        normalized_key = str(gpu_key)
        normalized_mapping[normalized_key] = _normalize_device_group(
            gpu_devices, f"GPU {normalized_key}"
        )

    if not normalized_mapping:
        raise ValueError("No valid GPU mappings found in IB device JSON")

    return json.dumps(normalized_mapping, separators=(",", ":"))


def validate_experimental_sgl_marlin(server_args: Any):
    view = server_args._resolved()
    if view.moe_runner_backend != "experimental_sgl_marlin":
        return

    # ===== TO BE REFACTORED ====
    from sglang.srt.lora.marlin_lora_temp.policy import (
        validate_experimental_sgl_marlin_server_args,
    )

    validate_experimental_sgl_marlin_server_args(server_args, view)


def validate_prefill_decode_interval(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.prefill_decode_interval < 0:
        raise ValueError("--prefill-decode-interval must be non-negative.")


def check_two_batch_overlap(server_args: Any):
    # With no EP a2a backend, two-batch-overlap is only valid on the non-EP
    # DP TP-MoE path (overlapping the DP all_gatherv / reduce_scatterv with
    # the other ubatch's compute), which requires DP attention. Enabling it
    # there needs no extra opt-in env flag.
    cfg = resolving_view(server_args)

    cp_tbo = (
        is_hip()
        and cfg.enable_dsa_prefill_context_parallel
        and cfg.dsa_prefill_cp_mode == "round-robin-split"
    )
    if (
        cfg.enable_two_batch_overlap
        and cfg.moe_a2a_backend == "none"
        and not cfg.enable_dp_attention
        and not cp_tbo
    ):
        raise ValueError(
            "When enabling two batch overlap without an EP a2a backend "
            "(moe_a2a_backend='none'), --enable-dp-attention is required "
            "(DeepSeek-V4 non-EP DP TBO path)."
        )
