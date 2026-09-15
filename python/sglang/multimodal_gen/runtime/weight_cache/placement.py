# SPDX-License-Identifier: Apache-2.0
"""Fail-closed Phase 1A admission before auto placement changes user intent."""

import math


def pin_requested_components(args) -> None:
    if args.weight_cache_mode == "off":
        return
    if args.weight_cache_mode != "client" or args.weight_cache_fallback != "error":
        raise ValueError(
            "Weight cache Phase 1A supports standalone daemons and strict client mode only"
        )
    requested = args.weight_cache_components
    if not requested or set(requested) not in ({"dit"}, {"transformer"}):
        raise ValueError(
            "Weight cache Phase 1A supports only the transformer (dit) component"
        )
    if args.use_fsdp_inference and args.is_arg_explicitly_set("use_fsdp_inference"):
        raise ValueError("Weight cache does not support explicit FSDP inference")
    # This checks original canonical/group and legacy explicit controls before
    # adding an internal requirement. It does not forge user-explicit options.
    args.require_component_resident("transformer", feature_name="Weight cache")
    args.disable_fsdp_for_component("transformer")
    args.use_fsdp_inference = False
    if args.lora_path is not None or any(
        name.startswith("lora_") for name in args._explicit_arg_names
    ):
        raise ValueError("Weight cache Phase 1A does not support LoRA options")
    if args.backend == "diffusers" or args.comfyui_mode:
        raise ValueError("Weight cache requires the native SGLang pipeline")
    if args.disagg_role != "monolithic" or args.disagg_mode:
        raise ValueError("Weight cache Phase 1A does not support disaggregation")
    if not math.isfinite(args.weight_cache_timeout) or args.weight_cache_timeout <= 0:
        raise ValueError("weight_cache_timeout must be positive and finite")
    if (
        type(args.weight_cache_max_deliveries) is not int
        or args.weight_cache_max_deliveries <= 0
    ):
        raise ValueError("weight_cache_max_deliveries must be a positive integer")


def validate_resolved_arguments(args) -> None:
    if args.weight_cache_mode == "off":
        return
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if not current_platform.is_cuda():
        raise ValueError("Weight cache Phase 1A requires CUDA")
    for name in (
        "num_gpus",
        "nnodes",
        "tp_size",
        "sp_degree",
        "ulysses_degree",
        "ring_degree",
        "kv_gather_degree",
        "cfg_parallel_degree",
        "dp_size",
    ):
        if getattr(args, name) != 1:
            raise ValueError(f"Weight cache Phase 1A requires {name}=1")
    if args.node_rank != 0:
        raise ValueError("Weight cache Phase 1A requires node_rank=0")
    if args.residency_mode(
        "transformer"
    ) != "resident" or args.should_use_fsdp_for_component("transformer"):
        raise ValueError(
            "Auto placement violated the weight-cache resident/non-FSDP requirement"
        )


def local_device_index(args, local_rank: int = 0) -> int:
    return (
        args.gpu_ids[local_rank]
        if args.gpu_ids is not None
        else args.base_gpu_id + local_rank
    )
