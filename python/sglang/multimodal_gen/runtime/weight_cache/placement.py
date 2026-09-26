# SPDX-License-Identifier: Apache-2.0
"""Fail-closed Phase 1A admission before auto placement changes user intent."""

import math


def requested_component_names(args) -> tuple[str, ...]:
    names = tuple(
        "transformer" if name == "dit" else name
        for name in args.weight_cache_components
    )
    if (
        not names
        or len(set(names)) != len(names)
        or "transformer" not in names
        or not set(names).issubset({"transformer", "text_encoder"})
    ):
        raise ValueError(
            "Weight cache requires dit/transformer, optionally with text_encoder; duplicate selectors are not allowed"
        )
    # Canonical order makes aliases and CLI ordering describe the same bundle.
    return tuple(name for name in ("transformer", "text_encoder") if name in names)


def pin_requested_components(args) -> None:
    if args.weight_cache_mode == "off":
        return
    if args.weight_cache_mode != "client" or args.weight_cache_fallback != "error":
        raise ValueError(
            "Weight cache Phase 1A supports standalone daemons and strict client mode only"
        )
    requested = requested_component_names(args)
    if args.use_fsdp_inference:
        raise ValueError("Weight cache does not support explicit FSDP inference")
    if args.dit_cpu_offload or args.dit_layerwise_offload:
        raise ValueError(
            "Weight cache conflicts with explicit DiT CPU/layerwise offload"
        )
    from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
        cpu_offload_component_matches,
        layerwise_component_matches_any_selection,
        normalize_layerwise_offload_components,
    )

    layerwise = (
        normalize_layerwise_offload_components(args.layerwise_offload_components) or []
    )
    for name in requested:
        if (
            cpu_offload_component_matches(name, args.cpu_offload_components)
            or "all" in layerwise
            or (name == "transformer" and "dit" in layerwise)
            or layerwise_component_matches_any_selection(name, layerwise)
            or (name == "text_encoder" and args.text_encoder_cpu_offload)
        ):
            raise ValueError(
                f"Weight cache conflicts with an explicit {name} offload selector"
            )
        # Check original explicit controls before adding internal requirements.
        args.require_component_resident(name, feature_name="Weight cache")
        args.disable_fsdp_for_component(name)
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
    for name in requested_component_names(args):
        if args.residency_mode(
            name
        ) != "resident" or args.should_use_fsdp_for_component(name):
            raise ValueError(
                f"Auto placement violated the weight-cache resident/non-FSDP requirement for {name}"
            )


def local_device_index(args, local_rank: int = 0) -> int:
    return (
        args.gpu_ids[local_rank]
        if args.gpu_ids is not None
        else args.base_gpu_id + local_rank
    )
