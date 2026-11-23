"""Startup constraints for the eager fixed-TP virtual pipeline."""

from __future__ import annotations

import os
from typing import Any

from sglang.srt.arg_groups.model_override_base import model_config_of, resolving_view
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.arg_groups.resolution_hooks import run_hook
from sglang.srt.configs.vpp import resolve_vpp_layout
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig, Phase
from sglang.srt.runtime_context import get_platform


def validate_vpp_options(server_args: Any) -> None:
    """Reject unsupported topology even on the dummy-model startup path."""
    cfg = resolving_view(server_args)
    if type(cfg.pp_virtual_stages) is not int or cfg.pp_virtual_stages < 1:
        raise ValueError("--pp-virtual-stages must be a positive integer")
    if cfg.pp_virtual_stages == 1:
        if cfg.pp_vpp_layer_partition is not None:
            raise ValueError(
                "--pp-vpp-layer-partition requires --pp-virtual-stages > 1"
            )
        return
    device = cfg.device
    if (device is not None and device != "cuda") or (
        device is None and not get_platform().is_cuda
    ):
        raise ValueError("VPP requires CUDA")
    if cfg.pp_size < 2 or cfg.pp_async_batch_depth < 0:
        raise ValueError("VPP requires --pp-size > 1 and nonnegative async batch depth")
    unsupported = (
        ("data parallelism", cfg.dp_size != 1 or cfg.attn_dp_size != 1),
        (
            "context parallelism",
            cfg.attn_cp_size != 1 or cfg.dcp_size != 1 or cfg.enable_prefill_cp,
        ),
        ("MoE data/weight parallelism", cfg.moe_dp_size != 1 or cfg.dwdp_size != 1),
        ("speculative decoding", cfg.speculative_algorithm is not None),
        ("LoRA", bool(cfg.enable_lora) or bool(cfg.lora_paths)),
        ("HiCache", cfg.enable_hierarchical_cache),
        ("HiSparse", cfg.enable_hisparse),
        ("external KV linker", cfg.enable_unified_cache_external_linker),
        ("diffusion models", cfg.dllm_algorithm is not None),
        ("sequence parallelism", cfg.enable_layernorm_sp),
        ("attention input scattering", cfg.enable_attn_tp_input_scattered),
        ("two-batch overlap", cfg.enable_two_batch_overlap),
        (
            "hidden-state output",
            cfg.enable_return_hidden_states or cfg.enable_return_aux_hidden_states,
        ),
        ("routed-expert output", cfg.enable_return_routed_experts),
        (
            "bounded SWA replay",
            cfg.enable_encoder_swa_bounded_replay
            or cfg.enable_decoder_swa_bounded_replay,
        ),
        ("torch.compile", cfg.enable_torch_compile),
        ("elastic EP", cfg.elastic_ep_backend is not None),
        ("non-SGLang model implementations", cfg.model_impl not in ("auto", "sglang")),
        ("embedding mode", cfg.is_embedding),
        ("kernel plugins", cfg.enable_amd_monokernel or cfg.enable_nvidia_megakernel),
    )
    for feature, enabled in unsupported:
        if enabled:
            raise ValueError(f"VPP does not support {feature} yet")
    if cfg.disaggregation_mode != "null":
        if cfg.disaggregation_transfer_backend not in ("mooncake", "nixl"):
            raise ValueError("VPP PD requires Mooncake or NIXL")
        if cfg.optimistic_prefill_attempts or (
            cfg.disaggregation_decode_allocation_policy != "early"
        ):
            raise ValueError(
                "VPP PD requires early allocation without optimistic prefill"
            )
        if (
            cfg.disaggregation_decode_enable_offload_kvcache
            or cfg.enable_pd_role_switch
        ):
            raise ValueError(
                "VPP PD does not support KV offload or runtime role switch yet"
            )
    if cfg.scheduler_recv_interval > 1:
        raise ValueError("VPP requires one input poll per iteration")
    if os.getenv("SGLANG_PP_LAYER_PARTITION") is not None:
        raise ValueError(
            "VPP uses --pp-vpp-layer-partition; unset SGLANG_PP_LAYER_PARTITION"
        )
    partition = cfg.pp_vpp_layer_partition
    if partition is not None and (
        len(partition) != cfg.pp_size * cfg.pp_virtual_stages
        or any(type(count) is not int or count <= 0 for count in partition)
    ):
        raise ValueError(
            "VPP partition requires one positive layer count per logical stage"
        )


def handle_vpp(server_args: Any) -> None:
    """Disable whole-model capture before graph resolution and validate depth."""
    cfg = resolving_view(server_args)
    if cfg.pp_virtual_stages == 1:
        return
    run_hook(validate_vpp_options, server_args)
    resolve_vpp_layout(model_config_of(server_args), cfg)
    raw_graph = cfg.cuda_graph_config
    if isinstance(raw_graph, CudaGraphConfig):
        raw_graph = raw_graph.to_dict()
    for phase, explicit_backend in (
        (Phase.DECODE, cfg.cuda_graph_backend_decode),
        (Phase.PREFILL, cfg.cuda_graph_backend_prefill),
    ):
        backend = (raw_graph or {}).get(phase, {}).get("backend", explicit_backend)
        if backend not in (None, Backend.DISABLED):
            raise ValueError(
                f"VPP requires the {phase} CUDA graph backend to be disabled"
            )
    declare_resolution(
        server_args,
        "handle_vpp",
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        enable_decoder_swa_bounded_replay=False,
    )
