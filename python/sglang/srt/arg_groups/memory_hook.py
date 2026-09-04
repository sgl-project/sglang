# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the GPU memory budget."""

from __future__ import annotations

import copy
import logging
from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    post_capture_kv_sizing_planned,
    resolved_view,
    resolving_view,
    use_mla_backend,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

logger = logging.getLogger(__name__)

_DEFAULT_PP_PREFILL_CUDA_GRAPH_MAX_TOKENS = 8192


def handle_gpu_memory_settings(server_args: Any, gpu_mem):
    """
    Configure GPU memory-dependent settings including
    chunked_prefill_size, cuda_graph_config[decode].max_bs, and mem_fraction_static.

    Here are our heuristics:
    - Set chunked_prefill_size and cuda_graph_config[decode].max_bs based on the GPU memory capacity.
      This is because GPUs with more memory are generally more powerful, we need to use a larger
      chunked_prefill_size and a larger decode max_bs to fully utilize the GPU.
    - Then set mem_fraction_static based on chunked_prefill_size and decode max_bs.

      GPU memory capacity = model weights + KV cache pool + activations + cuda graph buffers

      The argument mem_fraction_static is defined as (model weights + KV cache pool) / GPU memory capacity,
      or equivalently, mem_fraction_static = (GPU memory capacity - activations - cuda graph buffers) / GPU memory capacity.

      In order to compute mem_fraction_static, we need to estimate the size of activations and cuda graph buffers.
      The activation memory is proportional to the chunked_prefill_size.
      The cuda graph memory is proportional to the decode max_bs.
      We use reserved_mem = chunked_prefill_size * 1.5 + max_bs * 2 to estimate the size of activations and cuda graph buffers in GB,
      and set mem_fraction_static = (GPU memory capacity - reserved_mem) / GPU memory capacity.

      The coefficient 1.5 is a heuristic value, in the future, we can do better estimation by looking at the model types, hidden sizes or even do a dummy run.
    """
    from sglang.srt.arg_groups.cuda_graph_hook import (
        generate_cpu_graph_batch_sizes,
        generate_decode_cuda_graph_batch_sizes,
        generate_prefill_cuda_graph_batch_sizes,
    )

    cfg = resolving_view(server_args)
    # A copy, so an earlier declaration keeps the value it recorded.
    cuda_graph_config = copy.deepcopy(cfg.cuda_graph_config)
    decode_cuda_graph_config = cuda_graph_config.decode
    prefill_cuda_graph_config = cuda_graph_config.prefill

    # ------------------------------------------------------------------
    # GPU-dependent capacity defaults
    # ------------------------------------------------------------------

    if gpu_mem is not None:
        if gpu_mem < 20 * 1024:
            # T4, 4080
            # (chunked_prefill_size 2k, max_bs 8)
            if cfg.chunked_prefill_size is None:
                declare_resolution(
                    server_args,
                    "_handle_gpu_memory_settings",
                    chunked_prefill_size=2048,
                )
            if decode_cuda_graph_config.max_bs is None:
                decode_cuda_graph_config.max_bs = 8
        elif gpu_mem < 35 * 1024:
            # A10, 4090, 5090
            # (chunked_prefill_size 4k, max_bs 48 if tp < 4 else 160)
            # 32GB Blackwell (RTX 5090) can hold decode cuda graphs well past
            # bs=24; the previous cap forced eager decode at bs>=32 and
            # collapsed high-concurrency throughput vs vLLM.
            if cfg.chunked_prefill_size is None:
                declare_resolution(
                    server_args,
                    "_handle_gpu_memory_settings",
                    chunked_prefill_size=4096,
                )
            if decode_cuda_graph_config.max_bs is None:
                if cfg.tp_size < 4:
                    decode_cuda_graph_config.max_bs = 48
                else:
                    decode_cuda_graph_config.max_bs = 160
        elif gpu_mem < 60 * 1024:
            # A100 (40GB), L40,
            # (chunked_prefill_size 4k, max_bs 32 if tp < 4 else 160)
            if cfg.chunked_prefill_size is None:
                declare_resolution(
                    server_args,
                    "_handle_gpu_memory_settings",
                    chunked_prefill_size=4096,
                )
            if decode_cuda_graph_config.max_bs is None:
                if cfg.tp_size < 4:
                    decode_cuda_graph_config.max_bs = 32
                else:
                    decode_cuda_graph_config.max_bs = 160
        elif gpu_mem < 90 * 1024:
            # H100, A100
            # (chunked_prefill_size 8k, max_bs 256 if tp < 4 else 512)
            if cfg.chunked_prefill_size is None:
                declare_resolution(
                    server_args,
                    "_handle_gpu_memory_settings",
                    chunked_prefill_size=8192,
                )
            if decode_cuda_graph_config.max_bs is None:
                if cfg.tp_size < 4:
                    decode_cuda_graph_config.max_bs = 256
                else:
                    decode_cuda_graph_config.max_bs = 512
        elif gpu_mem < 160 * 1024:
            # H20, H200
            # (chunked_prefill_size 8k, max_bs 256 if tp < 4 else 512)
            if cfg.chunked_prefill_size is None:
                declare_resolution(
                    server_args,
                    "_handle_gpu_memory_settings",
                    chunked_prefill_size=8192,
                )
            if decode_cuda_graph_config.max_bs is None:
                if cfg.tp_size < 4:
                    decode_cuda_graph_config.max_bs = 256
                else:
                    decode_cuda_graph_config.max_bs = 512
        else:
            # B200, MI300
            # (chunked_prefill_size 16k, max_bs 512)
            if cfg.chunked_prefill_size is None:
                declare_resolution(
                    server_args,
                    "_handle_gpu_memory_settings",
                    chunked_prefill_size=16384,
                )
            if decode_cuda_graph_config.max_bs is None:
                decode_cuda_graph_config.max_bs = 512
    else:
        # Fallback defaults when gpu_mem is None
        if cfg.chunked_prefill_size is None:
            declare_resolution(
                server_args,
                "_handle_gpu_memory_settings",
                chunked_prefill_size=4096,
            )
        if decode_cuda_graph_config.max_bs is None:
            decode_cuda_graph_config.max_bs = 160

    # ------------------------------------------------------------------
    # CUDA graph batch-size materialization
    # ------------------------------------------------------------------

    if cfg.device != "cpu":
        if decode_cuda_graph_config.bs is None:
            decode_cuda_graph_config.bs = generate_decode_cuda_graph_batch_sizes(
                server_args, decode_cuda_graph_config.max_bs
            )
        else:
            decode_cuda_graph_config.max_bs = max(decode_cuda_graph_config.bs)
    else:
        # Reuse decode_cuda_graph_config.bs for cpu graph and use torch_compile_max_bs for cpu graph batch size limit,
        # as cpu graph is based on torch.compile
        if decode_cuda_graph_config.bs is not None:
            declare_resolution(
                server_args,
                "_handle_gpu_memory_settings",
                torch_compile_max_bs=max(decode_cuda_graph_config.bs),
            )
        else:
            # If decode_cuda_graph_config.bs is not set, we will preferentially use torch_compile_max_bs
            # to generate decode_cuda_graph_config.bs
            declare_resolution(
                server_args,
                "_handle_gpu_memory_settings",
                torch_compile_max_bs=cfg.torch_compile_max_bs
                or decode_cuda_graph_config.max_bs,
            )
            decode_cuda_graph_config.bs = generate_cpu_graph_batch_sizes(server_args)

        assert cfg.torch_compile_max_bs > 0, (
            "cuda_graph_config[decode].bs should contain positive batch sizes"
        )
        decode_cuda_graph_config.max_bs = cfg.torch_compile_max_bs

    if prefill_cuda_graph_config.max_bs is None:
        # Refer to pr #15927, by default we set the prefill max_bs to the chunked prefill size.
        # For MLA backend, the introduction of piecewise cuda graph will influence the kernel dispatch difference compared to the original mode.
        # To avoid the performance regression, we set max_bs to 2048 by default.
        if not use_mla_backend(server_args):
            prefill_cuda_graph_config.max_bs = cfg.chunked_prefill_size
        else:
            prefill_cuda_graph_config.max_bs = 2048

        # For opt-in PP breakable graphs, capture small aggregate-token
        # buckets by default and leave larger forwards on the eager path.
        # Explicit max_bs or bs settings retain their existing semantics.
        if (
            cfg.pp_size > 1
            and prefill_cuda_graph_config.backend == Backend.BREAKABLE
            and (Phase.PREFILL, "bs") not in server_args._cuda_graph_config_locked
            and prefill_cuda_graph_config.max_bs
            > _DEFAULT_PP_PREFILL_CUDA_GRAPH_MAX_TOKENS
        ):
            prefill_cuda_graph_config.max_bs = _DEFAULT_PP_PREFILL_CUDA_GRAPH_MAX_TOKENS

        # If max_total_tokens is set, cap prefill max_bs to not exceed max_total_tokens.
        if cfg.max_total_tokens is not None:
            prefill_cuda_graph_config.max_bs = min(
                prefill_cuda_graph_config.max_bs, cfg.max_total_tokens
            )

        # For Llama2 series models, max_bs is limited to 4096.
        # TODO(yuwei): remove this after the issue is fixed
        if "llama-2" in cfg.model_path.lower():
            prefill_cuda_graph_config.max_bs = min(
                prefill_cuda_graph_config.max_bs, 4096
            )

    if prefill_cuda_graph_config.bs is None:
        prefill_cuda_graph_config.bs = generate_prefill_cuda_graph_batch_sizes(
            prefill_cuda_graph_config.max_bs
        )

    if cuda_graph_config != cfg.cuda_graph_config:
        declare_resolution(
            server_args,
            "_handle_gpu_memory_settings",
            cuda_graph_config=cuda_graph_config,
        )

    # ------------------------------------------------------------------
    # Static memory and runtime headroom
    # ------------------------------------------------------------------

    if cfg.mem_fraction_static is None:
        model_config = model_config_of(server_args)
        is_vlm = (
            model_config.is_multimodal
            and not cfg.language_only
            and not cfg.language_model_only
            and cfg.disaggregation_mode != "decode"
        )
        post_capture_kv_sizing = post_capture_kv_sizing_planned(server_args)

        if post_capture_kv_sizing:
            # Post-capture sizing measures free memory after graph capture, so
            # skip the graph/activation reserve; keep only the floor + parallel slack.
            reserved_mem = 1536
            reserved_mem += cfg.tp_size * cfg.pp_size / 8 * 1024
        else:
            # Tokens the activation working set scales with (per serving mode).
            if cfg.disaggregation_mode == "decode":
                running_requests = (
                    cfg.max_running_requests or decode_cuda_graph_config.max_bs or 1
                )
                draft_tokens = cfg.speculative_num_draft_tokens or 1
                activation_tokens = max(running_requests * draft_tokens, 2048)
            elif cfg.chunked_prefill_size > 0:
                activation_tokens = max(cfg.chunked_prefill_size, 2048)
            else:
                activation_tokens = max(cfg.max_prefill_tokens, 2048)
            # Constant meta data (e.g., from attention backend) + activation slack.
            reserved_mem = 512
            reserved_mem += activation_tokens * 1.5
            # Some adjustments for large parallel size
            reserved_mem += cfg.tp_size * cfg.pp_size / 8 * 1024
            reserved_mem += reserve_for_graph_mb(server_args)
            if gpu_mem is not None and gpu_mem > 60 * 1024:
                reserved_mem = max(reserved_mem, 10 * 1024)
            # Reserve headroom for DeepEP all-to-all buffers on top of the floor.
            reserved_mem += reserve_for_deepep_a2a_mb(server_args)

        mem_fraction_static = (
            round((gpu_mem - reserved_mem) / gpu_mem, 3)
            if gpu_mem is not None
            else 0.95
        )

        # Multimodal models need more memory for the image processing.
        if is_vlm:
            mem_fraction_static = adjust_mem_fraction_for_vlm(
                mem_fraction_static,
                model_config,
                post_capture_kv_sizing,
                gpu_mem,
            )

        declare_resolution(
            server_args,
            "_handle_gpu_memory_settings",
            mem_fraction_static=mem_fraction_static,
        )

    # ------------------------------------------------------------------
    # Symmetric-memory preallocation
    # ------------------------------------------------------------------

    if cfg.enable_symm_mem and not envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.is_set():
        envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.set(4)
        logger.warning(
            "Symmetric memory is enabled, setting symmetric memory prealloc size to 4GB as default."
            "Use environment variable SGLANG_SYMM_MEM_PREALLOC_GB_SIZE to change the prealloc size."
        )


def reserve_for_graph_mb(server_args: Any) -> float:

    cfg = resolving_view(server_args)
    decode_cuda_graph_config = cfg.cuda_graph_config.decode
    prefill_cuda_graph_config = cfg.cuda_graph_config.prefill

    reserved_mem = 0.0
    if (
        cfg.disaggregation_mode != "prefill"
        and decode_cuda_graph_config.backend != Backend.DISABLED
    ):
        reserved_mem += decode_cuda_graph_config.max_bs * 2

    if (
        resolved_view(server_args).enable_dp_attention
        and cfg.disaggregation_mode != "prefill"
    ):
        # DP attention needs more padding for some operations, and much more for large
        # cuda graph max bs (torch allocator / implementation inefficiencies).
        reserved_mem += decode_cuda_graph_config.max_bs * cfg.dp_size * 3
        if decode_cuda_graph_config.max_bs > 300:
            reserved_mem += decode_cuda_graph_config.max_bs * cfg.dp_size * 1.5

    if (
        cfg.disaggregation_mode != "decode"
        and prefill_cuda_graph_config.backend != Backend.DISABLED
    ):
        if not use_mla_backend(server_args):
            # Only non-torch memory is counted; torch memory is reused by cuda graph capture.
            reserved_mem += len(prefill_cuda_graph_config.bs) * 8
        else:
            # MLA backend overhead is much higher than expected with fa3.
            reserved_mem += 1.5 * 1024

        if (
            prefill_cuda_graph_config.backend == Backend.BREAKABLE
            and resolved_view(server_args).moe_a2a_backend == "deepep"
        ):
            # Prefill-BCG DeepEP delta (bridge pool + NVL first-touch
            # during capture); decode-side DeepEP is a baseline cost.
            reserved_mem += 1 * 1024

    return reserved_mem


def reserve_for_deepep_a2a_mb(server_args: Any) -> float:
    # DeepEP all-to-all buffers captured in the decode graph are real extra
    # allocations, reserved on top of the floor.

    cfg = resolving_view(server_args)
    decode_cuda_graph_config = cfg.cuda_graph_config.decode
    if (
        cfg.disaggregation_mode != "prefill"
        and decode_cuda_graph_config.backend != Backend.DISABLED
        and resolved_view(server_args).moe_a2a_backend == "deepep"
    ):
        return 2 * 1024
    return 0.0


def adjust_mem_fraction_for_vlm(
    mem_fraction_static: float,
    model_config,
    post_capture_kv_sizing: bool,
    gpu_mem,
) -> float:
    if post_capture_kv_sizing:
        # Graph and activation memory will be measured after capture, so only
        # reserve a fixed 8 GiB of VLM headroom for image-encoder execution here.
        return (
            mem_fraction_static - 8 * 1024 / gpu_mem
            if gpu_mem is not None
            else mem_fraction_static
        )

    vision_config = getattr(model_config.hf_config, "vision_config", None)
    if vision_config is None:
        return mem_fraction_static

    # roughly reduce the mem_fraction_static base on params of Vit
    # a base mem_fraction_static factor for regular Vit
    base_mem_fraction_reduction_ratio = 0.95

    vit_num_layers = getattr(vision_config, "num_hidden_layers", 24)
    vit_hidden_size = getattr(vision_config, "hidden_size", 1024)

    # baseline ViT params (ViT-L/14)
    baseline_vit_layers = 24
    baseline_vit_hidden_size = 1024

    # weight params count
    current_complexity_score = vit_num_layers * (vit_hidden_size**2)
    baseline_complexity_score = baseline_vit_layers * (baseline_vit_hidden_size**2)
    complexity_ratio = (
        current_complexity_score / baseline_complexity_score
        if baseline_complexity_score > 0
        else 1.0
    )

    # every time the complexity grows 100%, adjust final factor for 10%
    sensitivity_scale = 0.1
    dynamic_adjustment_factor = 1.0 - sensitivity_scale * (complexity_ratio - 1.0)
    dynamic_adjustment_factor = max(0.8, min(1.05, dynamic_adjustment_factor))

    final_overall_factor = base_mem_fraction_reduction_ratio * dynamic_adjustment_factor
    return mem_fraction_static * final_overall_factor
