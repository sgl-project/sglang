# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for diffusion-LM inference."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import (
    _dllm_attention_backend,
    _dllm_overlap_disable,
    _dllm_page_size,
    declare_resolution,
    resolving_view,
    run_post_process_pass,
)
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


def handle_dllm_cuda_graph_compatibility(server_args: Any):
    """Disable CUDA graphs before memory sizing for dLLM on HIP."""
    cfg = resolving_view(server_args)
    if cfg.dllm_algorithm is None or not get_platform().is_hip:
        return
    if (
        cfg.cuda_graph_config.decode.backend != Backend.DISABLED
        or cfg.cuda_graph_config.prefill.backend != Backend.DISABLED
    ):
        logger.warning("Cuda graph is disabled for diffusion LLM inference on AMD GPUs")
        declare_resolution(
            server_args,
            "_handle_dllm_cuda_graph_compatibility",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
            ),
        )
        declare_resolution(
            server_args,
            "_handle_dllm_cuda_graph_compatibility",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
            ),
        )


def handle_dllm_inference(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.dllm_algorithm is None:
        return
    run_post_process_pass(server_args, _dllm_attention_backend)

    # Backend validation runs after the dLLM attention backend declaration,
    # while memory sizing intentionally happens earlier in the pipeline.
    from sglang.srt.arg_groups.overrides import attention_backends_of
    from sglang.srt.dllm.config import (
        DllmConfig,
        _validate_multi_block_prefill_backend,
    )

    dllm_config = DllmConfig.from_server_args(server_args)
    prefill_attention_backend, _ = attention_backends_of(resolving_view(server_args))
    _validate_multi_block_prefill_backend(
        block_size=dllm_config.block_size,
        prefill_block_size=dllm_config.prefill_block_size,
        prefill_attention_backend=prefill_attention_backend,
    )
    run_post_process_pass(server_args, _dllm_overlap_disable)

    # The page-size alignment + block-size cap for dllm moved to the
    # resolution pipeline (arg_groups/overrides.py: _dllm_page_size).
    # Invoked outside the radix gate: the alignment fill keeps its radix
    # gate inside the pass, the block-size cap applies regardless (it
    # replaces the unconditional scheduler-init fallback).

    run_post_process_pass(server_args, _dllm_page_size)

    if not cfg.disable_radix_cache:
        if cfg.enable_hierarchical_cache:
            logger.warning(
                "Hierarchical cache is disabled because of using diffusion LLM inference"
            )
            declare_resolution(
                server_args,
                "_handle_dllm_inference",
                enable_hierarchical_cache=False,
            )
        if cfg.enable_lmcache:
            logger.warning(
                "LMCache is disabled because of using diffusion LLM inference"
            )
            declare_resolution(
                server_args, "_handle_dllm_inference", enable_lmcache=False
            )
        if cfg.enable_flexkv:
            logger.warning(
                "FlexKV is disabled because of using diffusion LLM inference"
            )
            declare_resolution(
                server_args, "_handle_dllm_inference", enable_flexkv=False
            )

    if cfg.pp_size > 1:
        logger.warning(
            "Pipeline parallelism is disabled because of using diffusion LLM inference"
        )
        declare_resolution(
            server_args,
            "_handle_dllm_inference",
            pp_size=1,
        )

    if cfg.enable_lora:
        logger.warning("Currently LoRA is not supported by diffusion LLM inference.")
        declare_resolution(server_args, "_handle_dllm_inference", enable_lora=False)

    if cfg.disaggregation_mode != "null":
        logger.warning(
            "Currently disaggregation is not supported by diffusion LLM inference."
        )
        declare_resolution(
            server_args,
            "_handle_dllm_inference",
            disaggregation_mode="null",
        )

    if cfg.enable_mixed_chunk:
        logger.warning(
            "Mixed chunked prefill is disabled because of using diffusion LLM inference."
        )
        declare_resolution(
            server_args,
            "_handle_dllm_inference",
            enable_mixed_chunk=False,
        )
