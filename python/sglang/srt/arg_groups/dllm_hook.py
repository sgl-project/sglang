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
    resolved_view,
    resolving_view,
    run_post_process_pass,
)
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


def handle_dllm_cuda_graph_compatibility(server_args: Any):
    """Settle which dLLM CUDA graphs can run, before memory sizing.

    Both gates must land before `handle_gpu_memory_settings`: a phase disabled
    here contributes nothing to `reserve_for_graph_mb`, and the prefill bucket
    list is generated there too.
    """
    cfg = resolving_view(server_args)
    if cfg.dllm_algorithm is None:
        return
    if get_platform().is_hip:
        if (
            cfg.cuda_graph_config.decode.backend != Backend.DISABLED
            or cfg.cuda_graph_config.prefill.backend != Backend.DISABLED
        ):
            logger.warning(
                "Cuda graph is disabled for diffusion LLM inference on AMD GPUs"
            )
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
        return

    _disable_dllm_prefill_graph_on_deepep_misalignment(server_args)


def _disable_dllm_prefill_graph_on_deepep_misalignment(server_args: Any) -> None:
    """Drop the dLLM prefill graph when its buckets cannot satisfy DeepEP.

    DeepEP a2a can hang capturing a breakable prefill bucket that is not a
    multiple of 8, while pure dLLM prefill can only replay a bucket that is a
    multiple of `lcm(page_size, block_size)`. When that alignment is not itself
    a multiple of 8 the two constraints have no common bucket, so keep the
    prefill forward eager instead of capturing something unreplayable.
    """
    from sglang.srt.arg_groups.memory_hook import dllm_prefill_graph_alignment

    cfg = resolving_view(server_args)
    if resolved_view(server_args).moe_a2a_backend != "deepep":
        return
    alignment = dllm_prefill_graph_alignment(server_args)
    if alignment is None or alignment % 8 == 0:
        return

    logger.warning(
        "Disabling the dLLM prefill CUDA graph: DeepEP a2a requires bucket "
        "sizes divisible by 8, but pure dLLM prefill is aligned to %d.",
        alignment,
    )
    declare_resolution(
        server_args,
        "_handle_dllm_cuda_graph_compatibility",
        cuda_graph_config=with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        ),
    )


def _reconcile_dllm_prefill_graph_buckets(server_args: Any) -> None:
    """Rebuild the prefill buckets if the page size moved after memory sizing.

    Memory sizing predicts the page size with `resolve_default_page_size`,
    because `_page_size_default` and the backend page constraints have not run
    at its slot. Every pass that can move it afterwards caps at the dLLM block
    size, so the alignment normally lands identically -- but nothing enforces
    that, and a mismatch would capture buckets the scheduler can never emit,
    silently keeping every pure dLLM prefill eager.

    Capture correctness wins over the reserve: `reserve_for_graph_mb` already
    consumed the earlier list, but that reserve is a heuristic for
    `mem_fraction_static`, whereas a bucket list the runner cannot match is
    simply dead capture.
    """
    from sglang.srt.arg_groups.memory_hook import (
        generate_dllm_prefill_cuda_graph_batch_sizes,
    )

    cfg = resolving_view(server_args)
    locked = getattr(server_args, "_cuda_graph_config_locked", set())
    if (Phase.PREFILL, "bs") in locked:
        return

    prefill_config = cfg.cuda_graph_config.prefill
    final_bs = generate_dllm_prefill_cuda_graph_batch_sizes(
        server_args, prefill_config.max_bs, quiet=True
    )
    if final_bs is None or final_bs == list(prefill_config.bs or []):
        return

    logger.warning(
        "dLLM prefill graph buckets were sized against a different page size "
        "than the resolution settled on; replacing %d captured buckets with "
        "%d. The graph reserve in mem_fraction_static was computed from the "
        "former.",
        len(prefill_config.bs or []),
        len(final_bs),
    )
    declare_resolution(
        server_args,
        "_handle_dllm_inference",
        cuda_graph_config=with_phase(cfg.cuda_graph_config, Phase.PREFILL, bs=final_bs),
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
    _reconcile_dllm_prefill_graph_buckets(server_args)

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
