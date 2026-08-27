# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the MoE kernel configuration."""

from __future__ import annotations

import logging
import os
from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolved_view,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.utils.common import is_npu

logger = logging.getLogger(__name__)


def handle_moe_kernel_config(server_args: Any):
    # The quantization-driven runner resolutions moved to the pipeline
    # (arg_groups/overrides.py: _moe_runner_backend_quant_constraints);
    # the compatibility asserts and fusion writes stay below.
    cfg = resolving_view(server_args)
    from sglang.srt.arg_groups.overrides import (
        _moe_runner_backend_quant_constraints,
        _moe_runner_fusion_disable,
        run_post_process_pass,
    )

    run_post_process_pass(server_args, _moe_runner_backend_quant_constraints)

    view = resolved_view(server_args)
    if view.moe_runner_backend == "flashinfer_cutlass":
        assert view.quantization in [
            "modelopt_fp4",
            "modelopt_fp8",
            "modelopt_mixed",
            None,
        ], f"Invalid quantization '{view.quantization}'. \nFlashInfer Cutlass MOE supports only: 'modelopt_fp4', 'modelopt_fp8', 'modelopt_mixed', or bfloat16 (None)."
        assert view.ep_size in [
            1,
            cfg.tp_size,
        ], "The expert parallel size must be 1 or the same as the tensor parallel size"

    if view.moe_runner_backend == "flashinfer_cutedsl":
        # modelopt_mixed with non-NVFP4 MoE layers is rejected at load time.
        assert (
            view.quantization in ["modelopt_fp4", "modelopt_mixed", "nvfp4_online"]
            or server_args.get_model_config().nvfp4_moe_meta is not None
        ), f"Invalid quantization '{view.quantization}'. \nFlashInfer CuteDSL MOE currently supports only: 'modelopt_fp4', 'modelopt_mixed' (with NVFP4 MoE layers), 'nvfp4_online', or hybrid NVFP4 models."
        assert view.ep_size in [
            1,
            cfg.tp_size,
        ], "The expert parallel size must be 1 or the same as the tensor parallel size"
        assert view.moe_a2a_backend in [
            "none",
            "deepep",
            "flashinfer",
        ], (
            f"flashinfer_cutedsl supports moe_a2a_backend='none', 'deepep', or 'flashinfer', "
            f"got '{view.moe_a2a_backend}'."
        )
        if view.moe_a2a_backend == "deepep" and (
            view.quantization == "nvfp4_online"
            or envs.SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.get()
        ):
            raise ValueError(
                "flashinfer_cutedsl per-token NVFP4 activation requires "
                "moe_a2a_backend='none' or 'flashinfer'."
            )

    if view.moe_runner_backend in ["flashinfer_trtllm", "experimental_sgl_trtllm"]:
        assert view.quantization in [
            "modelopt_fp4",
            "nvfp4_online",
            "fp8",
            "mxfp8",
            "modelopt_fp8",
            "modelopt_mixed",
            "compressed-tensors",
            None,
        ], f"Invalid quantization '{view.quantization}'. \nFlashInfer TRTLLM MOE supports only: 'modelopt_fp4', 'nvfp4_online', 'fp8', 'modelopt_fp8', 'modelopt_mixed', 'compressed-tensors', or bfloat16 (None)."

    if view.moe_runner_backend == "flashinfer_trtllm_routed":
        assert view.quantization in [
            "fp8",
            "mxfp8",
            "modelopt_fp4",
            "modelopt_mixed",
            "nvfp4_online",
            None,
        ], f"Invalid quantization '{view.quantization}'. \nFlashInfer TRTLLM routed MOE supports only: 'fp8', 'mxfp8', 'modelopt_fp4', 'modelopt_mixed', 'nvfp4_online', or bfloat16 (None)."

    # The runner-driven shared-experts fusion disables moved to the
    # pipeline (arg_groups/overrides.py: _moe_runner_fusion_disable),
    # invoked here at the legacy write slots.
    run_post_process_pass(server_args, _moe_runner_fusion_disable)

    if resolved_view(server_args).moe_runner_backend == "cutlass" and resolved_view(
        server_args
    ).quantization in [
        "fp8",
        "mxfp8",
    ]:
        assert (
            resolved_view(server_args).ep_size == 1
        ), "FP8/MXFP8 Cutlass MoE is only supported with ep_size == 1"


def handle_a2a_moe(server_args: Any):
    # The backend overrides and the ep_size=tp_size adjustments moved to
    # the resolution pipeline (arg_groups/overrides.py:
    # _a2a_backend_overrides / _a2a_ep_size); the per-backend logs,
    # asserts, fusion/deepep_mode/env/cuda-graph writes stay below.
    cfg = resolving_view(server_args)
    from sglang.srt.arg_groups.overrides import (
        _a2a_backend_overrides,
        _a2a_ep_size,
        _a2a_fusion_adjustments,
        run_post_process_pass,
    )

    run_post_process_pass(server_args, _a2a_backend_overrides)
    run_post_process_pass(server_args, _a2a_ep_size)

    # The a2a-driven shared-experts fusion adjustments moved to the
    # pipeline (arg_groups/overrides.py: _a2a_fusion_adjustments),
    # invoked here at the legacy write slots.
    run_post_process_pass(server_args, _a2a_fusion_adjustments)

    a2a_backend = resolved_view(server_args).moe_a2a_backend
    if cfg.enable_waterfill:
        declare_resolution(
            server_args, "_handle_a2a_moe", enforce_shared_experts_fusion=True
        )
        logger.info(f"Waterfill is enabled with moe_a2a_backend='{a2a_backend}'.")

    if a2a_backend == "deepep":
        if cfg.moe_runner_backend == "flashinfer_cutedsl":
            if cfg.deepep_mode == "auto":
                declare_resolution(
                    server_args,
                    "_handle_a2a_moe",
                    deepep_mode="low_latency",
                )
                logger.warning(
                    "Forcing --deepep-mode low_latency: flashinfer_cutedsl "
                    "FP4 MoE has no DeepEP normal-dispatch handler, so "
                    "deepep auto mode would crash during prefill. "
                    "low_latency covers both prefill and decode."
                )
            elif cfg.deepep_mode == "normal":
                raise ValueError(
                    "flashinfer_cutedsl FP4 MoE only supports DeepEP "
                    "low_latency dispatch (masked layout). DeepEP normal "
                    "(prefill) dispatch has no CuteDSL FP4 handler. Pass "
                    "--deepep-mode low_latency or auto."
                )
        if cfg.deepep_mode == "normal":
            logger.warning("Cuda graph is disabled because deepep_mode=`normal`")
            declare_resolution(
                server_args,
                "_handle_a2a_moe",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
                ),
            )
            declare_resolution(
                server_args,
                "_handle_a2a_moe",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
                ),
            )

    if a2a_backend == "deepep_v2":
        server_args._validate_deepep_v2_model_architecture()
        if resolved_view(server_args).enable_deterministic_inference:
            raise ValueError(
                "DeepEP v2 does not forward deterministic=True to "
                "ElasticBuffer, so deterministic sorting remains disabled. "
                "Disable --enable-deterministic-inference or use "
                "--moe-a2a-backend deepep."
            )
        # ElasticBuffer requires CUMEM, but not NVLS or its preallocation.
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
        # Respect model-level runner declarations before resolving auto.
        resolved_runner = resolved_view(server_args).moe_runner_backend
        if resolved_runner == "auto":
            declare_resolution(
                server_args, "_handle_a2a_moe", moe_runner_backend="deep_gemm"
            )
            logger.warning(
                "DeepEP v2 MoE: resolved --moe-runner-backend auto -> deep_gemm."
            )
        elif resolved_runner != "deep_gemm":
            raise ValueError(
                "DeepEP v2 MoE currently supports only "
                f"--moe-runner-backend deep_gemm. Got {resolved_runner!r}. "
                "Add a runner adapter before enabling DeepEP v2 with other "
                "MoE runners."
            )
        if cfg.enable_two_batch_overlap or cfg.enable_single_batch_overlap:
            raise ValueError(
                "DeepEP v2 MoE has not implemented the TBO/SBO overlap hooks yet. "
                "Disable --enable-two-batch-overlap and "
                "--enable-single-batch-overlap when using --moe-a2a-backend deepep_v2."
            )
        if cfg.enforce_shared_experts_fusion:
            raise ValueError(
                "DeepEP v2 MoE has not validated fused shared experts yet. "
                "Remove --enforce-shared-experts-fusion when using "
                "--moe-a2a-backend deepep_v2."
            )
        # Prefill reads host counts and is not graph-capturable.
        declare_resolution(
            server_args,
            "_handle_a2a_moe",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
            ),
        )
        logger.warning(
            f"DeepEP v2 MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{cfg.tp_size}]."
        )
        logger.warning(
            "DeepEP v2 MoE is using deepep_v2_mode=%s. This controls "
            "ElasticBuffer direct/hybrid mode and is independent from "
            "--deepep-mode normal/low_latency. DeepEP v2 MoE enables the "
            "decode CUDA graph on the masked decode path (any comm mode) "
            "and disables shared expert fusion. "
            "SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK is a "
            "per-rank communication buffer capacity, not a model limit; "
            "increase it for large prefill/chunked-prefill workloads.",
            cfg.deepep_v2_mode,
        )

    # The resolving view, not the field: `_a2a_backend_overrides` may have
    # moved this already (waterfill forces `deepep`).
    a2a_now = resolved_view(server_args).moe_a2a_backend
    if (a2a_now == "none" and is_npu()) or a2a_now == "ascend_tp":
        # FIXME (OrangeRedeng): for some reasons if pass "ascend_tp" accuracy drops to zero
        declare_resolution(
            server_args,
            "_handle_a2a_moe",
            moe_a2a_backend="none",
        )

    if cfg.moe_a2a_backend == "flashinfer":
        assert (
            resolved_view(server_args).enable_dp_attention
            and cfg.dp_size == cfg.tp_size
        ), "Flashinfer MoE A2A is only supported with dp_size == tp_size and --enable-dp-attention"
        if cfg.deepep_mode != "auto":
            logger.warning("--deepep-mode is ignored for Flashinfer MoE A2A")
        if not envs.SGLANG_MOE_NVFP4_DISPATCH.is_set() and (
            resolved_view(server_args).quantization == "modelopt_fp4"
            or server_args.get_model_config().nvfp4_moe_meta is not None
        ):
            envs.SGLANG_MOE_NVFP4_DISPATCH.set(True)
            logger.warning(
                "SGLANG_MOE_NVFP4_DISPATCH is set to True for Flashinfer MoE A2A"
            )
        assert resolved_view(server_args).moe_runner_backend in [
            "flashinfer_cutlass",
            "flashinfer_cutedsl",
            "flashinfer_trtllm_routed",
        ], "Flashinfer MoE A2A is only supported with flashinfer_cutlass, flashinfer_cutedsl or flashinfer_trtllm_routed moe runner backend"

    if a2a_backend == "mori":
        if cfg.deepep_mode == "auto":
            declare_resolution(
                server_args,
                "_handle_a2a_moe",
                deepep_mode="normal",
            )
            logger.warning("auto set deepep_mode=`normal` for MORI EP")

        # Check chunked prefill for mori
        # Skip validation if chunked prefill is disabled (i.e., size <= 0).
        # Skip validation if disaggregation mode is decode.
        if cfg.chunked_prefill_size > 0 and cfg.disaggregation_mode != "decode":
            assert (
                server_args._required_mori_dispatch_tokens_per_rank()
            ) <= envs.SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get(), (
                "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK (default 4096) "
                "must be >= the per-rank MoRI dispatch tokens "
                "(chunked_prefill_size by default)"
            )

    if a2a_backend == "pplx":
        if cfg.deepep_mode == "normal":
            raise ValueError(
                "moe_a2a_backend='pplx' only supports low-latency mode; "
                "set --deepep-mode to 'low_latency' or 'auto'."
            )
        if cfg.deepep_mode == "auto":
            declare_resolution(
                server_args,
                "_handle_a2a_moe",
                deepep_mode="low_latency",
            )
            logger.warning("auto set deepep_mode=`low_latency` for PPLX EP")
        # pplx-kernels' AllToAll needs numDPGroups (== attention dp_size) > 1;
        # without DP attention numDPGroups == 1 and construction fails deep in
        # the kernel. This also implies ep_size >= 2.
        assert resolved_view(server_args).enable_dp_attention and cfg.dp_size >= 2, (
            "moe_a2a_backend='pplx' requires --enable-dp-attention with at "
            "least 2 DP groups (--dp-size >= 2)."
        )
        # pplx runs the masked DeepGEMM expert path (sm_90a): reject other
        # runners and resolve auto -> deep_gemm. Unquantized bf16 pplx needs
        # an explicit deep_gemm backend, otherwise the expert layer falls
        # through to the deprecated masked path and asserts at runtime.
        assert resolved_view(server_args).moe_runner_backend in ("deep_gemm", "auto"), (
            "moe_a2a_backend='pplx' is only supported with --moe-runner-backend "
            "deep_gemm (or auto)."
        )
        if cfg.moe_runner_backend == "auto":
            declare_resolution(
                server_args,
                "_handle_a2a_moe",
                moe_runner_backend="deep_gemm",
            )
            logger.warning("auto set moe_runner_backend=`deep_gemm` for PPLX EP")

        # Check per-rank dispatch tokens for pplx
        # Skip validation if chunked prefill is disabled (i.e., size <= 0)
        # Skip validation if disaggregation mode is decode
        if cfg.chunked_prefill_size > 0 and cfg.disaggregation_mode != "decode":
            assert (
                server_args._required_pplx_dispatch_tokens_per_rank()
            ) <= envs.SGLANG_PPLX_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get(), (
                "SGLANG_PPLX_NUM_MAX_DISPATCH_TOKENS_PER_RANK (default 128) "
                "must be >= the per-rank pplx dispatch tokens "
                "(chunked_prefill_size, or the decode cuda-graph batch size)"
            )
