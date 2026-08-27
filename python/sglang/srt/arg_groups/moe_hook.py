# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the MoE kernel configuration."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import (
    resolved_view,
    resolving_view,
)
from sglang.srt.environ import envs

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
