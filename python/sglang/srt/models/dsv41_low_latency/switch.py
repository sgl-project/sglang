# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""`--enable-dsv41-low-latency-decode`: the deployment it is validated on, and
the row bounds of its paths."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

# Row bounds per path, in tokens of one forward. Past a bound the stock kernel
# runs. FlashInfer's cutedsl_low_latency MXFP8 GEMM takes at most 8 rows.
MXFP8_MAX_ROWS = 8
# Compressor, indexer weights_proj and wk: slower than the stock GEMMs past 8 rows
# on GB300 (measured at 12 and 24 rows).
LOW_M_BF16_MAX_ROWS = 8
# Routed-expert GEMM pair; route_wide.cu takes 9-64 rows. 48 rows (DSpark verify
# at batch 8) was measured faster than FlashInfer at every row count.
ROUTED_EXPERT_MAX_ROWS = 48
# Gluon swap-AB sparse decode attention; FlashMLA is faster from 32 rows on GB300.
SWAPAB_MAX_ROWS = 24
# LM head on FlashInfer's CuTe DSL low-M BF16 GEMM; slower than cuBLAS at 24 rows.
LM_HEAD_MAX_ROWS = 16
# Rows of the wo_b mailbox (Lamport all-reduce + mHC): its decode overlap range.
WO_B_MAILBOX_MAX_TOKENS = 16

_GPU_SM = 103
_TP_SIZE = 4


def enabled() -> bool:
    from sglang.srt.runtime_context import get_exec

    return get_exec().features.enable_dsv41_low_latency_decode


def deployment_violations(server_args: ServerArgs, model_type: str) -> List[str]:
    """Every requirement of the switch that this deployment misses."""
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.runtime_context import attn_dp_enabled_of, get_platform

    cfg = resolving_view(server_args)
    platform = get_platform()
    checks = (
        ("a DeepSeek-V4.1 model", model_type == "deepseek_v41"),
        (
            f"an SM{_GPU_SM} GPU (GB300 / B300)",
            platform.is_cuda and platform.device_sm == _GPU_SM,
        ),
        (f"--tp-size {_TP_SIZE}", cfg.tp_size == _TP_SIZE),
        (
            "--ep-size 1 (the routed-expert GEMMs serve all experts locally)",
            cfg.ep_size == 1,
        ),
        ("--moe-a2a-backend none", cfg.moe_a2a_backend == "none"),
        ("no pipeline parallelism", cfg.pp_size == 1),
        ("no DP attention", not attn_dp_enabled_of(cfg)),
        ("no context parallelism", cfg.attn_cp_size == 1),
        ("one node", cfg.nnodes == 1),
        ("--speculative-algorithm DSPARK", cfg.speculative_algorithm == "DSPARK"),
        (
            "the default DeepSeek-V4 attention backend",
            cfg.dsv4_attn_backend in ("auto", "flashmla"),
        ),
        (
            "decode CUDA graphs",
            not cfg.disable_cuda_graph
            and cfg.cuda_graph_config.decode.backend != Backend.DISABLED,
        ),
        ("no encoder SWA bounded replay", not cfg.enable_encoder_swa_bounded_replay),
        ("no decoder SWA bounded replay", not cfg.enable_decoder_swa_bounded_replay),
        ("no two-batch overlap", not cfg.enable_two_batch_overlap),
        ("no PD disaggregation", cfg.disaggregation_mode == "null"),
        ("no LoRA", not cfg.enable_lora),
        # The low-M LM head returns BF16 logits through its quant method.
        ("no --enable-fp32-lm-head", not cfg.enable_fp32_lm_head),
        ("no --rl-on-policy-target", cfg.rl_on_policy_target is None),
        ("no deterministic inference", not cfg.enable_deterministic_inference),
    )
    return [requirement for requirement, met in checks if not met]


def check_deployment(server_args: ServerArgs, model_type: str) -> None:
    violations = deployment_violations(server_args, model_type)
    if violations:
        raise ValueError(
            "--enable-dsv41-low-latency-decode is validated on one deployment; "
            "this one needs: " + "; ".join(violations)
        )
