from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def apply_kimi_k3_spec_backend_defaults(server_args: ServerArgs) -> None:
    """Apply speculative backend defaults for Kimi hybrid models."""
    cfg = resolving_view(server_args)

    if cfg.speculative_algorithm is None:
        return

    _apply_rocm_dspark_atom_kernels(server_args, cfg)

    # Use the fused Kimi-K3/DSPARK CuTeDSL kernel for KDA target verification.
    # Decode is left free (its bf16-ssm SM100+ flashinfer default is fine -- the
    # target only verifies under spec); the verify backend is pinned directly.
    if cfg.linear_attn_verify_backend is None:
        declare_resolution(
            server_args,
            "apply_kimi_k3_spec_backend_defaults",
            linear_attn_verify_backend="nv_cutedsl",
        )
        logger.info(
            "Kimi hybrid model with speculative decoding: pinning "
            "--linear-attn-verify-backend to nv_cutedsl (uses the fused "
            "Kimi-K3/DSPARK CuTeDSL kernel)."
        )

    # dspark's draft is dense MQA; trtllm_mha avoids flashinfer's blocking
    # per-step host plan. DSPARK-only: other spec algos use MLA-family drafts.
    if (
        cfg.speculative_algorithm == "DSPARK"
        and cfg.speculative_draft_attention_backend is None
        and get_platform().is_sm100
    ):
        declare_resolution(
            server_args,
            "apply_kimi_k3_spec_backend_defaults",
            speculative_draft_attention_backend="trtllm_mha",
        )
        logger.info(
            "Kimi hybrid DSPARK: defaulting "
            "--speculative-draft-attention-backend to trtllm_mha."
        )


def _apply_rocm_dspark_atom_kernels(server_args: ServerArgs, cfg) -> None:
    """Match ATOM's ROCm DSPARK kernels.

    ATOM's MoE is SiTU A4W4 (``gemm1_a4w4`` / ``gemm2_a4w4``). The recipe's
    ``AITER_SITUV2_A8W4=1`` and ``AITER_FLYDSL_FORCE=1`` select FlyDSL
    ``mfma_moe1`` / ``opus_moe_stage2`` instead. ATOM also keeps the draft on
    aiter; ``aiter`` is not a draft-backend choice, so the prefill backend is
    rejected and the draft falls back to the triton ``_verify_mla_prefix_stage1``
    kernel. Override both before workers load weights.
    """
    if not get_platform().is_hip or cfg.speculative_algorithm != "DSPARK":
        return
    os.environ["AITER_SITUV2_A4W4"] = "1"
    os.environ["AITER_SITUV2_A8W4"] = "0"
    os.environ["AITER_FLYDSL_STAGE2_FP8"] = "1"
    os.environ["AITER_FLYDSL_FORCE"] = "0"
    logger.info(
        "Kimi DSPARK on ROCm: MoE env set to ATOM A4W4 "
        "(AITER_SITUV2_A4W4=1, AITER_SITUV2_A8W4=0, AITER_FLYDSL_FORCE=0)."
    )
    if cfg.speculative_draft_attention_backend is None:
        declare_resolution(
            server_args,
            "apply_kimi_k3_spec_backend_defaults",
            speculative_draft_attention_backend="aiter",
        )
        logger.info(
            "Kimi DSPARK on ROCm: defaulting "
            "--speculative-draft-attention-backend to aiter."
        )


def apply_kimi_k3_linear_attn_defaults(server_args: ServerArgs) -> None:
    """KDA decode-fallback default for Kimi hybrid models (spec-independent)."""
    cfg = resolving_view(server_args)

    # Preempts the generic SM100+bf16 flashinfer switch (a GDN default): on
    # KDA shapes the triton packed decode measures ~35% faster than
    # recurrent_kda across bs 1-256, and ReplaySSM requires triton.
    if (
        cfg.linear_attn_decode_backend is None
        and cfg.mamba_ssm_dtype == "bfloat16"
        and get_platform().is_sm100
    ):
        declare_resolution(
            server_args,
            "apply_kimi_k3_linear_attn_defaults",
            linear_attn_decode_backend="triton",
        )
        logger.info(
            "Kimi hybrid model with bf16 SSM state: defaulting "
            "--linear-attn-decode-backend to triton."
        )
