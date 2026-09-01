from __future__ import annotations

import logging
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
