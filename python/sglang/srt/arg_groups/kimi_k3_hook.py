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


def _uses_native_kimi_linear_unbounded_kda(
    server_args: ServerArgs, *, model_arch=None, hf_config=None
) -> bool:
    """Return whether every TP rank has the native equal-head/D128 contract."""
    if model_arch != "KimiLinearForCausalLM" or hf_config is None:
        return False
    linear_attn_config = getattr(hf_config, "linear_attn_config", None)
    if not isinstance(linear_attn_config, dict):
        return False
    num_heads = linear_attn_config.get("num_heads")
    head_dim = linear_attn_config.get("head_dim")
    tp_size = resolving_view(server_args).tp_size
    return (
        isinstance(num_heads, int)
        and isinstance(head_dim, int)
        and isinstance(tp_size, int)
        and tp_size > 0
        and num_heads > 0
        and num_heads % tp_size == 0
        and head_dim == 128
    )


def apply_kimi_k3_linear_attn_defaults(
    server_args: ServerArgs, *, model_arch=None, hf_config=None
) -> None:
    """Apply architecture-specific KDA defaults for Kimi hybrid models."""
    cfg = resolving_view(server_args)

    native_kimi_linear = _uses_native_kimi_linear_unbounded_kda(
        server_args, model_arch=model_arch, hf_config=hf_config
    )
    if native_kimi_linear and get_platform().is_sm100:
        changes = {}
        ssm_dtype = cfg.mamba_ssm_dtype
        if ssm_dtype is None:
            changes["mamba_ssm_dtype"] = "bfloat16"
            ssm_dtype = "bfloat16"
        if ssm_dtype != "bfloat16":
            return
        if cfg.linear_attn_decode_backend is None:
            changes["linear_attn_decode_backend"] = "cake"
        if cfg.linear_attn_prefill_backend is None:
            changes["linear_attn_prefill_backend"] = "cake"
        if changes:
            declare_resolution(
                server_args,
                "apply_kimi_k3_linear_attn_defaults",
                **changes,
            )
        if "mamba_ssm_dtype" in changes:
            logger.info(
                "Kimi-Linear equal-head/D128: defaulting --mamba-ssm-dtype "
                "to bfloat16 for Cake's native KDA route."
            )
        if {
            "linear_attn_decode_backend",
            "linear_attn_prefill_backend",
        } & changes.keys():
            logger.info(
                "Kimi-Linear equal-head/D128 with bf16 SSM state: defaulting KDA "
                "prefill and decode to Cake's native unbounded-softplus route."
            )
        return

    # Preempts the generic SM100+bf16 flashinfer switch (a GDN default): on
    # KDA shapes the triton packed decode measures ~35% faster than
    # recurrent_kda across bs 1-256, and ReplaySSM requires triton.
    if (
        cfg.linear_attn_decode_backend is None
        and cfg.linear_attn_backend != "cake"
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
