from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def apply_kimi_k3_spec_backend_defaults(server_args: ServerArgs) -> None:
    """Apply speculative backend defaults for Kimi hybrid models."""
    cfg = resolving_view(server_args)
    from sglang.srt.utils import is_sm100_supported

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
        and is_sm100_supported()
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


def disable_kimi_k3_symm_mem(server_args: ServerArgs) -> None:
    """Turn `--enable-symm-mem` back off unless every phase runs eager.

    Symm-mem allocations are per-forward, so an address captured into a graph is
    neither reserved for its lifetime nor at the same offset on every rank. Under
    capture that corrupts spec decode: accept collapses to 1.000, or the server
    silently emits garbage with accept pinned at the ceiling. Prefill counts too --
    the same allocation sits in any captured RowParallelLinear.

    Gates on the arch itself: this runs from cuda-graph resolution, which is earlier
    than the model-specific hook block.
    """
    cfg = resolving_view(server_args)
    from sglang.srt.connector import ConnectorType
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.utils import parse_connector_type

    if not cfg.enable_symm_mem:
        return
    if parse_connector_type(cfg.model_path) == ConnectorType.INSTANCE:
        return
    if server_args.get_model_config().hf_config.architectures[0] not in (
        "KimiLinearForCausalLM",
        "KimiK3ForConditionalGeneration",
    ):
        return
    graph = cfg.cuda_graph_config
    if (
        graph.decode.backend == Backend.DISABLED
        and graph.prefill.backend == Backend.DISABLED
    ):
        return
    declare_resolution(
        server_args,
        "disable_kimi_k3_symm_mem",
        enable_symm_mem=False,
    )
    logger.warning(
        "Kimi hybrid model: ignoring --enable-symm-mem because CUDA graphs are on. "
        "The symmetric-memory pool's per-forward allocations are not valid for the "
        "lifetime of a captured graph, which corrupts speculative decoding and can "
        "silently produce wrong output. The auto-probed K3 fused all-reduce is faster "
        "anyway. Disable capture on every phase "
        "(--cuda-graph-backend-decode=disabled --cuda-graph-backend-prefill=disabled) "
        "if you genuinely need symmetric memory."
    )


def apply_kimi_k3_linear_attn_defaults(server_args: ServerArgs) -> None:
    """KDA decode-fallback default for Kimi hybrid models (spec-independent)."""
    cfg = resolving_view(server_args)
    from sglang.srt.utils import is_sm100_supported

    # Preempts the generic SM100+bf16 flashinfer switch (a GDN default): on
    # KDA shapes the triton packed decode measures ~35% faster than
    # recurrent_kda across bs 1-256, and ReplaySSM requires triton.
    if (
        cfg.linear_attn_decode_backend is None
        and cfg.mamba_ssm_dtype == "bfloat16"
        and is_sm100_supported()
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
