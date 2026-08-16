from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def apply_kimi_k3_spec_backend_defaults(server_args: ServerArgs) -> None:
    """Apply speculative backend defaults for Kimi hybrid models."""
    from sglang.srt.utils import is_sm100_supported

    if server_args.speculative_algorithm is None:
        return

    # Use the fused Kimi-K3/DSPARK CuTeDSL kernel for KDA target verification.
    # Decode is left free (its bf16-ssm SM100+ flashinfer default is fine -- the
    # target only verifies under spec); the verify backend is pinned directly.
    if server_args.linear_attn_verify_backend is None:
        server_args.linear_attn_verify_backend = "nv_cutedsl"
        logger.info(
            "Kimi hybrid model with speculative decoding: pinning "
            "--linear-attn-verify-backend to nv_cutedsl (uses the fused "
            "Kimi-K3/DSPARK CuTeDSL kernel)."
        )

    # dspark's draft is dense MQA; trtllm_mha avoids flashinfer's blocking
    # per-step host plan. DSPARK-only: other spec algos use MLA-family drafts.
    if (
        server_args.speculative_algorithm == "DSPARK"
        and server_args.speculative_draft_attention_backend is None
        and is_sm100_supported()
    ):
        server_args.speculative_draft_attention_backend = "trtllm_mha"
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
    from sglang.srt.connector import ConnectorType
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.utils import parse_connector_type

    if not server_args.enable_symm_mem:
        return
    if parse_connector_type(server_args.model_path) == ConnectorType.INSTANCE:
        return
    if server_args.get_model_config().hf_config.architectures[0] not in (
        "KimiLinearForCausalLM",
        "KimiK3ForConditionalGeneration",
    ):
        return
    graph = server_args.cuda_graph_config
    if (
        graph.decode.backend == Backend.DISABLED
        and graph.prefill.backend == Backend.DISABLED
    ):
        return
    server_args.enable_symm_mem = False
    logger.warning(
        "Kimi hybrid model: ignoring --enable-symm-mem because CUDA graphs are on. "
        "The symmetric-memory pool's per-forward allocations are not valid for the "
        "lifetime of a captured graph, which corrupts speculative decoding and can "
        "silently produce wrong output. The auto-probed K3 fused all-reduce is faster "
        "anyway. Disable capture on every phase "
        "(--cuda-graph-backend-decode=disabled --cuda-graph-backend-prefill=disabled) "
        "if you genuinely need symmetric memory."
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
    tp_size = getattr(server_args, "tp_size", 1)
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
    from sglang.srt.utils import is_sm100_supported

    native_kimi_linear = _uses_native_kimi_linear_unbounded_kda(
        server_args, model_arch=model_arch, hf_config=hf_config
    )
    if native_kimi_linear and is_sm100_supported():
        if server_args.mamba_ssm_dtype is None:
            server_args.mamba_ssm_dtype = "bfloat16"
            logger.info(
                "Kimi-Linear equal-head/D128: defaulting --mamba-ssm-dtype "
                "to bfloat16 for Cake's native KDA route."
            )
        if server_args.mamba_ssm_dtype != "bfloat16":
            return
        if server_args.linear_attn_decode_backend is None:
            server_args.linear_attn_decode_backend = "cake"
        if server_args.linear_attn_prefill_backend is None:
            server_args.linear_attn_prefill_backend = "cake"
        logger.info(
            "Kimi-Linear equal-head/D128 with bf16 SSM state: defaulting KDA "
            "prefill and decode to Cake's native unbounded-softplus route."
        )
        return

    # Preempts the generic SM100+bf16 flashinfer switch (a GDN default): on
    # KDA shapes the triton packed decode measures ~35% faster than
    # recurrent_kda across bs 1-256, and ReplaySSM requires triton.
    if (
        server_args.linear_attn_decode_backend is None
        and server_args.linear_attn_backend != "cake"
        and server_args.mamba_ssm_dtype == "bfloat16"
        and is_sm100_supported()
    ):
        server_args.linear_attn_decode_backend = "triton"
        logger.info(
            "Kimi hybrid model with bf16 SSM state: defaulting "
            "--linear-attn-decode-backend to triton."
        )
