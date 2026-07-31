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

    The NCCL symmetric-memory pool hands out per-forward allocations, and an address
    taken from it during capture is neither reserved for the graph's lifetime nor
    guaranteed to sit at the same offset on every rank. Under capture that corrupts
    speculative decoding, measured under TP8:

      - with --enable-linear-replayssm-spec: accept collapses to 1.000 every run, so
        spec dies and throughput falls below non-spec (81 vs 423 tok/s). Output stays
        correct.
      - without it: 4 of 7 server starts silently emit garbage ("Janet!!!!!!!...")
        while reporting accept pinned at the 8.000 ceiling and a *higher* throughput
        than a healthy run. Nothing errors. This is the dangerous one.

    Only capture is affected, hence the all-eager escape hatch. The condition reads the
    resolved cuda_graph_config rather than the deprecated disable_cuda_graph field,
    which no longer covers --cuda-graph-backend-{decode,prefill}=disabled. Decode is the
    measured path; prefill is included because the same per-forward allocation sits
    inside any captured RowParallelLinear.

    Nothing is lost by dropping the flag: the K3 fused all-reduce (auto-probed, and
    skipped precisely when this flag is set) is both faster than the symmetric-memory
    path and correct under capture.
    """
    from sglang.srt.model_executor.cuda_graph_config import Backend

    if not server_args.enable_symm_mem:
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


def apply_kimi_k3_linear_attn_defaults(server_args: ServerArgs) -> None:
    """KDA decode-fallback default for Kimi hybrid models (spec-independent)."""
    from sglang.srt.utils import is_sm100_supported

    # Preempts the generic SM100+bf16 flashinfer switch (a GDN default): on
    # KDA shapes the triton packed decode measures ~35% faster than
    # recurrent_kda across bs 1-256, and ReplaySSM requires triton.
    if (
        server_args.linear_attn_decode_backend is None
        and server_args.mamba_ssm_dtype == "bfloat16"
        and is_sm100_supported()
    ):
        server_args.linear_attn_decode_backend = "triton"
        logger.info(
            "Kimi hybrid model with bf16 SSM state: defaulting "
            "--linear-attn-decode-backend to triton."
        )
