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

    # Keep KDA target-verify on triton: flashinfer recurrent_kda verify is slower on
    # all measured shapes and can't run ragged/compact layouts (uniform [N,T]).
    # Decode is left free (its bf16-ssm SM100+ flashinfer default is fine -- the
    # target only verifies under spec); the verify backend is pinned directly.
    if server_args.linear_attn_verify_backend is None:
        server_args.linear_attn_verify_backend = "triton"
        logger.info(
            "Kimi hybrid model with speculative decoding: pinning "
            "--linear-attn-verify-backend to triton (keeps KDA verify on "
            "the triton kernel)."
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
