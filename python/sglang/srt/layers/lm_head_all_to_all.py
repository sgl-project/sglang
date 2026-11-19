"""FP32 logits written straight from the TP LM-head all-to-all output."""

from typing import Optional

import torch

from sglang.kernels.ops.activation.softcap import gathered_softcap_logits
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.log_utils import print_info_once

_DECLINED = (
    "TP LM-head all-to-all: SGLANG_OPT_TP_LM_HEAD_ALL_TO_ALL_FP32 is not used; {}."
)


def fp32_output_enabled(
    *, use_tp_lm_head_all_to_all: bool, final_logit_softcapping: Optional[float]
) -> bool:
    """SGLANG_OPT_TP_LM_HEAD_ALL_TO_ALL_FP32 for a processor on the TP LM-head
    all-to-all; ``all_to_all_output_to_fp32`` still checks each call."""
    if not (
        use_tp_lm_head_all_to_all and envs.SGLANG_OPT_TP_LM_HEAD_ALL_TO_ALL_FP32.get()
    ):
        return False
    if not get_platform().is_blackwell:
        reason = "it needs CUDA sm100 or newer"
    elif final_logit_softcapping:
        reason = "the model softcaps its logits"
    else:
        return True
    print_info_once(_DECLINED.format(reason))
    return False


def all_to_all_output_to_fp32(
    all_to_all_output: torch.Tensor,
    *,
    tp_size: int,
    vocab_size: int,
    logit_scale: Optional[float],
    logits_buffer: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Source-major blocks [tp_size * rows, shard] as [rows, vocab_size] fp32 logits
    in one launch, into ``logits_buffer`` when it fits; None for other inputs.

    The stock path makes the same values with a reassembling copy and an fp32 cast;
    bf16 widens to fp32 exactly, so the logits are bit-identical.
    """
    shard = all_to_all_output.shape[1]
    # Read per call: a model may set logit_scale after constructing its processor.
    if logit_scale is not None:
        reason = "the model scales its logits"
    elif all_to_all_output.dtype != torch.bfloat16 or tp_size * shard < vocab_size:
        reason = (
            "it needs bf16 blocks that cover the vocabulary, got "
            f"{all_to_all_output.dtype} {tp_size} x {shard} for {vocab_size}"
        )
    else:
        reason = None
    if reason is not None:
        print_info_once(_DECLINED.format(reason))
        return None
    print_info_once(
        "TP LM-head all-to-all: the received blocks go straight into fp32 logits "
        "(SGLANG_OPT_TP_LM_HEAD_ALL_TO_ALL_FP32)."
    )
    rows = all_to_all_output.shape[0] // tp_size
    shape = (rows, vocab_size)
    if (
        logits_buffer is None
        or tuple(logits_buffer.shape) != shape
        or logits_buffer.dtype != torch.float32
        or logits_buffer.stride(1) != 1
    ):
        logits_buffer = torch.empty(
            shape, dtype=torch.float32, device=all_to_all_output.device
        )
    # Source-major blocks are the layout of a TP all-gather; no softcap: cast only.
    return gathered_softcap_logits(
        gathered=all_to_all_output.view(tp_size, rows, shard),
        final_logit_softcapping=None,
        output=logits_buffer,
    )
