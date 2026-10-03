# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Select fused completion and read kernels for each consumer."""

from functools import lru_cache, partial
from typing import Callable, Optional, Tuple

import torch

from sglang.srt.layers.layer_boundary.contracts import FfnInputFusion
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.layers.layer_boundary.output import UnreducedOutput
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    NORM_QUANT_READOUT,
    Fp8Input,
    NormQuantReadout,
    _norm_weight,
    aiter_ar_fusion_applies,
    flashinfer_ar_fusion_applies,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.moe import post_experts_reduction_group
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import get_bool_env_var, is_gfx95_supported, is_hip

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()


@lru_cache(maxsize=1)
def _fuses_mxfp4_allreduce() -> bool:
    """Whether the fused AR+RMSNorm+MXFP4 kernel is available. gfx950 only, and
    probed on first use so that importing opens no CUDA context."""
    if not (_use_aiter and is_gfx95_supported()):
        return False
    if get_bool_env_var("SGLANG_DISABLE_FUSED_AR_MXFP4_QUANT", default="false"):
        return False
    try:
        return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


def _complete_quant_input(norm, hidden_states, residual, quant_format, keep_bf16):
    """The all-reduce, residual add, input norm and the consumer's quantization
    in one aiter kernel; None when no kernel serves this format. The result
    carries the consumer's tuple, preceded by the unquantized normed output
    under ``keep_bf16`` for a second projection that reads it."""
    if "mxfp4" in quant_format:
        if not _fuses_mxfp4_allreduce():
            return None
        from sglang.srt.distributed.communication_op import (
            tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant,
        )

        quantized = tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant(
            hidden_states,
            residual,
            _norm_weight(norm),
            norm.variance_epsilon,
            emit_bf16=keep_bf16,
        )
        if quantized is None:
            return _quant_input_over_plain_allreduce(
                norm, hidden_states, residual, keep_bf16
            )
        if keep_bf16:
            fp4, residual_out, scale, normed = quantized
            return (normed, fp4, scale), residual_out
        fp4, residual_out, scale = quantized
        return (fp4, scale), residual_out

    # Engages when the consumer's GEMM takes per-token (1xK) activation scales,
    # i.e. an entry projection carrying per-channel FP8 weights.
    if quant_format == "fp8_per_token" and hasattr(
        norm, "forward_with_allreduce_fusion_quant_per_token"
    ):
        return norm.forward_with_allreduce_fusion_quant_per_token(
            hidden_states,
            residual,
            use_attn_tp_group=False,
            keep_bf16=keep_bf16,
        )
    return None


def _quant_input_over_plain_allreduce(norm, hidden_states, residual, keep_bf16):
    """The fully-fused MXFP4 kernel does not serve this shape: keep the norm and
    quantization fused and unfuse only the all-reduce, so the consumer still
    reads its tuple instead of quantizing the batch itself."""
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.layers.quantization.rocm_mxfp4_utils import fused_rms_mxfp4_quant

    quantized, normed, _, residual_out = fused_rms_mxfp4_quant(
        tensor_model_parallel_all_reduce(hidden_states),
        _norm_weight(norm),
        norm.variance_epsilon,
        None,
        None,
        None,
        residual,
        output_unquantized_inp1=keep_bf16,
    )
    if keep_bf16:
        return (normed, *quantized), residual_out
    return quantized, residual_out


def attn_input_fusions(plan, read=NORM_QUANT_READOUT) -> Tuple[Callable, ...]:
    """The fused kernels that complete what the previous layer left together
    with the residual update and the input norm, in the order they are tried.
    Each takes (owed, residual, forward_batch, post_residual_addition) and
    returns None when it does not take the batch. They add the residual
    plainly; the boundary tries them only when the update it writes in is a
    plain add. A backend's come first."""
    given = plan.fusions.attn_input_fusions(plan) if plan.fusions else ()
    if not hasattr(plan.norm, "forward_with_allreduce_fusion"):
        return given
    declares_quant = isinstance(read, NormQuantReadout)
    fuses_quant = (
        declares_quant
        and read.fp8_input is not None
        and _use_aiter
        and not get_bool_env_var("SGLANG_DISABLE_FUSED_AR_QUANT", default="false")
        and get_exec().comm.enable_aiter_allreduce_fusion
        and hasattr(plan.norm, "forward_with_allreduce_fusion_quant_per_group")
    )
    return (
        *given,
        partial(
            fused_attn_input,
            plan,
            fuses_quant=fuses_quant,
            keep_bf16=fuses_quant and read.fp8_input is Fp8Input.TUPLE_AND_BF16,
            quant_format=read.quant_format if declares_quant else "",
        ),
    )


def fused_attn_input(
    plan,
    owed: UnreducedOutput,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    post_residual_addition: Optional[torch.Tensor],
    *,
    fuses_quant: bool,
    keep_bf16: bool,
    quant_format: str = "",
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Complete the sum the previous layer left, add it to the residual and
    apply the input norm in one aiter or flashinfer kernel; None when the
    kernel does not take this batch. The optional FP8 result follows the consumer read declaration;
    the separate ``quant_format`` path remains the read adapter's responsibility."""
    if (
        not isinstance(owed, UnreducedOutput)
        # The kernel does not add it.
        or post_residual_addition is not None
        # The kernel reduces over the MoE output's group.
        or owed.group is not post_experts_reduction_group()
    ):
        return None
    hidden_states = owed.partial
    if not (
        aiter_ar_fusion_applies(hidden_states, forward_batch)
        or flashinfer_ar_fusion_applies(hidden_states.shape[0])
    ):
        return None
    if quant_format:
        quant_result = _complete_quant_input(
            plan.norm, hidden_states, residual, quant_format, keep_bf16
        )
        if quant_result is not None:
            return quant_result
    # Per-group scales carry a different layout, so this serves only a format
    # that asked for them or did not name one.
    if fuses_quant and quant_format in ("", "fp8"):
        # Falls back to AR+RMSNorm + separate quant internally when the
        # fully-fused kernel cannot service the shape.
        quant_result = plan.norm.forward_with_allreduce_fusion_quant_per_group(
            hidden_states,
            residual,
            use_attn_tp_group=False,
            keep_bf16=keep_bf16,
        )
        if quant_result is not None:
            return quant_result
    return plan.norm.forward_with_allreduce_fusion(
        hidden_states, residual, use_attn_tp_group=False
    )


def ffn_input_fusions(plan) -> Tuple["FfnInputFusion", ...]:
    """The fused kernels that can take the attention -> FFN steps, in the
    order they are tried. They add the residual plainly; the boundary tries
    them only when the update it writes in is a plain add. A backend's come
    first."""
    given = plan.fusions.ffn_input_fusions(plan) if plan.fusions else ()
    if not hasattr(plan.norm, "forward_with_allreduce_fusion"):
        return given
    return (
        *given,
        FfnInputFusion(
            completes=SumGroup.ATTN_TP,
            run=partial(fused_ffn_input, plan),
            preserves_residual=(
                flashinfer_preserves_residual
                if isinstance(plan.norm, (RMSNorm, GemmaRMSNorm))
                and type(plan.norm).forward_with_allreduce_fusion
                in (
                    RMSNorm.forward_with_allreduce_fusion,
                    GemmaRMSNorm.forward_with_allreduce_fusion,
                )
                else None
            ),
        ),
    )


def flashinfer_preserves_residual(value, forward_batch):
    """The CUDA wrapper returns a fresh residual, including backend fallback."""
    return (
        not _use_aiter
        and get_parallel().attn_tp_size > 1
        and flashinfer_ar_fusion_applies(value.shape[0])
    )


def fused_ffn_input(
    plan,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The attention-TP all-reduce, residual add and norm in one aiter or
    flashinfer kernel; None when neither takes the batch."""
    if not (
        aiter_ar_fusion_applies(hidden_states, forward_batch)
        or flashinfer_ar_fusion_applies(hidden_states.shape[0])
    ):
        return None
    return plan.norm.forward_with_allreduce_fusion(
        hidden_states, residual, use_attn_tp_group=True
    )
