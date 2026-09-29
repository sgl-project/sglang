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
"""The plain residual: add and RMSNorm, and the fused all-reduce + add + norm
kernels."""

from typing import Optional

import torch

from sglang.srt.layers.communicator.adapters.attention import (
    _redistribute_from_attn_tp_shards,
    _redistribute_to_attn_tp_shards,
    get_attn_tp_context,
)
from sglang.srt.layers.communicator.residual import LayerResidual
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.flashinfer_comm_fusion import (
    is_flashinfer_allreduce_unavailable,
    uses_cutedsl_ar_fusion,
)
from sglang.srt.layers.quantization.fp8_utils import (
    _use_aiter_bpreshuffle_gfx95,
    materialize_bpreshuffle_fp8_scale_tuple,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_exec, get_parallel, get_platform
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_flashinfer_available,
    is_gfx95_supported,
    is_gfx1250_supported,
    is_hip,
)

_is_cuda = is_cuda()
_is_flashinfer_available = is_flashinfer_available()
_is_sm90_supported = _is_cuda and get_platform().is_sm90
_is_sm100_supported = _is_cuda and get_platform().is_sm100
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
_is_gfx95_supported = is_gfx95_supported()
_is_gfx1250_supported = is_gfx1250_supported()


if _use_aiter:
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype as _aiter_fp8_dtype

    if _is_gfx1250_supported:
        from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant
    else:
        from aiter.ops.rmsnorm import add_rmsnorm_quant as _aiter_add_rmsnorm_quant
        from aiter.ops.rmsnorm import rmsnorm_quant as _aiter_rmsnorm_quant

    if _is_gfx95_supported:
        from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant

        from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
            fused_rms_mxfp4_quant,
        )


def _fused_rmsnorm_fp8_per_token_quant(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    residual: Optional[torch.Tensor] = None,
):
    """Fused (optional residual-add +) RMSNorm + FP8 per-token quantization.

    Only used with the aiter (ROCm) backend.

    Args:
        residual: if provided, computes hidden_states + residual before RMSNorm
                  and returns updated residual_out as second element.

    Returns:
        If residual is None:  (out_fp8, scale)
        If residual provided: ((out_fp8, scale), residual_out)
    """
    if _is_gfx1250_supported:
        # per-token quant == group quant with group_size == hidden size, giving
        # an (M, 1) scale.
        N = hidden_states.shape[-1]
        (out_fp8, scale), _out1, _out2, residual_out = fused_rms_fp8_group_quant(
            hidden_states,
            weight,
            epsilon,
            group_size=N,
            dtype_quant=_aiter_fp8_dtype,
            res1=residual,
        )
        if residual is not None:
            return (out_fp8, scale), residual_out
        return (out_fp8, scale)

    M, N = hidden_states.shape
    out_fp8 = torch.empty((M, N), dtype=_aiter_fp8_dtype, device=hidden_states.device)
    scale = torch.empty(M, dtype=torch.float32, device=hidden_states.device)
    if residual is not None:
        residual_out = torch.empty_like(hidden_states)
        _aiter_add_rmsnorm_quant(
            out_fp8,
            hidden_states,
            residual,
            residual_out,
            scale,
            weight,
            epsilon,
            0,  # group_size=0 → per-token
        )
        return (out_fp8, scale.unsqueeze(1)), residual_out
    else:
        _aiter_rmsnorm_quant(
            out_fp8,
            hidden_states,
            scale,
            weight,
            epsilon,
            0,  # group_size=0 → per-token
        )
        return (out_fp8, scale.unsqueeze(1))


# TODO: According to the discussion in https://github.com/flashinfer-ai/flashinfer/issues/1223#issuecomment-3047256465
# We set the max token num to 128 for allreduce fusion with min-latency case(use_oneshot=True).
FUSE_ALLREDUCE_MAX_BATCH_SIZE = 2048


def apply_flashinfer_allreduce_fusion(batch_size: int):
    return (
        # NOTE: flashinfer 0.6.1 caused performance regression on sm100 for allreduce fusion
        # Ref: https://github.com/sgl-project/sglang/issues/17237
        (_is_sm90_supported or _is_sm100_supported)
        and _is_flashinfer_available
        and not is_dp_attention_enabled()
        and get_exec().comm.flashinfer_allreduce_fusion_backend is not None
        # cutedsl runs its own fused path from the fusion communicator.
        and not uses_cutedsl_ar_fusion()
        and not is_flashinfer_allreduce_unavailable()
        # Symbolic size checks stay last: under Dynamo tracing they guard on
        # the dynamic token dim, so statically-off configs must short-circuit
        # before reaching them.
        and batch_size > 0
        and batch_size <= FUSE_ALLREDUCE_MAX_BATCH_SIZE
    )


def aiter_all_reduce_fusion_enabled_for(forward_mode: ForwardMode) -> bool:
    comm = get_exec().comm
    if not comm.enable_aiter_allreduce_fusion:
        return False
    if forward_mode.is_extend_or_draft_extend_or_mixed():
        return not comm.disable_aiter_allreduce_fusion_in_prefill
    return not comm.disable_aiter_allreduce_fusion_in_decode


def apply_aiter_all_reduce_fusion(
    input_tensor: torch.Tensor, forward_batch: ForwardBatch
):
    n = input_tensor.shape[-1]
    total_bytes = input_tensor.numel() * input_tensor.element_size()
    # Aiter's should_custom_ar uses <= max_size/2 (64 MB); match that boundary.
    return (
        _use_aiter
        and total_bytes > 0
        and n <= 16384
        and total_bytes <= 8 * 1024 * 8192
        and get_parallel().tp_size != 6
        and not is_dp_attention_enabled()
        and aiter_all_reduce_fusion_enabled_for(forward_batch.forward_mode)
    )


def _update_and_read_residual_plain(
    norm, hidden_states, residual, post_residual_addition
):
    if residual is None:
        return norm(hidden_states), hidden_states
    return norm(hidden_states, residual, post_residual_addition)


def _update_and_read_residual_aiter_mxfp4(
    norm, hidden_states, residual, post_residual_addition
):
    # post_residual_addition is not applied on this path.
    output, *_, residual_out = fused_rms_mxfp4_quant(
        hidden_states,
        norm.weight,
        norm.variance_epsilon,
        None,
        None,
        None,
        residual,
    )
    return output, hidden_states if residual is None else residual_out


def _update_and_read_residual_aiter_fp8_group(
    norm, hidden_states, residual, post_residual_addition
):
    """aiter (ROCm gfx95) fused RMSNorm + FP8 group quant. Under DSA the
    unquantized bf16 output rides along as a third element, so the DSA indexer
    can skip dequantizing. post_residual_addition is not applied on this path."""
    needs_bf16 = get_attn_tp_context().is_dsa
    output, unquantized, _, residual_out = fused_rms_fp8_group_quant(
        hidden_states,
        norm.weight,
        norm.variance_epsilon,
        inp2=None,
        inp2_weight=None,
        inp2_epsilon=None,
        group_size=128,
        dtype_quant=torch.float8_e4m3fn,
        res1=residual,
        output_unquantized_inp1=needs_bf16,
        transpose_scale=False,
    )
    if _use_aiter_bpreshuffle_gfx95:
        output = materialize_bpreshuffle_fp8_scale_tuple(output)
    if needs_bf16:
        output = (output[0], output[1], unquantized)
    return output, hidden_states if residual is None else residual_out


def _update_and_read_residual_aiter_fp8_per_token(
    norm, hidden_states, residual, post_residual_addition
):
    if residual is None:
        output = _fused_rmsnorm_fp8_per_token_quant(
            hidden_states, norm.weight.data, norm.variance_epsilon
        )
        return output, hidden_states
    if post_residual_addition is not None:
        residual = residual + post_residual_addition
    return _fused_rmsnorm_fp8_per_token_quant(
        hidden_states, norm.weight.data, norm.variance_epsilon, residual=residual
    )


def _attn_input_update_and_read_residual(quant_format: str):
    """Add the previous layer's output to the residual and read the attention
    input from it: the input norm, fused with the quantization this format
    wants. Without a residual (the first layer, or one already folded into
    hidden_states) the input itself becomes the residual."""
    if _use_aiter and _is_gfx95_supported and "mxfp4" in quant_format:
        return _update_and_read_residual_aiter_mxfp4
    if _use_aiter and _is_gfx95_supported and quant_format == "fp8":
        return _update_and_read_residual_aiter_fp8_group
    if _use_aiter and quant_format == "fp8_per_token":
        return _update_and_read_residual_aiter_fp8_per_token
    return _update_and_read_residual_plain


class Add:
    """A plain residual: the output is added into it."""

    adds_plainly = True
    at_producer = False

    def update(self, hidden_states, residual):
        hidden_states += residual
        return hidden_states

    def residual_to_attn_tp_shard(self, residual, context):
        return _redistribute_to_attn_tp_shards(residual, context)

    def residual_from_attn_tp_shards(self, residual):
        return _redistribute_from_attn_tp_shards(residual)


class NormQuantRead:
    """The input is the residual's norm, fused with the quantization the
    consumer asks for (``quant_format``) and a post-residual addition; a plain
    add of the previous output runs in the same kernel. An empty batch skips
    the norm."""

    norms_plainly = True

    def enter(self, hidden_states):
        return hidden_states

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        if residual.shape[0] == 0:
            return residual, residual
        return _attn_input_update_and_read_residual(quant_format)(
            norm, residual, None, None
        )

    def update_and_read(
        self,
        update,
        hidden_states,
        residual,
        norm,
        quant_format="",
        post_residual_addition=None,
    ):
        if not update.adds_plainly:
            return self.read(update.update(hidden_states, residual), norm, quant_format)
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states
        return _attn_input_update_and_read_residual(quant_format)(
            norm, hidden_states, residual, post_residual_addition
        )


class NormRead:
    """The input is the residual's norm; a plain add of the previous output
    runs in the same kernel. An empty batch skips the norm."""

    norms_plainly = True

    def enter(self, hidden_states):
        return hidden_states

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        if quant_format or post_residual_addition is not None:
            raise NotImplementedError(
                f"a norm read with {quant_format=} or a post-residual addition"
            )
        if residual.shape[0] == 0:
            return residual, residual
        return norm(residual), residual

    def update_and_read(
        self,
        update,
        hidden_states,
        residual,
        norm,
        quant_format="",
        post_residual_addition=None,
    ):
        if residual is None:
            # The layer stack starts at this stage: its input is the residual.
            return self.read(hidden_states, norm, quant_format, post_residual_addition)
        if not update.adds_plainly:
            return self.read(update.update(hidden_states, residual), norm, quant_format)
        if quant_format or post_residual_addition is not None:
            raise NotImplementedError(
                f"a norm read with {quant_format=} or a post-residual addition"
            )
        if hidden_states.shape[0] == 0:
            return hidden_states, residual
        return norm(hidden_states, residual)


ADD = Add()
NORM_QUANT_READ = NormQuantRead()
NORM_READ = NormRead()
# A plain residual: the attention reads with its input norm and the quantization
# it wants, the FFN with its norm, and each stage's output is added.
PLAIN_RESIDUAL = LayerResidual(
    attention_read=NORM_QUANT_READ,
    attention_update=ADD,
    ffn_read=NORM_READ,
    ffn_update=ADD,
)
