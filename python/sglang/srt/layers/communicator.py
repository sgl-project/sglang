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
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property, partial
from typing import Callable, Dict, Optional, Tuple, Union

import msgspec
import torch

from sglang.srt.distributed import (
    GroupCoordinator,
    attention_tensor_model_parallel_all_reduce,
    attention_tensor_model_parallel_quant_all_reduce,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
)
from sglang.srt.layers.aux_hidden_states import AuxHiddenStateAccumulator
from sglang.srt.layers.boundary_layout import Layout, TokenAxis
from sglang.srt.layers.cp.utils import (
    is_mla_cp_active,
    is_mla_cp_enabled,
)
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    attn_tp_reduce_scatter_tensor,
    can_use_dp_reduce_scatter,
    dp_gather_partial,
    dp_gather_replicate,
    dp_reduce_scatter_tensor,
    dp_scatter,
    get_dp_global_num_tokens,
    get_global_dp_buffer,
    get_local_dp_buffer,
    get_moe_cp_rank,
    get_moe_cp_size,
    is_allocation_symmetric,
    is_dp_attention_enabled,
    is_enable_moe_cp_allgather,
    moe_cp_all_gather_into_tensor,
)
from sglang.srt.layers.flashinfer_comm_fusion import (
    is_flashinfer_allreduce_unavailable,
    uses_cutedsl_ar_fusion,
)
from sglang.srt.layers.moe import (
    can_merge_post_experts_all_reduce,
    get_moe_a2a_backend,
    post_experts_output_is_complete,
    post_experts_reduction_group,
    should_use_dp_reduce_scatterv,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.quantization.fp8_utils import (
    _use_aiter_bpreshuffle_gfx95,
    materialize_bpreshuffle_fp8_scale_tuple,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import (
    LoRABatchLayout,
    get_exec,
    get_forward,
    get_lora,
    get_parallel,
    get_platform,
    get_spec,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_flashinfer_available,
    is_gfx95_supported,
    is_gfx1250_supported,
    is_hip,
    is_npu,
)

_is_cuda = is_cuda()
_is_flashinfer_available = is_flashinfer_available()
_is_sm90_supported = _is_cuda and get_platform().is_sm90
_is_sm100_supported = _is_cuda and get_platform().is_sm100
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
_is_gfx95_supported = is_gfx95_supported()
_is_gfx1250_supported = is_gfx1250_supported()
_is_npu = is_npu()
_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()

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
elif _is_npu:
    from sglang.srt.hardware_backend.npu.cmo import prepare_weight_cache


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


class ScatterMode(Enum):
    """
    Suppose we have TP=4, DP=2, enable-dp-attention, and the system handles seq a,b,c,d
    Model input/output: [ab, ab, cd, cd] for four ranks respectively
    SCATTERED: [a, b, c, d]
    TP_ATTN_FULL: [ab, ab, cd, cd], i.e. all ranks inside a TP attn group have full data of the group
    FULL: [abcd, abcd, abcd, abcd]
    MOE_FULL: full within the MoE group (cp_per_moe CP chunks), used when moe_dp_size < attn_cp_size
    """

    SCATTERED = auto()
    TP_ATTN_FULL = auto()
    FULL = auto()
    MOE_FULL = auto()

    @staticmethod
    def model_input_output():
        """The scatter mode for model forward pass input and output data"""
        if is_dsa_enable_prefill_cp() or is_mla_cp_enabled():
            return ScatterMode.SCATTERED

        return ScatterMode.TP_ATTN_FULL


class AttentionInputs:
    def __init__(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        qkv_latent_func: Callable,
        *,
        is_pre_gathered: bool = False,
    ):
        self.hidden_states_local = hidden_states
        self.forward_batch = forward_batch
        self.qkv_latent_func = qkv_latent_func
        self.hidden_states_ = None
        self.qkv_latent_ = None
        # When True, hidden_states_local is already attn_tp-gathered upstream
        # (e.g. by MHC's prepare_attn for DSA). fetch_* must NOT gather again.
        self.is_pre_gathered = is_pre_gathered

    def tp_all_gather_hidden_states(self, hidden_states, forward_batch):
        total_tokens = forward_batch.input_ids.shape[0]
        output = hidden_states.new_empty((total_tokens, hidden_states.shape[-1]))
        get_parallel().tp_group.all_gather_into_tensor(output, hidden_states)
        return output

    def fetch_qkv_latent(self):
        if self.qkv_latent_ is not None:
            return self.qkv_latent_
        assert self.qkv_latent_func is not None
        self.qkv_latent_ = self.qkv_latent_func(
            self.hidden_states_local, self.forward_batch
        )
        if get_attn_tp_context().input_scattered and not self.is_pre_gathered:
            self.qkv_latent_ = self.tp_all_gather_hidden_states(
                self.qkv_latent_, self.forward_batch
            )
        return self.qkv_latent_

    def fetch_hidden_states(self):
        if self.hidden_states_ is not None:
            return self.hidden_states_
        self.hidden_states_ = self.hidden_states_local
        if get_attn_tp_context().input_scattered and not self.is_pre_gathered:
            self.hidden_states_ = self.tp_all_gather_hidden_states(
                self.hidden_states_, self.forward_batch
            )
        return self.hidden_states_


class AttnTpContext:
    def __init__(self):
        self.allow_input_scattered = False
        self.is_dsa = False

    def init_context(self, q_lora_rank, is_dsa, is_mhc=False):
        # Only MHC pre-gathers hidden states before DSA attention, so non-MHC DSA
        # cannot use scattered inputs.
        self.is_dsa = is_dsa
        self.allow_input_scattered = (
            get_parallel().enable_attn_tp_input_scattered
            and (_is_cuda or _is_npu)
            and q_lora_rank is not None
            and (is_mhc or not is_dsa)
            and get_parallel().tp_size > 1
            and not is_dp_attention_enabled()
            and get_moe_a2a_backend().is_none()
            and not enable_moe_dense_fully_dp()
            and not check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
            and get_spec().speculative_algorithm != "EAGLE3"
        )
        if get_parallel().enable_attn_tp_input_scattered:
            if not self.allow_input_scattered:
                logging.info(
                    "attn_tp_input_scattered is not enabled while other conditions are not met"
                )
            else:
                logging.info("attn_tp_input_scattered is enabled")

    def use_input_scattered(self, forward_batch: ForwardBatch):
        return (
            self.allow_input_scattered
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.input_ids is not None
            and not forward_batch.can_run_tbo
        )

    @property
    def input_scattered(self):
        return get_forward().attn_input_scattered

    def set_attn_inputs(self, attn_inputs: AttentionInputs):
        get_forward().set("attn_inputs", attn_inputs)

    def set_hidden_states_local(self, hidden_states: torch.Tensor) -> None:
        attn_inputs = get_forward().attn_inputs
        if attn_inputs is not None:
            attn_inputs.hidden_states_local = hidden_states

    def fetch_qkv_latent(self):
        attn_inputs = get_forward().attn_inputs
        assert attn_inputs is not None
        return attn_inputs.fetch_qkv_latent()

    def fetch_hidden_states(self):
        attn_inputs = get_forward().attn_inputs
        assert attn_inputs is not None
        return attn_inputs.fetch_hidden_states()

    def clear_attn_inputs(self) -> None:
        get_forward().set("attn_inputs", None)

    @contextmanager
    def maybe_input_scattered(self, forward_batch: ForwardBatch):
        flag = self.use_input_scattered(forward_batch)
        forward = get_forward()
        # scoped() also restores when the forward raises — the old in-place
        # swap leaked the flag on exceptions.
        with forward.scoped(attn_input_scattered=flag):
            try:
                yield
            finally:
                forward.set("attn_inputs", None)


ATTN_TP_CONTEXT = AttnTpContext()


def get_attn_tp_context():
    return ATTN_TP_CONTEXT


@dataclass
class _LayerModeComputationContext:
    num_layers: int
    layer_id: int
    is_layer_sparse: bool
    is_previous_layer_sparse: Optional[bool]
    is_next_layer_sparse: Optional[bool]

    def previous_layer(self):
        assert self.is_previous_layer_sparse is not None
        return _LayerModeComputationContext(
            num_layers=self.num_layers,
            layer_id=self.layer_id - 1,
            is_layer_sparse=self.is_previous_layer_sparse,
            is_previous_layer_sparse=None,
            is_next_layer_sparse=self.is_layer_sparse,
        )


def sparse_mlp_scatter_mode() -> ScatterMode:
    """SCATTERED hands a sparse MLP this rank's own token shard; FULL and
    MOE_FULL hand it a buffer gathered over the attn-TP or MoE-CP group."""
    if (
        # Token dispatch/combine will be handled outside of LayerCommunicator for these modes.
        not get_moe_a2a_backend().is_none()
        or should_use_flashinfer_cutlass_moe_fp4_allgather()
        or enable_dwdp()
    ):
        return ScatterMode.SCATTERED
    # DSA CP and MLA CP both don't support MOE_FULL yet; fall back to FULL.
    if is_enable_moe_cp_allgather() and not (
        is_dsa_enable_prefill_cp() or is_mla_cp_enabled()
    ):
        return ScatterMode.MOE_FULL
    return ScatterMode.FULL


@dataclass
class LayerScatterModes:
    layer_input_mode: ScatterMode
    attn_mode: ScatterMode
    # Can be further split into e.g. mlp_input_mode and mlp_output_mode if needed
    mlp_mode: ScatterMode
    middle_residual_mode: ScatterMode
    layer_output_mode: ScatterMode
    is_layer_sparse: bool = False
    # The model's last layer: its output goes to the final norm, not a next layer.
    is_last_layer: bool = False

    @classmethod
    def init_new(cls, **kwargs):
        context = _LayerModeComputationContext(**kwargs)
        return cls(
            layer_input_mode=cls._compute_layer_input_mode(context),
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=cls._compute_mlp_mode(context),
            middle_residual_mode=cls._compute_middle_residual_mode(context),
            layer_output_mode=cls._compute_layer_output_mode(context),
            is_layer_sparse=context.is_layer_sparse,
            is_last_layer=context.layer_id == context.num_layers - 1,
        )

    @classmethod
    def _compute_layer_input_mode(cls, context: _LayerModeComputationContext):
        if context.layer_id == 0:
            return ScatterMode.model_input_output()
        return cls._compute_layer_output_mode(context.previous_layer())

    @classmethod
    def _compute_mlp_mode(cls, context: _LayerModeComputationContext):
        if context.is_layer_sparse:
            return sparse_mlp_scatter_mode()
        else:
            if enable_moe_dense_fully_dp():
                return ScatterMode.SCATTERED
            # A TP-sharded dense MLP reduces over the whole TP group, which spans
            # every CP rank; a CP-sharded prefill must gather tokens across CP
            # first or the all-reduce sums different tokens' partial outputs.
            # MLA/DSA CP models do this in DSACPLayerCommunicator instead.
            if _generic_prefill_cp_shards_tokens() and not (
                is_dsa_enable_prefill_cp() or is_mla_cp_enabled()
            ):
                return ScatterMode.MOE_FULL
            return ScatterMode.FULL

    @classmethod
    def _should_gather_for_tbo(cls, context: _LayerModeComputationContext):
        return (
            not context.is_layer_sparse
            and context.is_next_layer_sparse
            and enable_moe_dense_fully_dp()
            and get_exec().overlap.enable_two_batch_overlap
        )

    @classmethod
    def _compute_middle_residual_mode(cls, context: _LayerModeComputationContext):
        mlp_mode = cls._compute_mlp_mode(context)
        if mlp_mode == ScatterMode.SCATTERED:
            return ScatterMode.SCATTERED
        if mlp_mode in (ScatterMode.FULL, ScatterMode.MOE_FULL):
            return ScatterMode.TP_ATTN_FULL
        raise NotImplementedError

    @classmethod
    def _compute_layer_output_mode(cls, context: _LayerModeComputationContext):
        mlp_mode = cls._compute_mlp_mode(context)
        if context.layer_id == context.num_layers - 1:
            return ScatterMode.model_input_output()
        if mlp_mode == ScatterMode.SCATTERED:
            if cls._should_gather_for_tbo(context):
                return ScatterMode.TP_ATTN_FULL
            return ScatterMode.SCATTERED
        if mlp_mode in (ScatterMode.FULL, ScatterMode.MOE_FULL):
            return ScatterMode.TP_ATTN_FULL
        raise NotImplementedError


def enable_moe_dense_fully_dp():
    return get_parallel().moe_dense_tp_size == 1


def _generic_prefill_cp_shards_tokens() -> bool:
    """Whether the strategy prefill CP path shards prefill tokens across CP ranks."""
    parallel = get_parallel()
    return parallel.attn_cp_size > 1 and parallel.enable_prefill_cp


def enable_dwdp():
    return get_parallel().dwdp_size > 1


def tp_reduce_scatter(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    context: "CommunicateContext",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Module-level so MHC communicators can reuse it without a
    ``LayerCommunicator`` instance."""
    if hidden_states.shape[0] == 0:
        return hidden_states, hidden_states
    assert hidden_states.shape[0] % context.tp_size == 0, (
        f"Expected total tokens {hidden_states.shape[0]} % tp_size {context.tp_size} to be 0"
    )
    local_tokens = hidden_states.shape[0] // context.tp_size
    output = hidden_states.new_empty(local_tokens, *hidden_states.shape[1:])
    get_parallel().tp_group.reduce_scatter_tensor(output, hidden_states)
    if residual is not None:
        residual = residual.tensor_split(context.tp_size)[context.tp_rank]
    return output, residual


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


class UnreducedOutput(msgspec.Struct, frozen=True):
    """A layer output that still owes its sum, left for the next layer's input.
    Hand it to the next layer, or pass it through reduce_output() before reading
    it any other way. The producer says what is owed: one all-reduce over
    ``group`` that keeps the layout, or ``reduce_and_redistribute``."""

    partial: torch.Tensor
    group: Optional[GroupCoordinator] = None
    # Under attention DP: the reduction that also brings ``partial`` back to this
    # rank's tokens (a reduce-scatter, or an all-reduce then a scatter).
    reduce_and_redistribute: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


def reduce_output(
    hidden_states: Union[torch.Tensor, UnreducedOutput, None],
) -> Optional[torch.Tensor]:
    """Run the reduction an UnreducedOutput still owes; pass anything else through."""
    if isinstance(hidden_states, UnreducedOutput):
        if hidden_states.reduce_and_redistribute is not None:
            return hidden_states.reduce_and_redistribute(hidden_states.partial)
        return hidden_states.group.all_reduce(hidden_states.partial)
    return hidden_states


def layer_input_buffer(
    hidden_states: Union[torch.Tensor, UnreducedOutput],
) -> torch.Tensor:
    """The tensor holding a layer's input, for reusing its memory without reading it."""
    if isinstance(hidden_states, UnreducedOutput):
        return hidden_states.partial
    return hidden_states


def _batch_size(forward_batch: ForwardBatch) -> int:
    return (
        forward_batch.input_ids.shape[0] if hasattr(forward_batch, "input_ids") else 0
    )


def _deferred_reduction_runs_on_the_tp_group() -> bool:
    """Whether an MoE output's one all-reduce runs over the TP group object
    itself, the group a dense FFN and reduce_moe_output reduce over."""
    parallel = get_parallel()
    if parallel.moe_ep_size > 1 and parallel.moe_tp_size > 1:
        # Some MoE blocks reduce EP and MoE-TP in two steps instead of merging.
        return False
    return post_experts_reduction_group() is parallel.tp_group


def _ffn_has_tokens(forward_batch: ForwardBatch) -> bool:
    if is_dp_attention_enabled():
        # The FFN runs on every DP rank's tokens, so every rank decides alike.
        return (getattr(forward_batch, "global_dp_buffer_len", None) or 0) > 0
    return _batch_size(forward_batch) > 0


def _unfused_completion_matches_the_ffn(forward_batch: ForwardBatch) -> bool:
    """Whether the next layer's all-reduce is the one the FFN would have run
    itself: one full-precision sum over the same TP group, on an output
    that is a plain partial sum. The output is not a plain partial sum when the
    MoE combine already summed it, a replicated shared expert is added after the
    reduction, or LoRA-B runs on the unreduced activations."""
    return (
        _ffn_has_tokens(forward_batch)
        and get_moe_a2a_backend().is_none()
        and not post_experts_output_is_complete(is_tp_path=True)
        and not get_exec().comm.enable_quant_communications
        and not envs.SGLANG_SHARED_EXPERT_TP1.get()
        and not get_lora().enable_lora
        and _deferred_reduction_runs_on_the_tp_group()
    )


class LayerCommunicator:
    # Communicators built without __init__ (e.g. test doubles) publish no LoRA layout.
    _publish_lora_layout: bool = False

    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        # Reduce scatter requires skipping all-reduce in model code after MoE/MLP, so only enable for models which have that implemented. Remove flag once done for all models that use LayerCommunicator.
        allow_reduce_scatter: bool = False,
        qkv_latent_func: Optional[Callable] = None,
        force_layernorm_before_dp_gather: bool = False,
        enable_fused_ar_quant: bool = False,
        fused_ar_quant_keep_bf16: bool = False,
        _is_sp_variant: bool = False,
    ):
        self.layer_scatter_modes = layer_scatter_modes
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.allow_reduce_scatter = allow_reduce_scatter
        self.is_last_layer = layer_scatter_modes.is_last_layer
        self.qkv_latent_func = qkv_latent_func
        self.force_layernorm_before_dp_gather = force_layernorm_before_dp_gather
        self.enable_fused_ar_quant = enable_fused_ar_quant
        self.fused_ar_quant_keep_bf16 = fused_ar_quant_keep_bf16

        self._context = CommunicateContext.init_new()
        self._context.force_layernorm_before_dp_gather = (
            force_layernorm_before_dp_gather
        )
        self._post_init_communicate()
        # Under attention DP, the base postprocess scatters the FFN output back
        # to this rank's tokens, which the next layer's input can run instead;
        # the MHC and DSA-CP postprocess do more, so theirs stays here.
        self._postprocess_scatters_to_local_tokens = (
            self._communicate_summable_tensor_pair_fn
            is CommunicateSummableTensorPairFn._scatter_hidden_states
        )
        self._speculative_algo = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        # LoRA kernels need the per-layer token layout only under DP attention.
        self._publish_lora_layout = get_parallel().enable_dp_attention and bool(
            get_lora().enable_lora
        )

        # Under LayerNorm SP the norm/residual run on the sequence shard with no
        # collectives, so delegate to an all-SCATTERED sibling while the region is
        # active. _is_sp_variant stops the sibling from building its own.
        self._sp_variant: Optional[LayerCommunicator] = None
        if not _is_sp_variant and layernorm_sp.layernorm_sp_enabled():
            self._sp_variant = LayerCommunicator(
                layer_scatter_modes=LayerScatterModes(
                    layer_input_mode=ScatterMode.SCATTERED,
                    attn_mode=ScatterMode.SCATTERED,
                    mlp_mode=ScatterMode.SCATTERED,
                    middle_residual_mode=ScatterMode.SCATTERED,
                    layer_output_mode=ScatterMode.SCATTERED,
                    is_last_layer=layer_scatter_modes.is_last_layer,
                ),
                input_layernorm=input_layernorm,
                post_attention_layernorm=post_attention_layernorm,
                allow_reduce_scatter=allow_reduce_scatter,
                qkv_latent_func=qkv_latent_func,
                force_layernorm_before_dp_gather=force_layernorm_before_dp_gather,
                enable_fused_ar_quant=enable_fused_ar_quant,
                fused_ar_quant_keep_bf16=fused_ar_quant_keep_bf16,
                _is_sp_variant=True,
            )

    def _post_init_communicate(self):
        self._communicate_simple_fn = CommunicateSimpleFn.get_fn(
            input_mode=self.layer_scatter_modes.layer_input_mode,
            output_mode=self.layer_scatter_modes.attn_mode,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = (
            CommunicateWithAllReduceAndLayerNormFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.attn_mode,
                residual_input_mode=self.layer_scatter_modes.layer_input_mode,
                hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,
                residual_output_mode=self.layer_scatter_modes.middle_residual_mode,
                context=self._context,
            )
        )
        self._communicate_summable_tensor_pair_fn = (
            CommunicateSummableTensorPairFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,
                residual_input_mode=self.layer_scatter_modes.middle_residual_mode,
                output_mode=self.layer_scatter_modes.layer_output_mode,
                context=self._context,
            )
        )

    def prepare_attn_and_capture_last_layer_outputs(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        captured_last_layer_outputs: Optional[AuxHiddenStateAccumulator] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
        quant_format: str = "",
    ):
        hidden_states, residual = self.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            quant_format=quant_format,
            post_residual_addition=post_residual_addition,
        )
        if captured_last_layer_outputs is not None:
            gathered_last_layer_output = self._communicate_simple_fn(
                hidden_states=residual,
                forward_batch=forward_batch,
                context=self._context,
            )
            if (
                gathered_last_layer_output is residual
                # An accumulator that copies on append already holds a snapshot.
                and not getattr(captured_last_layer_outputs, "copies_on_append", False)
                and not self._post_attn_residual_is_read_only(residual)
            ):
                gathered_last_layer_output = residual.clone()
            captured_last_layer_outputs.append(gathered_last_layer_output)
        return hidden_states, residual

    def _post_attn_residual_is_read_only(self, residual: torch.Tensor) -> bool:
        """True if ``prepare_mlp``'s post-attention RMSNorm leaves ``residual``
        untouched, so Eagle3 aux capture can keep its reference and skip the clone.

        Only the flashinfer all-reduce-fusion path writes a fresh ``residual_out``
        (see ``flashinfer_allreduce_residual_rmsnorm``); the aiter fused kernel and
        every plain norm fold into ``residual`` in place. That path is reachable
        only from the ``_gather_*`` communicate-fns, and only when they fall past
        their input-scattered branch.
        """
        norm_fn = getattr(
            self._communicate_with_all_reduce_and_layer_norm_fn,
            "func",
            self._communicate_with_all_reduce_and_layer_norm_fn,
        )
        uses_gather_norm = norm_fn in (
            CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
            CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual_moe,
        )
        return (
            uses_gather_norm
            and not get_attn_tp_context().input_scattered
            and apply_flashinfer_allreduce_fusion(residual.shape[0])
        )

    def publish_attn_lora_layout(self) -> None:
        """Attention consumes the DP-local token batch."""
        if self._publish_lora_layout:
            get_forward().set("lora_batch_layout", LoRABatchLayout.DP_LOCAL)

    def publish_mlp_lora_layout(self) -> None:
        """The MLP consumes the TP-global batch only after a FULL DP gather."""
        if self._publish_lora_layout:
            get_forward().set(
                "lora_batch_layout",
                (
                    LoRABatchLayout.TP_GLOBAL
                    if self.layer_scatter_modes.mlp_mode is ScatterMode.FULL
                    else LoRABatchLayout.DP_LOCAL
                ),
            )

    def prepare_attn(
        self,
        hidden_states: Union[torch.Tensor, UnreducedOutput],
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        quant_format: str = "",
        post_residual_addition: Optional[torch.Tensor] = None,
    ):
        self.publish_attn_lora_layout()

        if isinstance(hidden_states, UnreducedOutput) and residual is None:
            raise RuntimeError("an UnreducedOutput requires residual input")
        if (
            isinstance(hidden_states, UnreducedOutput)
            and hidden_states.reduce_and_redistribute is not None
        ):
            # No fused kernel runs under attention DP: the reduce-scatter back to
            # this rank's tokens comes first.
            hidden_states = reduce_output(hidden_states)
        unreduced = (
            hidden_states if isinstance(hidden_states, UnreducedOutput) else None
        )
        pending = unreduced is not None
        if pending:
            hidden_states = unreduced.partial

        # residual is None marks the first decoder layer, where the SP region
        # opens: re-evaluated per forward so a crash mid-loop cannot leak into
        # the next one.
        if self._sp_variant is not None:
            if residual is None:
                get_forward().set(
                    "sp_active", forward_batch.forward_mode == ForwardMode.EXTEND
                )
                if get_forward().sp_active:
                    hidden_states = layernorm_sp.sp_entry_scatter(hidden_states)
            if get_forward().sp_active:
                return self._sp_variant.prepare_attn(
                    hidden_states,
                    residual,
                    forward_batch,
                    quant_format,
                    post_residual_addition,
                )
        if get_attn_tp_context().input_scattered:
            hidden_states, residual = self._tp_reduce_scatter(
                hidden_states,
                residual,
            )
        if hidden_states.shape[0] == 0:
            residual = hidden_states
        elif residual is not None and pending:
            fused = (
                self._reduce_output_and_update_and_read_residual(
                    hidden_states, residual, forward_batch
                )
                if post_residual_addition is None
                # The fused kernel reduces over the MoE output's group.
                and unreduced.group is post_experts_reduction_group()
                else None
            )
            if fused is not None:
                hidden_states, residual = fused
            else:
                hidden_states = reduce_output(unreduced)
                hidden_states, residual = _attn_input_update_and_read_residual(
                    quant_format
                )(self.input_layernorm, hidden_states, residual, post_residual_addition)
        else:
            hidden_states, residual = _attn_input_update_and_read_residual(
                quant_format
            )(self.input_layernorm, hidden_states, residual, post_residual_addition)

        return self._finish_prepare_attn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

    def _finish_prepare_attn(self, hidden_states, residual, forward_batch):
        """Tail every prepare_attn path must run, or ``attn_inputs`` is unset."""
        hidden_states = self._communicate_simple_fn(
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            context=self._context,
        )
        if self.qkv_latent_func is not None:
            attn_inputs = AttentionInputs(
                hidden_states, forward_batch, self.qkv_latent_func
            )
            get_attn_tp_context().set_attn_inputs(attn_inputs)
        return hidden_states, residual

    def _reduce_output_and_update_and_read_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Complete the sum the previous layer left, add it to the residual and
        apply the input norm in one fused kernel; None when the kernel does not
        take this batch. The result is not quantized for ``quant_format``."""
        if (
            apply_aiter_all_reduce_fusion(hidden_states, forward_batch)
            or apply_flashinfer_allreduce_fusion(hidden_states.shape[0])
        ) and hasattr(self.input_layernorm, "forward_with_allreduce_fusion"):
            if (
                self.enable_fused_ar_quant
                and _use_aiter
                and hasattr(
                    self.input_layernorm,
                    "forward_with_allreduce_fusion_quant_per_group",
                )
            ):
                # Falls back to AR+RMSNorm + separate quant internally when the
                # fully-fused kernel cannot service the shape.
                quant_result = (
                    self.input_layernorm.forward_with_allreduce_fusion_quant_per_group(
                        hidden_states,
                        residual,
                        use_attn_tp_group=False,
                        keep_bf16=self.fused_ar_quant_keep_bf16,
                    )
                )
                if quant_result is not None:
                    return quant_result
            return self.input_layernorm.forward_with_allreduce_fusion(
                hidden_states, residual, use_attn_tp_group=False
            )
        return None

    def _tp_reduce_scatter(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tp_reduce_scatter(hidden_states, residual, self._context)

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        cache=None,
    ):
        self.publish_mlp_lora_layout()
        if self._sp_variant is not None and get_forward().sp_active:
            return self._sp_variant.prepare_mlp(
                hidden_states, residual, forward_batch, cache
            )
        if cache is not None:
            self._context.cache = cache

        return self._communicate_with_all_reduce_and_layer_norm_fn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            layernorm=self.post_attention_layernorm,
            context=self._context,
        )

    def maybe_prefetch_next_full_attention_kv(
        self,
        forward_batch: ForwardBatch,
        next_full_attention_layer_id: Optional[int],
    ) -> None:
        return

    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if self._sp_variant is not None and get_forward().sp_active:
            return self._sp_variant.postprocess_layer(
                hidden_states, residual, forward_batch
            )
        return self._communicate_summable_tensor_pair_fn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            context=self._context,
            allow_reduce_scatter=self.allow_reduce_scatter,
            is_layer_sparse=self.layer_scatter_modes.is_layer_sparse,
        )

    def _local_token_move_can_go_to_next_layer(self) -> bool:
        """Whether the next layer's input can run this layer's move of its FFN
        output back to this rank's tokens: the base postprocess scatter, outside
        an active LayerNorm SP region."""
        return self._postprocess_scatters_to_local_tokens and not (
            self._sp_variant is not None and get_forward().sp_active
        )

    def _reduce_scatter_step(
        self, forward_batch: ForwardBatch
    ) -> Optional[Callable[[torch.Tensor, torch.Tensor, ForwardBatch], None]]:
        """The reduce-scatter postprocess would run on this layer's FFN output;
        None when it runs only a scatter, or anything else."""
        return _reduce_and_redistribute_output_step(
            forward_batch,
            allow_reduce_scatter=self.allow_reduce_scatter,
            is_layer_sparse=self.layer_scatter_modes.is_layer_sparse,
        )

    def ffn_reduction_group(self) -> GroupCoordinator:
        """The group this layer's FFN output owes its sum over: the MoE output's
        group on a sparse layer, the TP group a dense MLP reduces over."""
        if self.layer_scatter_modes.is_layer_sparse:
            return post_experts_reduction_group()
        return get_parallel().tp_group

    def _select_ffn_completion(self, forward_batch: ForwardBatch) -> "FfnCompletion":
        """Decide once, before the FFN runs, what it skips and what completes its
        output: the next layer's input, or this layer's postprocess."""
        defer_moe_finalize = self.should_defer_moe_finalize(forward_batch)
        # Deferring implies fusing: a handoff skips the post-experts all-reduce.
        fuse_mlp_allreduce = defer_moe_finalize or self.should_defer_ffn_reduction(
            forward_batch
        )
        mlp_reduce_scatter = self.should_use_reduce_scatter(forward_batch)
        if fuse_mlp_allreduce:
            group = self.ffn_reduction_group()
            if self._postprocess_scatters_to_local_tokens:
                # Under attention DP the next layer also brings the sum back to
                # this rank's tokens.
                leave = partial(
                    UnreducedOutput,
                    reduce_and_redistribute=partial(
                        _all_reduce_then_to_local_tokens, group, forward_batch
                    ),
                )
            else:
                leave = partial(UnreducedOutput, group=group)
        elif not self.is_last_layer and self._local_token_move_can_go_to_next_layer():
            step = self._reduce_scatter_step(forward_batch)
            leave = (
                None
                if step is None
                else partial(
                    UnreducedOutput,
                    reduce_and_redistribute=partial(
                        _to_local_tokens, step, forward_batch
                    ),
                )
            )
        else:
            leave = None
        return FfnCompletion(
            defer_moe_finalize=defer_moe_finalize,
            fuse_mlp_allreduce=fuse_mlp_allreduce,
            mlp_reduce_scatter=mlp_reduce_scatter,
            leave=leave,
        )

    def ffn_exit(self, forward_batch: ForwardBatch) -> "FfnExit":
        """Decide once how this layer's FFN output reduction completes. Use the
        result as a context manager around the FFN call, then call ``finish``."""
        return FfnExit(self, forward_batch)

    def finish_layer_stack(
        self,
        hidden_states: Union[torch.Tensor, UnreducedOutput],
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Complete what this layer left for a next layer. Call it on the last
        layer of this rank before its output reaches the final norm, the next
        pipeline rank, or any other consumer outside the layers."""
        return reduce_output(hidden_states), residual

    def should_use_reduce_scatter(self, forward_batch: ForwardBatch):
        if not self.allow_reduce_scatter:
            return False
        if (
            self._communicate_summable_tensor_pair_fn
            is CommunicateSummableTensorPairFn._scatter_hidden_states
        ):
            if should_use_dp_reduce_scatterv():
                return True
            if (
                forward_batch.dp_padding_mode.is_max_len()
                and can_use_dp_reduce_scatter()
            ):
                return True
        # Prefill CP predicates must stay out of decode graph capture.
        if forward_batch.forward_mode.is_context_parallel_extend() and (
            dsa_use_prefill_cp(forward_batch) or is_mla_cp_active(forward_batch)
        ):
            return True
        if get_attn_tp_context().input_scattered and not self.is_last_layer:
            return True
        return False

    def should_defer_moe_finalize(
        self, forward_batch: ForwardBatch, m: int | None = None
    ) -> bool:
        """Whether the MoE may hand an unfinalized output to the next layer."""
        return False

    def _ffn_sum_can_move_to_next_layer(self) -> bool:
        # When MOE_FULL is active (moe_cp allgather), fusion must be disabled because
        # the fusion path skips postprocess_layer which contains the moe_cp scatter.
        # Without scatter, hidden_states remain at MOE_FULL size while residual is at
        # TP_ATTN_FULL size, causing a shape mismatch.
        if (
            is_enable_moe_cp_allgather()
            or self.layer_scatter_modes.mlp_mode == ScatterMode.MOE_FULL
        ):
            return False

        # The fused residual+LN reduces over a single group. Hybrid EP+TP spans
        # two disjoint groups; post_experts_all_reduce() merges them into one
        # _TP reduction when moe_dp_size == 1, which the fused kernel can absorb.
        # When merging is blocked, no single group covers both, so fusion stays off.
        parallel = get_parallel()
        if (
            parallel.moe_ep_size > 1
            and parallel.moe_tp_size > 1
            and not can_merge_post_experts_all_reduce()
        ):
            return False

        if (
            is_dp_attention_enabled()
            and self._speculative_algo is not None
            and self._speculative_algo.is_eagle()
        ):
            return False

        if get_attn_tp_context().input_scattered:
            return False

        # When mlp_mode is SCATTERED, the MLP runs on scattered data with no TP
        # all-reduce, so there is nothing to fuse with the next layer.
        return self.layer_scatter_modes.mlp_mode != ScatterMode.SCATTERED

    # NOTE: This function will cause torch recompilation
    def should_fuse_mlp_allreduce_with_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        if not self._ffn_sum_can_move_to_next_layer():
            return False
        batch_size = _batch_size(forward_batch)
        return (
            (
                apply_flashinfer_allreduce_fusion(batch_size)
                or (
                    _use_aiter
                    and batch_size > 0
                    and get_parallel().tp_size != 6
                    and not is_dp_attention_enabled()
                    and get_moe_a2a_backend().is_none()
                    and aiter_all_reduce_fusion_enabled_for(forward_batch.forward_mode)
                )
            )
            and (not self.is_last_layer)
            and (self._context.tp_size > 1)
        )

    def should_defer_ffn_reduction(self, forward_batch: ForwardBatch) -> bool:
        """Whether the FFN leaves its output's all-reduce to the next layer's
        input norm: whenever the fused kernel takes it, and otherwise when the
        next layer would run the same all-reduce the FFN itself would have."""
        if self.should_fuse_mlp_allreduce_with_next_layer(forward_batch):
            return True
        return (
            self._context.tp_size > 1
            and not self.is_last_layer
            and self._ffn_sum_can_move_to_next_layer()
            and _unfused_completion_matches_the_ffn(forward_batch)
            and not self.should_use_reduce_scatter(forward_batch)
            # Under attention DP the next layer must also run postprocess's
            # scatter back to this rank's tokens, and nothing more.
            and (
                not is_dp_attention_enabled()
                or (
                    self._local_token_move_can_go_to_next_layer()
                    and self._reduce_scatter_step(forward_batch) is None
                )
            )
        )


# MOE_FULL gathers across the MoE-CP group, which spans every CP rank when CP is on.
_SCATTER_MODE_SHARDED_AXES = {
    ScatterMode.SCATTERED: (
        TokenAxis.ATTN_DP,
        TokenAxis.ATTN_CP,
        TokenAxis.ATTN_TP_SCATTER,
    ),
    ScatterMode.TP_ATTN_FULL: (TokenAxis.ATTN_DP, TokenAxis.ATTN_CP),
    ScatterMode.FULL: (TokenAxis.ATTN_CP,),
    ScatterMode.MOE_FULL: (),
}


def scatter_mode_layouts(
    *, attn_dp_size: int, attn_cp_size: int, attn_tp_size: int
) -> Dict[ScatterMode, Layout]:
    axis_sizes = {
        TokenAxis.ATTN_DP: attn_dp_size,
        TokenAxis.ATTN_CP: attn_cp_size,
        TokenAxis.ATTN_TP_SCATTER: attn_tp_size,
    }
    return {
        mode: Layout.sharded_over(*axes, axis_sizes=axis_sizes)
        for mode, axes in _SCATTER_MODE_SHARDED_AXES.items()
    }


class FfnCompletion(msgspec.Struct, frozen=True):
    """One FFN's reduction decision. The flags are published while the FFN runs;
    ``leave`` wraps its output for the next layer's input to complete, or is None
    when this layer's postprocess completes it."""

    defer_moe_finalize: bool
    fuse_mlp_allreduce: bool
    mlp_reduce_scatter: bool
    leave: Optional[Callable[[torch.Tensor], UnreducedOutput]] = None


class FfnExit:
    """The scope that publishes an FfnCompletion while the FFN runs: inside the
    ``with`` block it is ``fuse_mlp_allreduce`` / ``mlp_reduce_scatter`` /
    ``defer_moe_finalize`` on ``get_forward()``."""

    __slots__ = (
        "communicator",
        "forward_batch",
        "defer_moe_finalize",
        "fuse_mlp_allreduce",
        "mlp_reduce_scatter",
        "_leave",
        "_scope",
    )

    def __init__(self, communicator: LayerCommunicator, forward_batch: ForwardBatch):
        self.communicator = communicator
        self.forward_batch = forward_batch
        completion = communicator._select_ffn_completion(forward_batch)
        self.defer_moe_finalize = completion.defer_moe_finalize
        self.fuse_mlp_allreduce = completion.fuse_mlp_allreduce
        self.mlp_reduce_scatter = completion.mlp_reduce_scatter
        self._leave = completion.leave
        self._scope = get_forward().scoped(
            fuse_mlp_allreduce=self.fuse_mlp_allreduce,
            mlp_reduce_scatter=self.mlp_reduce_scatter,
            defer_moe_finalize=self.defer_moe_finalize,
        )

    def __enter__(self) -> "FfnExit":
        self._scope.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._scope.__exit__(*exc_info)

    def finish(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, UnreducedOutput], torch.Tensor]:
        """Leave the reduction to the next layer's input, or postprocess."""
        if not isinstance(hidden_states, torch.Tensor):
            # A deferred MoE finalize handoff, consumed by the next prepare_attn.
            assert self.defer_moe_finalize, "unrequested deferred MoE handoff"
            return hidden_states, residual
        if self._leave is not None:
            return self._leave(hidden_states), residual
        return self.communicator.postprocess_layer(
            hidden_states, residual, self.forward_batch
        )


@dataclass
class CommunicateContext:
    process_group_sizes: Dict[ScatterMode, int]
    attn_tp_rank: int
    attn_tp_size: int
    attn_dp_size: int
    attn_cp_rank: int
    attn_cp_size: int
    tp_size: int
    cache = None
    tp_rank: int
    force_layernorm_before_dp_gather: bool = False

    def is_same_group_size(self, a: ScatterMode, b: ScatterMode):
        return self.process_group_sizes[a] == self.process_group_sizes[b]

    @cached_property
    def layouts(self) -> Dict[ScatterMode, Layout]:
        return scatter_mode_layouts(
            attn_dp_size=self.attn_dp_size,
            attn_cp_size=self.attn_cp_size,
            attn_tp_size=self.attn_tp_size,
        )

    def is_same_layout(self, a: ScatterMode, b: ScatterMode):
        return self.layouts[a] == self.layouts[b]

    @classmethod
    def init_new(cls):
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size
        attn_dp_size = get_parallel().attn_dp_size
        attn_cp_size = get_parallel().attn_cp_size
        attn_cp_rank = get_parallel().attn_cp_rank
        tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank
        moe_cp_size = get_moe_cp_size()
        process_group_sizes = {
            ScatterMode.SCATTERED: 1,
            ScatterMode.TP_ATTN_FULL: attn_tp_size,
            # TODO: support --moe-dense-tp-size > 1
            # With context parallel enabled, we should exclude
            # the attn_cp_size from the total tp_size
            ScatterMode.FULL: tp_size // attn_cp_size,
            ScatterMode.MOE_FULL: tp_size // (attn_cp_size // moe_cp_size),
        }
        return cls(
            process_group_sizes=process_group_sizes,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=attn_tp_size,
            attn_dp_size=attn_dp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )


class CommunicateSimpleFn:
    @staticmethod
    def get_fn(
        input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_layout(input_mode, output_mode):
            return CommunicateSimpleFn._trivial

        if (input_mode == ScatterMode.SCATTERED) and (
            output_mode == ScatterMode.TP_ATTN_FULL
        ):
            if _use_ag_after_qlora:
                return CommunicateSimpleFn._trivial
            return CommunicateSimpleFn._scattered_to_tp_attn_full

        raise NotImplementedError(f"{input_mode=} {output_mode=}")

    @staticmethod
    def _trivial(
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
    ) -> torch.Tensor:
        return hidden_states

    @staticmethod
    def _scattered_to_tp_attn_full(
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        forward_batch: ForwardBatch,
        context: CommunicateContext,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(hidden_states, tuple):
            gathered_hidden_states = []
            for local_hidden_states in hidden_states:
                with use_symmetric_memory(
                    get_parallel().tp_group,
                    disabled=not is_allocation_symmetric(),
                ):
                    output = torch.empty(
                        (
                            local_hidden_states.shape[0] * context.attn_tp_size,
                            *local_hidden_states.shape[1:],
                        ),
                        dtype=local_hidden_states.dtype,
                        device=local_hidden_states.device,
                    )
                attn_tp_all_gather_into_tensor(
                    output,
                    local_hidden_states,
                )
                gathered_hidden_states.append(output)
            return tuple(gathered_hidden_states)

        return _redistribute_from_attn_tp_shards(hidden_states)


def _redistribute_from_attn_tp_shards(tensor: torch.Tensor) -> torch.Tensor:
    gathered = get_local_dp_buffer(get_parallel().attn_tp_group)
    attn_tp_all_gather_into_tensor(gathered, tensor)
    return gathered


def _redistribute_to_attn_tp_shards(
    tensor: torch.Tensor, context: CommunicateContext
) -> torch.Tensor:
    return tensor.tensor_split(context.attn_tp_size)[context.attn_tp_rank]


def _reduce_and_redistribute_output_to_attn_tp_shards(
    hidden_states: torch.Tensor, context: CommunicateContext
) -> torch.Tensor:
    local_hidden_states = hidden_states.tensor_split(context.attn_tp_size)[
        context.attn_tp_rank
    ]
    attn_tp_reduce_scatter_tensor(local_hidden_states, hidden_states)
    return local_hidden_states


def _redistribute_input_to_moe_cp(
    hidden_states: torch.Tensor, forward_batch: ForwardBatch, moe_cp_size: int
) -> torch.Tensor:
    # Zigzag split can produce unequal token counts across CP ranks
    # (when seq_len % (cp_size * 2) != 0). NCCL allgather requires
    # equal input sizes, so pad to the max per-rank token count.
    per_rank_tokens = forward_batch.attn_cp_metadata.per_rank_actual_token
    max_tokens = max(per_rank_tokens)
    pad_size = max_tokens - hidden_states.shape[0]
    if pad_size > 0:
        hidden_states = torch.nn.functional.pad(hidden_states, [0, 0, 0, pad_size])

    output = torch.empty(
        (max_tokens * moe_cp_size, hidden_states.shape[1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    moe_cp_all_gather_into_tensor(output, hidden_states)
    return output


def _mlp_input_reduce_output(
    hidden_states: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    if (
        not forward_batch.forward_mode.is_decode_or_idle()
        and get_exec().comm.enable_quant_communications
    ):
        return attention_tensor_model_parallel_quant_all_reduce(hidden_states)
    return attention_tensor_model_parallel_all_reduce(hidden_states)


def _mlp_input_reduce_output_and_update_and_read_residual(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The attention-TP all-reduce, residual add and norm in one fused kernel;
    None when no fused kernel takes the batch."""
    if (
        apply_aiter_all_reduce_fusion(hidden_states, forward_batch)
        or apply_flashinfer_allreduce_fusion(hidden_states.shape[0])
    ) and hasattr(layernorm, "forward_with_allreduce_fusion"):
        hidden_states, residual = layernorm.forward_with_allreduce_fusion(
            hidden_states, residual, use_attn_tp_group=True
        )
        return hidden_states, residual
    return None


def _redistribute_input_to_dp(
    hidden_states: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    global_hidden_states = get_global_dp_buffer(get_parallel().tp_group)
    dp_gather_replicate(global_hidden_states, hidden_states, forward_batch)
    return global_hidden_states


def _reduce_and_redistribute_output_to_dp(
    hidden_states: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    global_hidden_states = get_global_dp_buffer(get_parallel().tp_group)
    dp_gather_partial(global_hidden_states, hidden_states, forward_batch)
    return global_hidden_states


class CommunicateWithAllReduceAndLayerNormFn:
    """Besides communication, needs to
    1. All reduce in tp_attn_group on hidden_states
    2. Apply layer norm
    """

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        hidden_states_output_mode: ScatterMode,
        residual_output_mode: ScatterMode,
        context: CommunicateContext,
    ):

        if (
            context.is_same_layout(hidden_states_input_mode, hidden_states_output_mode)
            and context.is_same_layout(residual_input_mode, residual_output_mode)
            and context.attn_tp_size == 1
        ):
            return CommunicateWithAllReduceAndLayerNormFn._simple

        if (
            hidden_states_input_mode == ScatterMode.SCATTERED
            and residual_input_mode == ScatterMode.SCATTERED
            and hidden_states_output_mode == ScatterMode.SCATTERED
            and residual_output_mode == ScatterMode.SCATTERED
        ):
            # Megatron LayerNorm sequence parallelism (layers/layernorm_sp.py):
            # activations stay sequence-sharded across the attn->mlp boundary, so
            # there is nothing to gather or scatter here -- just the residual add
            # plus LayerNorm on the local shard. The row-parallel o_proj already
            # issued the reduce-scatter (g-bar) that the all-reduce would have
            # done, and the g all-gather is fused into the next column-parallel
            # linear. Distinct from the branch above because under pure TP
            # attn_tp_size == tp_size > 1, so that gate does not fire.
            return CommunicateWithAllReduceAndLayerNormFn._simple

        if hidden_states_input_mode == ScatterMode.TP_ATTN_FULL and (
            residual_input_mode in (ScatterMode.SCATTERED, ScatterMode.TP_ATTN_FULL)
        ):
            fn = {
                (
                    ScatterMode.FULL,
                    ScatterMode.TP_ATTN_FULL,
                ): CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
                (
                    ScatterMode.MOE_FULL,
                    ScatterMode.TP_ATTN_FULL,
                ): CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual_moe,
                (
                    ScatterMode.SCATTERED,
                    ScatterMode.SCATTERED,
                ): CommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual,
            }.get((hidden_states_output_mode, residual_output_mode))
            if fn is not None:
                return partial(fn, residual_input_mode=residual_input_mode)

            if (
                hidden_states_output_mode == ScatterMode.TP_ATTN_FULL
                and residual_output_mode == ScatterMode.TP_ATTN_FULL
                and context.attn_tp_size > 1
            ):
                # Used when the dense MLP is tensor-parallelized along the
                # attention TP group (``moe_dense_tp_size > 1``): hidden states
                # need an all-reduce inside the attention TP group before the
                # next layernorm, while staying in TP_ATTN_FULL on both sides.
                return CommunicateWithAllReduceAndLayerNormFn._tp_attn_all_reduce_and_layernorm

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {hidden_states_output_mode=} {residual_output_mode=}"
        )

    @staticmethod
    def _simple(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
    ):
        # TODO move these `if shape != 0` into LayerNorm itself
        if hidden_states.shape[0] != 0:
            hidden_states, residual = layernorm(hidden_states, residual)
        return hidden_states, residual

    @staticmethod
    def _tp_attn_all_reduce_and_layernorm(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
    ):
        """All-reduce hidden states inside the attention TP group, then layernorm.

        Used when the dense MLP shares the attention TP group
        (``moe_dense_tp_size > 1``): both hidden states and residual stay in
        ``TP_ATTN_FULL`` across the boundary.
        """
        hidden_states = get_parallel().attn_tp_group.all_reduce(hidden_states)
        if hidden_states.shape[0] != 0:
            hidden_states, residual = layernorm(hidden_states, residual)
        return hidden_states, residual

    @staticmethod
    def _gather_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
    ):
        if get_attn_tp_context().input_scattered:
            return CommunicateWithAllReduceAndLayerNormFn._tp_all_reduce_with_scattered_residual(
                hidden_states,
                residual,
                layernorm,
                context,
            )

        if residual_input_mode == ScatterMode.SCATTERED and context.attn_tp_size > 1:
            residual = _redistribute_from_attn_tp_shards(residual)
        if context.attn_dp_size == 1:
            fused = _mlp_input_reduce_output_and_update_and_read_residual(
                hidden_states, residual, forward_batch, layernorm
            )
            if fused is not None:
                return fused
            hidden_states = _mlp_input_reduce_output(hidden_states, forward_batch)
            if _is_npu and context.cache is not None:
                _ = prepare_weight_cache(hidden_states, context.cache)
            return layernorm(hidden_states, residual)

        # Attention DP. Replicate: reduce, add and normalize locally, then gather.
        # Partial: one rank adds the residual, the gather sums it, then normalize.
        replicate = (
            context.force_layernorm_before_dp_gather or context.attn_tp_size == 1
        )
        if replicate and hidden_states.shape[0] != 0:
            if context.attn_tp_size > 1:
                hidden_states = attention_tensor_model_parallel_all_reduce(
                    hidden_states
                )
            with use_symmetric_memory(
                get_parallel().tp_group,
                disabled=not is_allocation_symmetric(),
            ):
                hidden_states, residual = layernorm(hidden_states, residual)
        elif context.attn_tp_rank == 0:
            hidden_states += residual
        if replicate:
            return _redistribute_input_to_dp(hidden_states, forward_batch), residual
        hidden_states = _reduce_and_redistribute_output_to_dp(
            hidden_states, forward_batch
        )
        dp_scatter(residual, hidden_states, forward_batch)
        if hidden_states.shape[0] != 0:
            hidden_states = layernorm(hidden_states)
        return hidden_states, residual

    @staticmethod
    def _scatter_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
    ):
        hidden_states = _reduce_and_redistribute_output_to_attn_tp_shards(
            hidden_states, context
        )
        if residual_input_mode == ScatterMode.TP_ATTN_FULL:
            residual = _redistribute_to_attn_tp_shards(residual, context)
        if hidden_states.shape[0] != 0:
            hidden_states, residual = layernorm(hidden_states, residual)
        return hidden_states, residual

    @staticmethod
    def _tp_all_reduce_with_scattered_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states

        scattered_states = hidden_states.tensor_split(context.tp_size)[context.tp_rank]
        scattered_states += residual
        residual = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = layernorm(residual)
        return hidden_states, residual

    @staticmethod
    def _gather_hidden_states_and_residual_moe(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
    ):
        """Allgather tokens for MoE when moe_dp_size < attn_cp_size.

        Steps:
          1. Standard attn-TP all-reduce + optional DP allgather + layernorm (same as
             _gather_hidden_states_and_residual for the dp>1 case, or simple all-reduce
             + layernorm for dp==1).
          2. moe_cp allgather: gather tokens from cp_per_moe CP ranks so each rank holds
             all tokens for its MoE group.

        Residual is left at TP_ATTN_FULL throughout.
        """
        # Early return on empty tensor is safe for MOE_CP because:
        # - During CP extend: zigzag split guarantees all CP ranks have non-zero tokens,
        #   so no rank hits this path while others proceed to the allgather.
        # - During decode: moe_cp allgather is skipped (guarded by is_context_parallel_extend).
        # - CUDA graph warmup: not applicable when --cuda-graph-backend-prefill=disabled is used.
        if hidden_states.shape[0] == 0:
            return hidden_states, residual

        # Step 1: Standard all-reduce/DP-allgather + layernorm (reuse existing logic).
        hidden_states, residual = (
            CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
                layernorm=layernorm,
                context=context,
                residual_input_mode=residual_input_mode,
            )
        )

        # Step 2: moe_cp allgather — gather across cp_per_moe CP ranks.
        # Only active during prefill (context-parallel extend); decode keeps existing path.
        moe_cp_size = get_moe_cp_size()
        if (
            moe_cp_size > 1
            and hidden_states.shape[0] > 0
            and forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
        ):
            hidden_states = _redistribute_input_to_moe_cp(
                hidden_states, forward_batch, moe_cp_size
            )

        return hidden_states, residual


def _dp_scatter_group() -> GroupCoordinator:
    parallel = get_parallel()
    if parallel.tp_size == parallel.attn_dp_size:
        return parallel.tp_group
    return parallel.attn_tp_group


def _reduce_and_redistribute_output_varlen(
    local_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> None:
    get_parallel().tp_group.reduce_scatterv(
        hidden_states,
        output=local_hidden_states,
        sizes=get_dp_global_num_tokens(),
    )


def _reduce_and_redistribute_output_max_len(
    local_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> None:
    dp_reduce_scatter_tensor(local_hidden_states, hidden_states)


def _redistribute_output(
    local_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> None:
    dp_scatter(local_hidden_states, hidden_states, forward_batch)


def _redistribute_output_from_moe_cp(
    hidden_states: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    moe_cp_rank = get_moe_cp_rank()
    # The allgather was padded to max_tokens_per_rank (equal chunks).
    # Extract this rank's actual (non-padded) tokens from its chunk.
    per_rank_tokens = forward_batch.attn_cp_metadata.per_rank_actual_token
    max_tokens_per_rank = max(per_rank_tokens)
    actual_local_tokens = per_rank_tokens[moe_cp_rank]
    return hidden_states.narrow(
        0, moe_cp_rank * max_tokens_per_rank, actual_local_tokens
    ).contiguous()


def _reduce_and_redistribute_output_step(
    forward_batch: ForwardBatch, *, allow_reduce_scatter: bool, is_layer_sparse: bool
) -> Optional[Callable[[torch.Tensor, torch.Tensor, ForwardBatch], None]]:
    """The reduce-scatter that brings a FULL-layout layer output back to this
    rank's tokens under attention DP when the FFN left its sum to it; None when
    the FFN reduced the output and only a scatter remains."""
    # A MoE block leaves its sum to reduce_scatterv whenever it applies
    # (should_skip_post_experts_all_reduce); a dense MLP does only under the
    # published mlp_reduce_scatter, which needs allow_reduce_scatter.
    if should_use_dp_reduce_scatterv() and (allow_reduce_scatter or is_layer_sparse):
        return _reduce_and_redistribute_output_varlen
    if (
        allow_reduce_scatter
        and forward_batch.dp_padding_mode.is_max_len()
        and can_use_dp_reduce_scatter()
    ):
        return _reduce_and_redistribute_output_max_len
    return None


def _to_local_tokens(
    step: Callable[[torch.Tensor, torch.Tensor, ForwardBatch], None],
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    local_hidden_states = get_local_dp_buffer(_dp_scatter_group())
    step(local_hidden_states, hidden_states, forward_batch)
    return local_hidden_states


def _all_reduce_then_to_local_tokens(
    group: GroupCoordinator, forward_batch: ForwardBatch, hidden_states: torch.Tensor
) -> torch.Tensor:
    return _to_local_tokens(
        _redistribute_output, forward_batch, group.all_reduce(hidden_states)
    )


class CommunicateSummableTensorPairFn:
    """It is allowed to make (hidden_states, residual) := (hidden_states + residual, None) if needed."""

    @classmethod
    def execute(
        cls,
        hidden_states_input_mode,
        residual_input_mode,
        output_mode,
        context,
        **kwargs,
    ):
        return cls.get_fn(
            hidden_states_input_mode=hidden_states_input_mode,
            residual_input_mode=residual_input_mode,
            output_mode=output_mode,
            context=context,
        )(context=context, **kwargs)

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_layout(
            hidden_states_input_mode, output_mode
        ) and context.is_same_layout(residual_input_mode, output_mode):
            return CommunicateSummableTensorPairFn._trivial

        fn = {
            (
                ScatterMode.FULL,
                ScatterMode.TP_ATTN_FULL,
                ScatterMode.TP_ATTN_FULL,
            ): CommunicateSummableTensorPairFn._scatter_hidden_states,
            (
                ScatterMode.SCATTERED,
                ScatterMode.SCATTERED,
                ScatterMode.TP_ATTN_FULL,
            ): CommunicateSummableTensorPairFn._gather,
            (
                ScatterMode.TP_ATTN_FULL,
                ScatterMode.TP_ATTN_FULL,
                ScatterMode.SCATTERED,
            ): CommunicateSummableTensorPairFn._scatter,
            (
                ScatterMode.MOE_FULL,
                ScatterMode.TP_ATTN_FULL,
                ScatterMode.TP_ATTN_FULL,
            ): CommunicateSummableTensorPairFn._scatter_hidden_states_moe,
        }.get((hidden_states_input_mode, residual_input_mode, output_mode))
        if fn is not None:
            return fn

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}"
        )

    @staticmethod
    def _trivial(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        **kwargs,
    ):
        return hidden_states, residual

    @staticmethod
    def _scatter_hidden_states(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        allow_reduce_scatter: bool = False,
        is_layer_sparse: bool = False,
    ):
        local_hidden_states = get_local_dp_buffer(_dp_scatter_group())
        step = (
            _reduce_and_redistribute_output_step(
                forward_batch,
                allow_reduce_scatter=allow_reduce_scatter,
                is_layer_sparse=is_layer_sparse,
            )
            or _redistribute_output
        )
        step(local_hidden_states, hidden_states, forward_batch)
        return local_hidden_states, residual

    @staticmethod
    def _gather(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        **kwargs,
    ):
        hidden_states += residual
        return _redistribute_from_attn_tp_shards(hidden_states), None

    @staticmethod
    def _scatter(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
    ):
        assert residual is None, "not yet handled residual!=None"
        return _redistribute_to_attn_tp_shards(hidden_states, context), None

    @staticmethod
    def _scatter_hidden_states_moe(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        **kwargs,
    ):
        """Scatter MoE output back to TP_ATTN_FULL after MOE_FULL computation.

        After moe_tensor_model_parallel_all_reduce (which runs unconditionally since
        mlp_reduce_scatter=False for this path), all ranks in the moe_cp group hold the
        full MoE result for all cp_per_moe token chunks. We simply slice out this rank's
        CP-local portion.

        If DP>1, further scatter back to the local DP slice.
        """
        # Only scatter back during prefill; decode was never allgathered so no-op.
        # Safe w.r.t. empty tensors: same reasoning as _gather_hidden_states_and_residual_moe
        # — CP extend always has non-zero tokens per rank, and decode skips this path.
        moe_cp_size = get_moe_cp_size()
        if (
            moe_cp_size > 1
            and forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
        ):
            hidden_states = _redistribute_output_from_moe_cp(
                hidden_states, forward_batch
            )

        if context.attn_dp_size > 1:
            hidden_states = _to_local_tokens(
                _redistribute_output, forward_batch, hidden_states
            )

        return hidden_states, residual
