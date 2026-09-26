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
from typing import Callable, Dict, List, Optional, Protocol, Tuple, Union

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
from sglang.srt.layers.boundary_layout import (
    DecoderLayerSides,
    EdgeDecl,
    Layout,
    StageInput,
    StageOutput,
    SumGroup,
    TokenAxis,
    decoder_layer_edges,
    decoder_layer_sides,
    input_scattered_layer_sides,
    scattered_residual_layer_sides,
    sequence_parallel_layer_sides,
)
from sglang.srt.layers.communicator_dsa_cp import (
    dsa_cp_gather_hidden_states,
    dsa_cp_reduce_scatter_hidden_states,
)
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
    is_moe_input_scattered_across_dp_ranks,
    post_experts_reduction_group,
    post_experts_sum_is_one_all_reduce,
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
    # The model's first layer: its input is the embedding, not a layer output.
    is_first_layer: bool = False
    # The model's last layer: its output goes to the final norm, not a next layer.
    is_last_layer: bool = False
    # Whether the layer before this one has a sparse MLP; None when the modes
    # were given directly, not planned from the layer sequence.
    is_previous_layer_sparse: Optional[bool] = None
    # Whether the layer after this one has a sparse MLP; None likewise.
    is_next_layer_sparse: Optional[bool] = None

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
            is_first_layer=context.layer_id == 0,
            is_last_layer=context.layer_id == context.num_layers - 1,
            is_previous_layer_sparse=context.is_previous_layer_sparse,
            is_next_layer_sparse=context.is_next_layer_sparse,
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
            # MLA/DSA CP models gather over the attention-CP group instead.
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


class ResidualOps(Protocol):
    """How a layer writes each stage's output into its residual and reads the
    next stage's input from it. The boundary steps complete sums and move
    tokens around these operations. AddAndNorm adds and normalizes; MHC's
    hyper-connection streams implement the same operations (communicator_mhc)."""

    # The write-back is a plain add, which one rank may run before the sum it
    # adds into completes: the DP partial order, and a residual that joins the
    # attention output's sum.
    adds_plainly: bool
    # The layer writes its FFN output into the residual itself instead of
    # leaving that to the next layer's input.
    updates_residual_after_ffn: bool

    def enter(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The residual the layer stack starts from, given its input."""

    def read_attention_input(self, residual, norm, quant_format: str) -> Tuple:
        """The attention input and the residual, from a residual that already
        holds the previous layer's output."""

    def update_and_read_attention_input(
        self, hidden_states, residual, norm, quant_format: str, post_residual_addition
    ) -> Tuple:
        """Write the previous layer's output into the residual, then read the
        attention input from it."""

    def update_and_read_ffn_input(self, hidden_states, residual, norm) -> Tuple:
        """Write the attention output into the residual, then read the FFN
        input from it."""

    def update_residual(self, hidden_states, residual) -> torch.Tensor:
        """The residual with the FFN output written into it."""

    def residual_to_attn_tp_shard(self, residual, context) -> torch.Tensor:
        """This attention-TP rank's slice of the residual."""

    def residual_from_attn_tp_shards(self, residual) -> torch.Tensor:
        """The residual gathered from every attention-TP rank's slice."""


class AddAndNorm:
    """A plain residual: a stage's output is added into it, and the next
    stage's input is its norm. An empty batch skips the norms."""

    adds_plainly = True
    updates_residual_after_ffn = False

    def enter(self, hidden_states):
        return hidden_states

    def read_attention_input(self, residual, norm, quant_format):
        if residual.shape[0] == 0:
            return residual, residual
        return _attn_input_update_and_read_residual(quant_format)(
            norm, residual, None, None
        )

    def update_and_read_attention_input(
        self, hidden_states, residual, norm, quant_format, post_residual_addition
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states
        return _attn_input_update_and_read_residual(quant_format)(
            norm, hidden_states, residual, post_residual_addition
        )

    def update_and_read_ffn_input(self, hidden_states, residual, norm):
        if hidden_states.shape[0] == 0:
            return hidden_states, residual
        return norm(hidden_states, residual)

    def update_residual(self, hidden_states, residual):
        hidden_states += residual
        return hidden_states

    def residual_to_attn_tp_shard(self, residual, context):
        return _redistribute_to_attn_tp_shards(residual, context)

    def residual_from_attn_tp_shards(self, residual):
        return _redistribute_from_attn_tp_shards(residual)


ADD_AND_NORM = AddAndNorm()


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


def _ffn_has_tokens(forward_batch: ForwardBatch) -> bool:
    if is_dp_attention_enabled():
        # The FFN runs on every DP rank's tokens, so every rank decides alike.
        return (getattr(forward_batch, "global_dp_buffer_len", None) or 0) > 0
    return _batch_size(forward_batch) > 0


def _unfused_completion_matches_the_ffn(forward_batch: ForwardBatch) -> bool:
    """Whether the next layer's all-reduce is the one the FFN would have run
    itself, as the MoE declares it (post_experts_sum_is_one_all_reduce), on a
    batch the FFN runs."""
    return _ffn_has_tokens(forward_batch) and post_experts_sum_is_one_all_reduce()


class LayerCommunicator:
    # Communicators built without __init__ (e.g. test doubles) publish no LoRA layout.
    _publish_lora_layout: bool = False
    # Whether this class's boundary steps may be chosen from both sides'
    # declarations; a subclass that picks its own steps says no.
    _takes_declared_boundaries = True
    # A plain residual unless the layer is built with its own.
    _residual_ops: ResidualOps = ADD_AND_NORM

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
        # False for a layer whose FFN always completes its own reduction.
        allow_deferred_ffn_reduction: bool = True,
        # How the layer writes its residual and reads its stages' inputs.
        residual_ops: ResidualOps = ADD_AND_NORM,
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
        self.allow_deferred_ffn_reduction = allow_deferred_ffn_reduction
        self._residual_ops = residual_ops

        self._context = CommunicateContext.init_new()
        self._context.force_layernorm_before_dp_gather = (
            force_layernorm_before_dp_gather
        )
        # The fused kernels every batch's attention input tries first.
        self._attn_input_fusions = self._select_attn_input_fusions()
        # The steps the layer's ordinary batches run.
        sides = self._declared_sides()
        self._declared = sides
        self._steps = (
            self._steps_from_declarations(
                sides,
                fusions=self._select_mlp_input_fusions(),
                force_layernorm_before_gather=force_layernorm_before_dp_gather,
            )
            if sides is not None
            else self._select_boundaries_from_scatter_modes()
        )
        # The steps a batch that shards its tokens over attention CP runs; None
        # without CP or where the steps come from the scatter modes.
        self._cp_steps = (
            self._steps_from_declarations(
                self._declared_sides(cp_active=True),
                fusions=self._select_mlp_input_fusions(),
                force_layernorm_before_gather=force_layernorm_before_dp_gather,
                cp_moves=_cp_moves(),
            )
            if sides is not None and get_parallel().attn_cp_size > 1
            else None
        )
        # The steps a batch with input-scattered attention runs; None where it
        # cannot run or the steps come from the scatter modes.
        self._input_scattered_steps = (
            self._steps_for_input_scattered(sides)
            if sides is not None and self._input_can_be_scattered()
            else None
        )
        self._speculative_algo = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        # LoRA kernels need the per-layer token layout only under DP attention.
        self._publish_lora_layout = get_parallel().enable_dp_attention and bool(
            get_lora().enable_lora
        )

        # Under LayerNorm SP, the steps the layer runs while the region is
        # active; None without SP. The two are exclusive: SP runs a model without
        # q_lora, which input-scattered attention needs.
        self._sp_steps = (
            _select_boundary_steps(
                sequence_parallel_layer_sides(axis_sizes=_token_axis_sizes()),
                residual_ops=residual_ops,
                attention_fusions=self._attn_input_fusions,
                enters_stack=self.layer_scatter_modes.is_first_layer,
            )
            if layernorm_sp.layernorm_sp_enabled()
            else None
        )

    @property
    def input_rows(self) -> Layout:
        """The rows the layer's input and residual arrive on in a batch that
        runs its ordinary steps."""
        if self._declared is not None:
            return self._declared.input_rows
        return self._context.layouts[self.layer_scatter_modes.layer_input_mode]

    def _input_can_be_scattered(self) -> bool:
        """Whether a batch may run this layer with input-scattered attention:
        configured, on pure TP without an a2a backend or a dense MLP on every
        rank. The rest of what ``AttnTpContext.init_context`` requires is only
        known once the model is built."""
        parallel = get_parallel()
        return (
            parallel.enable_attn_tp_input_scattered
            and parallel.tp_size > 1
            and parallel.attn_dp_size == 1
            and parallel.attn_cp_size == 1
            and get_moe_a2a_backend().is_none()
            and not enable_moe_dense_fully_dp()
        )

    def _declared_sides(
        self, *, cp_active: bool = False
    ) -> Optional[DecoderLayerSides]:
        """The declarations this layer's boundaries are chosen from: an
        attention and an FFN, in a layer whose previous layer is known, with
        attention CP only for a prefill CP whose FFN gathers over the whole CP
        group. None when the steps come from the scatter modes. ``cp_active``
        gives a batch that shards its tokens over CP; the others hold every
        token on each CP rank. An active LayerNorm SP region and input-scattered
        attention have their own."""
        modes = self.layer_scatter_modes
        parallel = get_parallel()
        # A MoE dispatched per DP shard computes on this rank's local rows and
        # hands its layer's output on there; so does a dense MLP on every rank.
        moe_on_local_rows = is_moe_input_scattered_across_dp_ranks()
        dense_on_local_rows = enable_moe_dense_fully_dp()

        def on_local_rows(sparse: bool) -> bool:
            return moe_on_local_rows if sparse else dense_on_local_rows

        def gathers_for_tbo(sparse: bool, next_sparse: Optional[bool]) -> bool:
            # Under two-batch overlap a dense layer on local rows hands the
            # sparse layer after it, where the split happens, the attention's
            # rows.
            return (
                dense_on_local_rows
                and get_exec().overlap.enable_two_batch_overlap
                and not sparse
                and bool(next_sparse)
            )

        if not (
            self._takes_declared_boundaries
            and (parallel.attn_cp_size == 1 or _cp_on_declarations())
            # MoE layers under attention DP and GQA prefill CP keep the
            # scatter-mode steps.
            and not (
                parallel.attn_cp_size > 1
                and parallel.attn_dp_size > 1
                and modes.is_layer_sparse
                and not _gathers_over_attention_cp()
            )
            and (modes.is_first_layer or modes.is_previous_layer_sparse is not None)
        ):
            return None
        if parallel.attn_cp_size > 1 and _cp_moves().reduce_scatter is not None:
            # A CP extend's FFN may leave its sum to the reduce-scatter that
            # takes each rank's shard back (DSA and MLA CP).
            may_leave = not cp_active
            may_leave_to_reduce_scatter = True
        else:
            # Otherwise under CP the FFN completes its own sum: the next layer's
            # input holds only this rank's chunk, and a reduce-scatter back over
            # attention DP would split across the CP ranks.
            may_leave = may_leave_to_reduce_scatter = parallel.attn_cp_size == 1
        return decoder_layer_sides(
            axis_sizes=_token_axis_sizes(cp_active=cp_active),
            ffn_on_local_rows=on_local_rows(modes.is_layer_sparse),
            previous_on_local_rows=(
                not modes.is_first_layer
                and on_local_rows(modes.is_previous_layer_sparse)
                and not gathers_for_tbo(
                    modes.is_previous_layer_sparse, modes.is_layer_sparse
                )
            ),
            is_last_layer=modes.is_last_layer,
            hands_on_attention_rows=gathers_for_tbo(
                modes.is_layer_sparse, modes.is_next_layer_sparse
            ),
            attention_gathers_local_rows=_use_ag_after_qlora,
            ffn_group=SumGroup.MOE_OUTPUT if modes.is_layer_sparse else SumGroup.TP,
            leaves_for_next_layer=self.allow_deferred_ffn_reduction and may_leave,
            leaves_for_reduce_scatter=self.allow_reduce_scatter
            and may_leave_to_reduce_scatter,
            # A MoE block leaves its sum to reduce_scatterv whenever that combine
            # applies (should_skip_post_experts_all_reduce).
            leaves_for_reduce_scatterv=(
                self.allow_reduce_scatter or modes.is_layer_sparse
            )
            and may_leave,
        )

    def _select_boundaries_from_scatter_modes(self) -> "BoundarySteps":
        """The layer's steps chosen from its scatter modes, for the layers the
        declarations do not cover yet."""
        attention_input, postprocess = self._post_init_communicate()
        ffn_input, fused = self._select_mlp_input()
        modes = self.layer_scatter_modes
        ops = self._residual_ops
        ffn_rows = self._context.layouts[modes.mlp_mode]
        ffn_output = StageOutput(
            ffn_rows,
            group=SumGroup.MOE_OUTPUT if modes.is_layer_sparse else SumGroup.TP,
            leaves_for_next_layer=self.allow_deferred_ffn_reduction,
            leaves_for_reduce_scatter=self.allow_reduce_scatter,
            # A MoE block leaves its sum to reduce_scatterv whenever it applies
            # (should_skip_post_experts_all_reduce); a dense MLP does only under
            # the published mlp_reduce_scatter.
            leaves_for_reduce_scatterv=(
                self.allow_reduce_scatter or modes.is_layer_sparse
            ),
        )
        return BoundarySteps(
            attention_input=attention_input,
            ffn_input=ffn_input,
            ffn_input_rows=ffn_rows,
            ffn_output=ffn_output,
            # Under attention DP the base postprocess scatters the FFN output
            # back to this rank's tokens, which the next layer's input can run
            # instead; the MHC and DSA-CP postprocess do more.
            ffn_output_move=(
                None
                if postprocess is CommunicateSummableTensorPairFn._scatter_hidden_states
                else partial(postprocess, residual_ops=ops)
            ),
            # Not when the way back is the MoE-CP scatter, nor when a SCATTERED
            # FFN computes whole tokens with nothing left to sum.
            ffn_sum_is_movable=modes.mlp_mode
            not in (ScatterMode.MOE_FULL, ScatterMode.SCATTERED),
            fused=fused,
            attention_prepare=partial(
                _attention_input_step,
                layer_input=_complete_scattered_input,
                fusions=self._attn_input_fusions,
                enters_stack=modes.is_first_layer,
                residual_ops=ops,
            ),
        )

    def _steps_for_input_scattered(self, sides: DecoderLayerSides) -> "BoundarySteps":
        """The steps a batch with input-scattered attention runs at this layer,
        for the layer's ordinary declarations ``sides``. A plain residual comes
        back to the full rows inside the attention output's sum; one whose
        write-back is not a plain add stays on each rank's slice."""
        if self._residual_ops.adds_plainly:
            scattered = input_scattered_layer_sides(
                axis_sizes=_token_axis_sizes(),
                ffn_group=sides.ffn_output.group,
                hands_on_partial=self.allow_reduce_scatter and not self.is_last_layer,
            )
            handoff = _hand_qkv_hook_its_input
        else:
            scattered = scattered_residual_layer_sides(
                axis_sizes=_token_axis_sizes(),
                ffn_group=sides.ffn_output.group,
                is_first_layer=self.layer_scatter_modes.is_first_layer,
                is_last_layer=self.is_last_layer,
                leaves_for_reduce_scatter=self.allow_reduce_scatter,
            )
            # DSA and hook-less attention take the slice gathered.
            handoff = _hand_scattered_input_to_attention
        return _select_boundary_steps(
            scattered,
            residual_ops=self._residual_ops,
            attention_handoff=handoff,
            attention_fusions=self._attn_input_fusions,
            enters_stack=self.layer_scatter_modes.is_first_layer,
        )

    def _steps_from_declarations(
        self, sides: DecoderLayerSides, **kwargs
    ) -> "BoundarySteps":
        """The steps this layer runs for a set of declarations."""
        return _select_boundary_steps(
            sides,
            residual_ops=self._residual_ops,
            attention_fusions=self._attn_input_fusions,
            enters_stack=self.layer_scatter_modes.is_first_layer,
            **kwargs,
        )

    def _post_init_communicate(self) -> Tuple[Callable, Callable]:
        """The attention input move and the postprocess the scatter modes
        choose."""
        if _generic_prefill_cp_shards_tokens() and _gathers_over_attention_cp():
            # These tables have no attention-CP gather.
            raise NotImplementedError(
                "a DSA or MLA prefill CP layer outside the declarations"
            )
        return (
            CommunicateSimpleFn.get_fn(
                input_mode=self.layer_scatter_modes.layer_input_mode,
                output_mode=self.layer_scatter_modes.attn_mode,
                context=self._context,
            ),
            CommunicateSummableTensorPairFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,
                residual_input_mode=self.layer_scatter_modes.middle_residual_mode,
                output_mode=self.layer_scatter_modes.layer_output_mode,
                context=self._context,
            ),
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
            gathered_last_layer_output = self._batch_steps(
                forward_batch
            ).attention_input(
                hidden_states=residual,
                forward_batch=forward_batch,
                context=self._context,
            )
            if (
                gathered_last_layer_output is residual
                # An accumulator that copies on append already holds a snapshot.
                and not getattr(captured_last_layer_outputs, "copies_on_append", False)
                and not self._post_attn_residual_is_read_only(residual, forward_batch)
            ):
                gathered_last_layer_output = residual.clone()
            captured_last_layer_outputs.append(gathered_last_layer_output)
        return hidden_states, residual

    def _post_attn_residual_is_read_only(
        self, residual: torch.Tensor, forward_batch: ForwardBatch
    ) -> bool:
        """True if ``prepare_mlp``'s post-attention RMSNorm leaves ``residual``
        untouched, so Eagle3 aux capture can keep its reference and skip the clone.

        Of the base fused entry's kernels only flashinfer's writes a fresh
        ``residual_out`` (see ``flashinfer_allreduce_residual_rmsnorm``); the aiter
        kernel and every plain norm fold into ``residual`` in place. It runs when
        ``prepare_mlp`` selected a fused entry that may return a new residual and
        the batch is not input-scattered.
        """
        return (
            any(
                f.may_return_new_residual
                for f in self._batch_steps(forward_batch).fused
            )
            and not get_attn_tp_context().input_scattered
            and apply_flashinfer_allreduce_fusion(residual.shape[0])
        )

    def publish_attn_lora_layout(self) -> None:
        """Attention consumes the DP-local token batch."""
        if self._publish_lora_layout:
            get_forward().set("lora_batch_layout", LoRABatchLayout.DP_LOCAL)

    def publish_mlp_lora_layout(self, steps: "BoundarySteps") -> None:
        """The MLP consumes the TP-global batch only when the batch's FFN input
        is gathered over attention DP."""
        if self._publish_lora_layout:
            get_forward().set(
                "lora_batch_layout",
                (
                    LoRABatchLayout.DP_LOCAL
                    if TokenAxis.ATTN_DP in steps.ffn_input_rows.sharded
                    else LoRABatchLayout.TP_GLOBAL
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
        # The SP region opens at the first layer, re-evaluated per forward so a
        # crash mid-loop cannot leak into the next one. It sets what the batch's
        # steps are chosen from, and the first layer's input owes nothing.
        if self._sp_steps is not None and self.layer_scatter_modes.is_first_layer:
            get_forward().set(
                "sp_active", layernorm_sp.runs_sp(forward_batch.forward_mode)
            )
            if get_forward().sp_active:
                hidden_states = layernorm_sp.sp_entry_scatter(hidden_states)
        hidden_states, residual = self._batch_steps(forward_batch).attention_prepare(
            hidden_states,
            residual,
            forward_batch,
            self.input_layernorm,
            self._context,
            quant_format=quant_format,
            post_residual_addition=post_residual_addition,
        )
        return self._finish_prepare_attn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

    def _in_sp_region(self) -> bool:
        return self._sp_steps is not None and get_forward().sp_active

    def _batch_steps(self, forward_batch: ForwardBatch) -> "BoundarySteps":
        """The steps this batch runs: an active LayerNorm SP region's,
        input-scattered attention's, a CP extend's, or the layer's ordinary
        ones."""
        if self._in_sp_region():
            return self._sp_steps
        if (
            self._input_scattered_steps is not None
            and get_attn_tp_context().input_scattered
        ):
            return self._input_scattered_steps
        if self._cp_steps is not None and _batch_shards_over_cp(forward_batch):
            return self._cp_steps
        return self._steps

    def _finish_prepare_attn(self, hidden_states, residual, forward_batch):
        """Tail every prepare_attn path must run, or ``attn_inputs`` is unset."""
        steps = self._batch_steps(forward_batch)
        hidden_states = steps.attention_input(
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            context=self._context,
        )
        hidden_states = steps.attention_handoff(
            hidden_states, forward_batch, self.qkv_latent_func
        )
        return hidden_states, residual

    def _select_attn_input_fusions(self) -> Tuple[Callable, ...]:
        """The fused kernels that complete what the previous layer left together
        with the residual update and the input norm, in the order they are tried.
        Each takes (owed, residual, forward_batch, post_residual_addition) and
        returns None when it does not take the batch. They add the residual
        plainly."""
        if not (
            self._residual_ops.adds_plainly
            and hasattr(self.input_layernorm, "forward_with_allreduce_fusion")
        ):
            return ()
        self._attn_input_fuses_quant = (
            self.enable_fused_ar_quant
            and _use_aiter
            and hasattr(
                self.input_layernorm, "forward_with_allreduce_fusion_quant_per_group"
            )
        )
        return (self._reduce_output_and_update_and_read_residual,)

    def _reduce_output_and_update_and_read_residual(
        self,
        owed: UnreducedOutput,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        post_residual_addition: Optional[torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Complete the sum the previous layer left, add it to the residual and
        apply the input norm in one aiter or flashinfer kernel; None when the
        kernel does not take this batch. The result is not quantized for
        ``quant_format``."""
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
            apply_aiter_all_reduce_fusion(hidden_states, forward_batch)
            or apply_flashinfer_allreduce_fusion(hidden_states.shape[0])
        ):
            return None
        if self._attn_input_fuses_quant:
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

    def _select_mlp_input(self) -> Tuple[Callable, Tuple["FusedMlpInput", ...]]:
        """The attention-TP -> FFN boundary's steps, chosen from the layouts, and
        the fused kernels they try first."""
        kind = mlp_input_kind(self.layer_scatter_modes, self._context)
        residual_input_mode = self.layer_scatter_modes.layer_input_mode
        ops = self._residual_ops
        if kind is MlpInputKind.NORM:
            return partial(_mlp_input_norm, residual_ops=ops), ()
        if kind is MlpInputKind.ATTN_TP_ALL_REDUCE:
            return partial(_mlp_input_attn_tp_all_reduce, residual_ops=ops), ()
        if kind is MlpInputKind.SCATTER:
            return (
                partial(
                    _mlp_input_scatter,
                    scatters_residual=residual_input_mode == ScatterMode.TP_ATTN_FULL,
                    residual_ops=ops,
                ),
                (),
            )
        # Neither fused kernel runs under attention DP.
        fusions = (
            self._select_mlp_input_fusions() if self._context.attn_dp_size == 1 else ()
        )
        steps = partial(
            _mlp_input_gather,
            order=_mlp_input_order(
                self._context,
                residual_input_mode,
                tuple(f.run for f in fusions),
                residual_ops=ops,
            ),
        )
        if kind is MlpInputKind.GATHER_MOE_CP:
            steps = partial(_mlp_input_gather_moe_cp, gather=steps)
        return steps, fusions

    def _select_mlp_input_fusions(self) -> Tuple["FusedMlpInput", ...]:
        """The fused kernels that can take the attention -> FFN steps, in the
        order they are tried. They add the residual plainly."""
        if not (
            self._residual_ops.adds_plainly
            and hasattr(self.post_attention_layernorm, "forward_with_allreduce_fusion")
        ):
            return ()
        return (
            FusedMlpInput(
                completes=SumGroup.ATTN_TP,
                run=self._mlp_input_reduce_output_and_update_and_read_residual,
                may_return_new_residual=True,
            ),
        )

    def _mlp_input_reduce_output_and_update_and_read_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """The attention-TP all-reduce, residual add and norm in one aiter or
        flashinfer kernel; None when neither takes the batch."""
        if not (
            apply_aiter_all_reduce_fusion(hidden_states, forward_batch)
            or apply_flashinfer_allreduce_fusion(hidden_states.shape[0])
        ):
            return None
        return self.post_attention_layernorm.forward_with_allreduce_fusion(
            hidden_states, residual, use_attn_tp_group=True
        )

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        cache=None,
    ):
        steps = self._batch_steps(forward_batch)
        self.publish_mlp_lora_layout(steps)
        if cache is not None:
            self._context.cache = cache

        return steps.ffn_input(
            hidden_states,
            residual,
            forward_batch,
            self.post_attention_layernorm,
            self._context,
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
        return self._complete_ffn_output_now(
            hidden_states,
            residual,
            forward_batch=forward_batch,
            dp_step=self._postprocess_dp_step(forward_batch),
        )

    def _local_token_move_can_go_to_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        """Whether the next layer's input can run this layer's move of its FFN
        output back to this rank's tokens: the base postprocess scatter, when
        the next layer's input also writes the output into the residual."""
        return (
            self._batch_steps(forward_batch).returns_over_dp
            and not self._residual_ops.updates_residual_after_ffn
        )

    def _postprocess_dp_step(
        self, forward_batch: ForwardBatch
    ) -> Optional[Callable[[torch.Tensor, torch.Tensor, ForwardBatch], None]]:
        """The reduce-scatter that brings this layer's FFN output back to this
        rank's tokens under attention DP; None when the base postprocess would
        only scatter, or does not move tokens."""
        steps = self._batch_steps(forward_batch)
        if not steps.returns_over_dp:
            return None
        return _reduce_and_redistribute_output_step(
            forward_batch,
            leaves_for_reduce_scatter=steps.ffn_output.leaves_for_reduce_scatter,
            leaves_for_reduce_scatterv=steps.ffn_output.leaves_for_reduce_scatterv,
        )

    def _ffn_leaves_sum_to_reduce_scatter(
        self, forward_batch: ForwardBatch, dp_step: Optional[Callable]
    ) -> bool:
        """Whether the FFN leaves its sum out because a reduce-scatter completes
        it: the attention-DP one ``dp_step`` names, or the CP / input-scattered
        one."""
        steps = self._batch_steps(forward_batch)
        if not steps.ffn_output.leaves_for_reduce_scatter:
            return False
        if dp_step is not None or steps.ffn_output_move_completes_sum:
            return True
        # The scatter-mode steps of a DSA or MLA CP extend (the subclasses that
        # pick their own steps). Prefill CP predicates must stay out of decode
        # graph capture.
        if (
            self._cp_steps is None
            and forward_batch.forward_mode.is_context_parallel_extend()
            and (dsa_use_prefill_cp(forward_batch) or is_mla_cp_active(forward_batch))
        ):
            return True
        return get_attn_tp_context().input_scattered and not self.is_last_layer

    def ffn_reduction_group(self, forward_batch: ForwardBatch) -> GroupCoordinator:
        """The group this layer's FFN output owes its sum over: the MoE output's
        group on a sparse layer, the TP group a dense MLP reduces over."""
        return _sum_group(self._batch_steps(forward_batch).ffn_output.group)

    def _select_ffn_completion(self, forward_batch: ForwardBatch) -> "FfnCompletion":
        """Decide once, before the FFN runs, what it skips and what completes its
        output: the next layer's input, or this layer's postprocess step."""
        dp_step = self._postprocess_dp_step(forward_batch)
        mlp_reduce_scatter = self._ffn_leaves_sum_to_reduce_scatter(
            forward_batch, dp_step
        )
        complete_now = partial(
            self._complete_ffn_output_now,
            forward_batch=forward_batch,
            dp_step=dp_step,
        )
        steps = self._batch_steps(forward_batch)
        if not steps.ffn_output.leaves_for_next_layer:
            return FfnCompletion(
                defer_moe_finalize=False,
                fuse_mlp_allreduce=False,
                mlp_reduce_scatter=mlp_reduce_scatter,
                complete=complete_now,
            )
        defer_moe_finalize = self.should_defer_moe_finalize(forward_batch)
        # Deferring implies fusing: a handoff skips the post-experts all-reduce.
        fuse_mlp_allreduce = defer_moe_finalize or self._ffn_sum_moves_to_next_layer(
            forward_batch, mlp_reduce_scatter=mlp_reduce_scatter, dp_step=dp_step
        )
        if fuse_mlp_allreduce:
            group = self.ffn_reduction_group(forward_batch)
            if steps.returns_over_dp:
                # Under attention DP the next layer also brings the sum back to
                # this rank's tokens.
                wrap = partial(
                    UnreducedOutput,
                    reduce_and_redistribute=partial(
                        _all_reduce_then_to_local_tokens, group, forward_batch
                    ),
                )
            else:
                wrap = partial(UnreducedOutput, group=group)
            complete = partial(_leave_to_next_layer, wrap)
        elif (
            dp_step is not None
            and not self.is_last_layer
            and self._local_token_move_can_go_to_next_layer(forward_batch)
        ):
            complete = partial(
                _leave_to_next_layer,
                partial(
                    UnreducedOutput,
                    reduce_and_redistribute=partial(
                        _to_local_tokens, dp_step, forward_batch
                    ),
                ),
            )
        else:
            complete = complete_now
        return FfnCompletion(
            defer_moe_finalize=defer_moe_finalize,
            fuse_mlp_allreduce=fuse_mlp_allreduce,
            mlp_reduce_scatter=mlp_reduce_scatter,
            complete=complete,
        )

    def _complete_ffn_output_now(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        *,
        forward_batch: ForwardBatch,
        dp_step: Optional[Callable],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This layer's postprocess, run with the attention-DP step already
        chosen: the move back to where the next layer reads the FFN output, then
        the write-back into the residual for a layer that does it itself."""
        steps = self._batch_steps(forward_batch)
        if steps.returns_over_dp:
            hidden_states = _to_local_tokens(
                dp_step or _redistribute_output, forward_batch, hidden_states
            )
        else:
            hidden_states, residual = steps.ffn_output_move(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
                context=self._context,
                allow_reduce_scatter=self.allow_reduce_scatter,
                is_layer_sparse=self.layer_scatter_modes.is_layer_sparse,
            )
        if residual is not None and self._residual_ops.updates_residual_after_ffn:
            hidden_states = self._residual_ops.update_residual(hidden_states, residual)
            residual = None
        return hidden_states, residual

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
        """Whether the FFN leaves its sum to a reduce-scatter, for layers that run
        their FFN outside ffn_exit."""
        return self._ffn_leaves_sum_to_reduce_scatter(
            forward_batch, self._postprocess_dp_step(forward_batch)
        )

    def should_defer_moe_finalize(
        self, forward_batch: ForwardBatch, m: int | None = None
    ) -> bool:
        """Whether the MoE may hand an unfinalized output to the next layer."""
        return False

    def _ffn_sum_can_move_to_next_layer(self, forward_batch: ForwardBatch) -> bool:
        # Under the MoE-CP all-gather the fusion path would skip postprocess_layer
        # and its MoE-CP scatter, leaving hidden_states longer than the residual.
        if (
            is_enable_moe_cp_allgather()
            or not self._batch_steps(forward_batch).ffn_sum_is_movable
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

        return not get_attn_tp_context().input_scattered

    # NOTE: This function will cause torch recompilation
    def should_fuse_mlp_allreduce_with_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        if not self._ffn_sum_can_move_to_next_layer(forward_batch):
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

    def _ffn_sum_moves_to_next_layer(
        self,
        forward_batch: ForwardBatch,
        *,
        mlp_reduce_scatter: bool,
        dp_step: Optional[Callable],
    ) -> bool:
        """Whether the FFN leaves its output's all-reduce to the next layer's
        input norm: whenever the fused kernel takes it, and otherwise when the
        next layer would run the same all-reduce the FFN itself would have."""
        if self.should_fuse_mlp_allreduce_with_next_layer(forward_batch):
            return True
        return (
            self._context.tp_size > 1
            and not self.is_last_layer
            and self._ffn_sum_can_move_to_next_layer(forward_batch)
            and _unfused_completion_matches_the_ffn(forward_batch)
            and not mlp_reduce_scatter
            # Under attention DP the next layer must also run postprocess's
            # scatter back to this rank's tokens, and nothing more.
            and (
                not is_dp_attention_enabled()
                or (
                    self._local_token_move_can_go_to_next_layer(forward_batch)
                    and dp_step is None
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
    ``complete(hidden_states, residual)`` then either wraps the output for the
    next layer's input to complete, or runs this layer's postprocess step."""

    defer_moe_finalize: bool
    fuse_mlp_allreduce: bool
    mlp_reduce_scatter: bool
    complete: Callable[[torch.Tensor, torch.Tensor], Tuple]


def _leave_to_next_layer(
    wrap: Callable[[torch.Tensor], UnreducedOutput],
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> Tuple[UnreducedOutput, torch.Tensor]:
    return wrap(hidden_states), residual


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
        "_complete",
        "_scope",
    )

    def __init__(self, communicator: LayerCommunicator, forward_batch: ForwardBatch):
        self.communicator = communicator
        self.forward_batch = forward_batch
        completion = communicator._select_ffn_completion(forward_batch)
        self.defer_moe_finalize = completion.defer_moe_finalize
        self.fuse_mlp_allreduce = completion.fuse_mlp_allreduce
        self.mlp_reduce_scatter = completion.mlp_reduce_scatter
        self._complete = completion.complete
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
        return self._complete(hidden_states, residual)


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
    gathered = get_local_dp_buffer(
        get_parallel().attn_tp_group, hidden_size=tensor.shape[-1]
    )
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


def moe_cp_gathered_rows(forward_batch: ForwardBatch) -> Optional[List[int]]:
    """The real rows each MoE-CP rank contributes when this batch's FFN input is
    gathered across the MoE-CP group, or None when it is not: only a context
    parallel extend with CP metadata gathers, and only when the group has more
    than one rank. The batch is read first, so a batch that is not a CP extend
    never reads the group."""
    if (
        forward_batch.forward_mode.is_context_parallel_extend()
        and forward_batch.attn_cp_metadata is not None
        and get_moe_cp_size() > 1
    ):
        return forward_batch.attn_cp_metadata.per_rank_actual_token
    return None


def _redistribute_input_to_moe_cp(
    hidden_states: torch.Tensor, rows: List[int], moe_cp_size: int
) -> torch.Tensor:
    # Zigzag split can produce unequal token counts across CP ranks
    # (when seq_len % (cp_size * 2) != 0). NCCL allgather requires
    # equal input sizes, so pad to the max per-rank token count.
    max_tokens = max(rows)
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


def _cp_shard_token_rows(forward_batch: ForwardBatch) -> List[int]:
    """Rows of each CP rank's shard that hold tokens; the shards are padded to
    one length after them."""
    metadata = forward_batch.attn_cp_metadata
    return metadata.per_rank_logical_token or metadata.per_rank_actual_token


def _redistribute_input_to_dp(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    cp_shard_counts: Optional[List[int]] = None,
) -> torch.Tensor:
    global_hidden_states = get_global_dp_buffer(get_parallel().tp_group)
    dp_gather_replicate(
        global_hidden_states, hidden_states, forward_batch, cp_shard_counts
    )
    return global_hidden_states


def _reduce_and_redistribute_output_to_dp(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    cp_shard_counts: Optional[List[int]] = None,
) -> torch.Tensor:
    global_hidden_states = get_global_dp_buffer(get_parallel().tp_group)
    dp_gather_partial(
        global_hidden_states, hidden_states, forward_batch, cp_shard_counts
    )
    return global_hidden_states


class FusedMlpInput(msgspec.Struct, frozen=True):
    """A kernel that completes the sum the attention output owes together with
    the residual add and the post-attention norm, in that order.

    ``run(hidden_states, residual, forward_batch)`` returns the FFN's input and
    the residual, or None when it does not take the batch; it returns None only
    before touching its inputs or starting a collective."""

    # The group whose sum it completes.
    completes: SumGroup
    run: Callable[..., Optional[Tuple[torch.Tensor, torch.Tensor]]]
    # It may hand back a new residual and leave the one it took unchanged.
    may_return_new_residual: bool


def _mlp_input_without_dp(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    gathers_residual: bool,
    fusions: Tuple[Callable, ...],
    residual_ops: ResidualOps = ADD_AND_NORM,
):
    if gathers_residual:
        residual = residual_ops.residual_from_attn_tp_shards(residual)
    for fused in fusions:
        result = fused(hidden_states, residual, forward_batch)
        if result is not None:
            return result
    hidden_states = _mlp_input_reduce_output(hidden_states, forward_batch)
    if _is_npu and context.cache is not None:
        _ = prepare_weight_cache(hidden_states, context.cache)
    return residual_ops.update_and_read_ffn_input(hidden_states, residual, layernorm)


def _mlp_input_dp_replicate(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    gathers_residual: bool,
    reduces_attention_tp: bool,
    places_cp_shards: bool = False,
    residual_ops: ResidualOps = ADD_AND_NORM,
):
    """Attention DP: complete the attention-TP sum if it is owed, write the
    output into the residual and read the FFN input locally, then gather. With
    ``places_cp_shards`` each CP rank puts its shard of the DP group's tokens
    beside the others' in the group's slot."""
    if gathers_residual:
        residual = residual_ops.residual_from_attn_tp_shards(residual)
    if hidden_states.shape[0] != 0:
        if reduces_attention_tp:
            hidden_states = attention_tensor_model_parallel_all_reduce(hidden_states)
        with use_symmetric_memory(
            get_parallel().tp_group,
            disabled=not is_allocation_symmetric(),
        ):
            hidden_states, residual = residual_ops.update_and_read_ffn_input(
                hidden_states, residual, layernorm
            )
    else:
        hidden_states, residual = residual_ops.update_and_read_ffn_input(
            hidden_states, residual, layernorm
        )
    cp_shard_counts = _cp_shard_token_rows(forward_batch) if places_cp_shards else None
    return (
        _redistribute_input_to_dp(hidden_states, forward_batch, cp_shard_counts),
        residual,
    )


def _mlp_input_dp_partial(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    gathers_residual: bool,
    places_cp_shards: bool = False,
):
    """Attention DP: one rank adds the residual, the gather sums it, then
    normalize. With ``places_cp_shards`` each CP rank puts its shard of the DP
    group's tokens beside the others' in the group's slot."""
    if gathers_residual:
        residual = _redistribute_from_attn_tp_shards(residual)
    if context.attn_tp_rank == 0:
        hidden_states += residual
    cp_shard_counts = _cp_shard_token_rows(forward_batch) if places_cp_shards else None
    hidden_states = _reduce_and_redistribute_output_to_dp(
        hidden_states, forward_batch, cp_shard_counts
    )
    dp_scatter(residual, hidden_states, forward_batch, cp_shard_counts)
    if hidden_states.shape[0] != 0:
        hidden_states = layernorm(hidden_states)
    return hidden_states, residual


def _mlp_input_order(
    context: CommunicateContext,
    residual_input_mode: ScatterMode,
    fusions: Tuple[Callable, ...],
    residual_ops: ResidualOps = ADD_AND_NORM,
) -> Callable:
    """The steps from the attention output to the FFN input, chosen from facts
    fixed at construction."""
    gathers_residual = (
        residual_input_mode == ScatterMode.SCATTERED and context.attn_tp_size > 1
    )
    if context.attn_dp_size == 1:
        return partial(
            _mlp_input_without_dp,
            gathers_residual=gathers_residual,
            fusions=fusions,
            residual_ops=residual_ops,
        )
    if (
        context.force_layernorm_before_dp_gather
        or context.attn_tp_size == 1
        or not residual_ops.adds_plainly
    ):
        return partial(
            _mlp_input_dp_replicate,
            gathers_residual=gathers_residual,
            reduces_attention_tp=context.attn_tp_size > 1,
            residual_ops=residual_ops,
        )
    return partial(_mlp_input_dp_partial, gathers_residual=gathers_residual)


def _sum_group(group: SumGroup) -> GroupCoordinator:
    parallel = get_parallel()
    if group is SumGroup.ATTN_TP:
        return parallel.attn_tp_group
    if group is SumGroup.TP:
        return parallel.tp_group
    return post_experts_reduction_group()


def _token_axis_sizes(*, cp_active: bool = False) -> Dict[TokenAxis, int]:
    """The token axes' sizes for a batch: attention CP shards tokens only on
    a CP extend (``cp_active``); otherwise every CP rank holds them all."""
    parallel = get_parallel()
    return {
        TokenAxis.ATTN_DP: parallel.attn_dp_size,
        TokenAxis.ATTN_CP: parallel.attn_cp_size if cp_active else 1,
        TokenAxis.ATTN_TP_SCATTER: parallel.attn_tp_size,
    }


def tbo_split_moves(layer_input_rows: Layout) -> Tuple[Callable, Callable]:
    """The moves around the two-batch-overlap split, which cuts the attention's
    rows: from the rows the first overlapped layer takes to the attention's,
    and back again for each half."""
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=_token_axis_sizes()
    )
    pair = CommunicateSummableTensorPairFn
    if layer_input_rows == attention:
        return pair._trivial, pair._trivial
    if layer_input_rows.sharded - attention.sharded == {TokenAxis.ATTN_TP_SCATTER}:
        # Each rank's slice: write the residual in and gather over attention
        # TP, then take the slice of each half.
        return pair._gather, pair._scatter
    raise NotImplementedError(f"{layer_input_rows=}")


def _cp_on_declarations() -> bool:
    """Whether attention CP is one the declarations cover: a prefill CP that
    shards tokens, with DSA or MLA attention, or with the FFN input gathered
    over a MoE-CP group that is the whole CP group."""
    return _generic_prefill_cp_shards_tokens() and (
        _gathers_over_attention_cp() or get_parallel().moe_dp_size == 1
    )


def _gathers_over_attention_cp() -> bool:
    """Whether a CP extend gathers the FFN input over the attention-CP group in
    equal shards and takes the output back with a reduce-scatter there: DSA and
    MLA CP. GQA prefill CP gathers over the MoE-CP group instead."""
    return is_dsa_enable_prefill_cp() or is_mla_cp_enabled()


def _batch_shards_over_cp(forward_batch: ForwardBatch) -> bool:
    """Whether this batch's tokens are split across the CP ranks. Only a context
    parallel extend is, so other batches, decode graph capture among them,
    never read the CP predicates."""
    if not forward_batch.forward_mode.is_context_parallel_extend():
        return False
    if _gathers_over_attention_cp():
        return dsa_use_prefill_cp(forward_batch) or is_mla_cp_active(forward_batch)
    return moe_cp_gathered_rows(forward_batch) is not None


class CpMoves(msgspec.Struct, frozen=True):
    """How a CP extend's rows reach an FFN that needs all of them and come back,
    chosen once for the kind of prefill CP: ``gather`` gathers the FFN input
    after each rank has completed its own block, ``take_back`` returns this
    rank's block of a complete output, and ``reduce_scatter``, where there is
    one, completes a sum left over the ranks of ``reduce_scatter_group()`` and
    returns the block in the same collective."""

    gather: Callable
    take_back: Callable
    reduce_scatter: Optional[Callable] = None
    reduce_scatter_group: Optional[Callable[[], GroupCoordinator]] = None


def _cp_moves() -> CpMoves:
    """DSA and MLA CP gather equal shards over the attention-CP group and can
    complete a sum over it. GQA prefill CP gathers blocks padded to the longest
    over the MoE-CP group and takes back only a complete output."""
    if _gathers_over_attention_cp():
        return CpMoves(
            gather=_mlp_input_gather_attention_cp,
            take_back=CommunicateSummableTensorPairFn._take_back_attention_cp_shard,
            reduce_scatter=CommunicateSummableTensorPairFn._reduce_scatter_over_cp,
            reduce_scatter_group=lambda: get_parallel().attn_cp_group,
        )
    return CpMoves(
        gather=_mlp_input_gather_moe_cp,
        take_back=CommunicateSummableTensorPairFn._scatter_hidden_states_moe,
    )


def _same_ranks(a: GroupCoordinator, b: GroupCoordinator) -> bool:
    return sorted(a.ranks) == sorted(b.ranks)


def _hand_qkv_hook_its_input(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    qkv_latent_func: Optional[Callable],
) -> torch.Tensor:
    """Give the attention's QKV hook the attention input."""
    if qkv_latent_func is not None:
        get_attn_tp_context().set_attn_inputs(
            AttentionInputs(hidden_states, forward_batch, qkv_latent_func)
        )
    return hidden_states


def _hand_scattered_input_to_attention(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    qkv_latent_func: Optional[Callable],
) -> torch.Tensor:
    """Input-scattered attention takes each rank's slice, and its QKV hook
    gathers the rows after the projection. DSA and attention without a hook
    consume full hidden states, so those are gathered here, and the hook is
    told they are."""
    ctx = get_attn_tp_context()
    if ctx.is_dsa or qkv_latent_func is None:
        hidden_states = _tp_all_gather_scattered_rows(hidden_states, forward_batch)
    if qkv_latent_func is not None:
        ctx.set_attn_inputs(
            AttentionInputs(
                hidden_states,
                forward_batch,
                qkv_latent_func,
                is_pre_gathered=ctx.is_dsa,
            )
        )
    return hidden_states


def _tp_all_gather_scattered_rows(
    hidden_states: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    # Input-scattered attention keeps the same number of tokens on every TP rank.
    total_tokens = forward_batch.input_ids.shape[0]
    output = hidden_states.new_empty((total_tokens, hidden_states.shape[-1]))
    get_parallel().tp_group.all_gather_into_tensor(output, hidden_states)
    return output


class BoundarySteps(msgspec.Struct, frozen=True):
    """The steps a batch runs at a layer's boundaries: into the attention,
    from the attention output to the FFN input, and the FFN output on to the
    rows the layer hands on."""

    # The half into the attention: completes what the input owes, writes the
    # previous output into the residual and reads the attention input
    # (_attention_input_step); attention_input then moves it.
    attention_prepare: Callable
    attention_input: Callable
    ffn_input: Callable
    # The rows ffn_input hands the FFN.
    ffn_input_rows: Layout
    # What the FFN exit reads: the FFN output's group and what it may leave.
    ffn_output: StageOutput
    # The postprocess that moves the FFN output on; None when it goes back over
    # attention DP, whose step the FFN exit and postprocess choose per batch.
    ffn_output_move: Optional[Callable]
    # Whether the next layer's input can take the FFN's sum.
    ffn_sum_is_movable: bool
    # Whether ffn_output_move also completes the sum the FFN leaves.
    ffn_output_move_completes_sum: bool = False
    # The fused kernels ffn_input tries first.
    fused: Tuple["FusedMlpInput", ...] = ()
    # Hands the attention its input once attention_input has moved it:
    # (hidden_states, forward_batch, qkv_latent_func) -> hidden_states.
    attention_handoff: Callable = _hand_qkv_hook_its_input

    @property
    def returns_over_dp(self) -> bool:
        return self.ffn_output_move is None


def _attention_input_step(
    hidden_states: Union[torch.Tensor, "UnreducedOutput"],
    residual: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    context: "CommunicateContext",
    *,
    quant_format: str,
    post_residual_addition: Optional[torch.Tensor],
    layer_input: Optional[Callable],
    fusions: Tuple[Callable, ...],
    enters_stack: bool,
    residual_ops: ResidualOps,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A boundary's half into an attention: complete what the previous layer
    left (a value that owes a sum, or a producer's handoff one of ``fusions``
    consumes with the add and norm), what the input owes by construction
    (``layer_input``), then write the previous output into the residual and read
    the attention input with ``norm``. The layer stack's first layer
    (``enters_stack``) starts its residual from its input."""
    enters = residual is None and enters_stack
    owed = None if isinstance(hidden_states, torch.Tensor) else hidden_states
    if owed is not None and residual is None:
        raise RuntimeError(f"{type(owed).__name__} requires residual input")
    if isinstance(owed, UnreducedOutput) and owed.reduce_and_redistribute is not None:
        # No fused kernel runs under attention DP: the reduce-scatter back to
        # this rank's tokens comes first.
        hidden_states, owed = reduce_output(owed), None
    if owed is not None:
        for fused in fusions:
            result = fused(owed, residual, forward_batch, post_residual_addition)
            if result is not None:
                return result
        hidden_states = owed.partial
    if layer_input is not None:
        hidden_states, residual = layer_input(hidden_states, residual, context)
    if enters:
        hidden_states, residual = residual_ops.enter(hidden_states), None
    if owed is not None and hidden_states.shape[0] != 0:
        hidden_states = reduce_output(owed)
    if residual is None:
        # The previous layer already wrote its output into the residual.
        return residual_ops.read_attention_input(hidden_states, norm, quant_format)
    return residual_ops.update_and_read_attention_input(
        hidden_states, residual, norm, quant_format, post_residual_addition
    )


class InputRead(Enum):
    """How a boundary's consumer reads its input from the residual: with the
    attention input norm (prepare_attn) or with the FFN input norm and its
    fused kernels (prepare_mlp)."""

    ATTENTION = auto()
    FFN = auto()


class Boundary(msgspec.Struct, frozen=True):
    """The steps one layer runs at one boundary, chosen from both sides'
    declarations. A layer runs the consumer's half of a boundary into one of
    its stages, and the producer's half of the boundary after its last stage;
    the neighbouring layer runs the other half of that one."""

    edge: EdgeDecl
    # The consumer's half: completing what the input owes, the add and the
    # norm; into the FFN also the move onto the rows it needs.
    prepare: Optional[Callable] = None
    # Into an attention, the move onto its rows after prepare.
    input_move: Optional[Callable] = None
    # The fused kernels an FFN's prepare tries first.
    fused: Tuple["FusedMlpInput", ...] = ()
    # The producer's half: the postprocess that moves the output onto the rows
    # the layer hands on; None when it goes back over attention DP, whose step
    # the FFN exit and postprocess choose per batch.
    output_move: Optional[Callable] = None
    # Whether output_move also completes the sum the producer leaves.
    output_move_completes_sum: bool = False

    @property
    def input_rows(self) -> Layout:
        """The rows the consumer is handed: what it needs, still sharded over
        the axes it gathers itself."""
        need = self.edge.need
        return Layout(
            need.layout.sharded
            | (self.edge.produced.layout.sharded & need.gathers_itself)
        )


def make_boundary(
    edge: EdgeDecl,
    *,
    reads: Optional[InputRead],
    fusions: Tuple = (),
    force_layernorm_before_gather: bool = False,
    cp_moves: Optional[CpMoves] = None,
    residual_ops: ResidualOps = ADD_AND_NORM,
    enters_stack: bool = False,
) -> Boundary:
    """The steps a layer runs at ``edge``, around the residual operations;
    ``cp_moves`` for an edge that gathers over or returns across attention CP.
    ``reads`` is how the consumer reads its input, or None when the consumer
    runs in the next layer and ``edge.need`` is the rows this layer hands on:
    then only the producer's half runs here. ``fusions`` are the fused kernels
    the consumer tries first (FusedMlpInput into an FFN, the attention input's
    entries into an attention); ``enters_stack`` for the edge into the layer
    stack's first attention. The consumer's half reads only this edge's
    declarations, never what the producer chose for a batch; a sum left for a
    batch arrives with the value."""
    if reads is None:
        if edge.need.layout != edge.residual_to:
            raise NotImplementedError(f"{edge=}")
        returns_over_dp, output_move, completes_sum = _select_ffn_output_move(
            edge.produced,
            residual=edge.residual,
            to=edge.residual_to,
            cp_moves=cp_moves,
            residual_ops=residual_ops,
        )
        return Boundary(
            edge,
            output_move=None if returns_over_dp else output_move,
            output_move_completes_sum=completes_sum,
        )
    if reads is InputRead.FFN:
        input_step, fused = _select_ffn_input(
            edge.produced,
            residual=edge.residual,
            residual_to=edge.residual_to,
            need=edge.need,
            force_layernorm_before_gather=force_layernorm_before_gather,
            fusions=fusions,
            residual_joins_sum=edge.residual_joins_sum,
            cp_moves=cp_moves,
            residual_ops=residual_ops,
        )
        return Boundary(edge, prepare=input_step, fused=fused)
    layer_input = None
    if edge.produced.always_leaves:
        # A reduce-scatter completes the TP sum onto each rank's slice, which the
        # attention takes: its group is the TP group without attention DP or CP.
        if (
            edge.produced.group is not SumGroup.TP
            or edge.residual.sharded
            or TokenAxis.ATTN_TP_SCATTER not in edge.need.gathers_itself
            or edge.residual_to.sharded != {TokenAxis.ATTN_TP_SCATTER}
        ):
            raise NotImplementedError(f"{edge=}")
        layer_input = tp_reduce_scatter
    return Boundary(
        edge,
        prepare=partial(
            _attention_input_step,
            layer_input=layer_input,
            fusions=fusions,
            enters_stack=enters_stack,
            residual_ops=residual_ops,
        ),
        input_move=_select_attention_input_move(edge.residual_to, edge.need),
    )


def _select_boundary_steps(
    sides: DecoderLayerSides,
    *,
    fusions: Tuple["FusedMlpInput", ...] = (),
    force_layernorm_before_gather: bool = False,
    cp_moves: Optional[CpMoves] = None,
    residual_ops: ResidualOps = ADD_AND_NORM,
    attention_handoff: Callable = _hand_qkv_hook_its_input,
    attention_fusions: Tuple[Callable, ...] = (),
    enters_stack: bool = False,
) -> BoundarySteps:
    """The steps of a decoder layer: its three boundaries, each chosen from the
    declarations of its two sides, around the layer's residual operations. Both
    edges that cross attention CP take the same ``cp_moves``; the attention's
    input tries ``attention_fusions``, and the layer stack's first layer
    (``enters_stack``) starts its residual there."""
    edges = decoder_layer_edges(sides)
    out_of_ffn = make_boundary(
        edges.out_of_ffn, reads=None, cp_moves=cp_moves, residual_ops=residual_ops
    )
    into_ffn = make_boundary(
        edges.into_ffn,
        reads=InputRead.FFN,
        fusions=fusions,
        force_layernorm_before_gather=force_layernorm_before_gather,
        cp_moves=cp_moves,
        residual_ops=residual_ops,
    )
    into_attention = make_boundary(
        edges.into_attention,
        reads=InputRead.ATTENTION,
        fusions=attention_fusions,
        residual_ops=residual_ops,
        enters_stack=enters_stack,
    )
    return BoundarySteps(
        attention_prepare=into_attention.prepare,
        attention_input=into_attention.input_move,
        ffn_input=into_ffn.prepare,
        ffn_input_rows=into_ffn.input_rows,
        ffn_output=edges.out_of_ffn.produced,
        ffn_output_move=out_of_ffn.output_move,
        ffn_output_move_completes_sum=out_of_ffn.output_move_completes_sum,
        ffn_sum_is_movable=edges.out_of_ffn.produced.group is not None,
        fused=into_ffn.fused,
        attention_handoff=attention_handoff,
    )


def _complete_scattered_input(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    context: "CommunicateContext",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The scatter-mode path's input completion: with input-scattered
    attention the layer's input is a TP partial that a reduce-scatter completes
    onto this rank's slice."""
    if get_attn_tp_context().input_scattered:
        return tp_reduce_scatter(hidden_states, residual, context)
    return hidden_states, residual


def _select_attention_input_move(rows: Layout, need: StageInput) -> Callable:
    """How the rows a layer takes become its attention's input: as they are,
    or gathered over attention TP from each rank's slice, unless the attention
    gathers them itself."""
    gathered = rows.sharded - need.layout.sharded - need.gathers_itself
    if not need.layout.sharded <= rows.sharded or gathered not in (
        frozenset(),
        {TokenAxis.ATTN_TP_SCATTER},
    ):
        raise NotImplementedError(f"{rows=} {need=}")
    if gathered:
        return CommunicateSimpleFn._scattered_to_tp_attn_full
    return CommunicateSimpleFn._trivial


def _select_ffn_input(
    produced: StageOutput,
    *,
    residual: Layout,
    residual_to: Layout,
    need: StageInput,
    force_layernorm_before_gather: bool,
    fusions: Tuple[FusedMlpInput, ...],
    residual_joins_sum: bool = False,
    cp_moves: Optional[CpMoves] = None,
    residual_ops: ResidualOps = ADD_AND_NORM,
) -> Tuple[Callable, Tuple[FusedMlpInput, ...]]:
    """The steps from the attention output to the FFN input, and the fused
    kernels they try first: complete the attention-TP sum, move the residual to
    the rows it has while the FFN runs, write the output into it and read the
    FFN input, and bring the rows to what the FFN's group needs: a gather over
    attention DP, or each rank's own slice. A kernel in ``fusions`` is tried
    only when nothing is gathered or sliced, and only if it completes the sum
    the attention output owes. A write-back that is not a plain add runs only
    after the sum completes."""
    # What the attention output owes decides the steps: the attention-TP sum,
    # always left by the output projection, or nothing.
    owes_attention_tp = produced.group is SumGroup.ATTN_TP
    if owes_attention_tp != produced.always_leaves or produced.group not in (
        None,
        SumGroup.ATTN_TP,
    ):
        raise NotImplementedError(f"{produced=}")
    gathered = produced.layout.sharded - need.layout.sharded - need.gathers_itself
    sliced = need.layout.sharded - produced.layout.sharded
    if sliced:
        # Each attention-TP rank takes its own slice: the reduce-scatter
        # completes the attention-TP sum and slices in one collective.
        if (
            sliced != {TokenAxis.ATTN_TP_SCATTER}
            or gathered
            or not owes_attention_tp
            or residual_to != need.layout
            or residual not in (produced.layout, need.layout)
        ):
            raise NotImplementedError(f"{produced=} {residual=} {need=}")
        return (
            partial(
                _mlp_input_scatter,
                scatters_residual=residual != residual_to,
                residual_ops=residual_ops,
            ),
            (),
        )
    if residual_to.sharded - produced.layout.sharded == {TokenAxis.ATTN_TP_SCATTER}:
        # The residual stays on each rank's slice while the FFN takes the
        # attention's rows (MHC on an input-scattered batch).
        if gathered or not owes_attention_tp or residual != residual_to:
            raise NotImplementedError(f"{produced=} {residual=} {need=}")
        return partial(_mlp_input_on_residual_shard, residual_ops=residual_ops), ()
    if gathered == {TokenAxis.ATTN_CP}:
        # Each CP rank completes its own chunk, then the CP moves gather them.
        if cp_moves is None:
            raise NotImplementedError(f"{produced=} {need=}")
        on_chunk, fused = _select_ffn_input(
            produced,
            residual=residual,
            residual_to=residual_to,
            need=StageInput(produced.layout),
            force_layernorm_before_gather=force_layernorm_before_gather,
            fusions=fusions,
            residual_joins_sum=residual_joins_sum,
            residual_ops=residual_ops,
        )
        return partial(cp_moves.gather, gather=on_chunk), fused
    if (
        residual_to != produced.layout
        or gathered
        not in (
            frozenset(),
            {TokenAxis.ATTN_DP},
            {TokenAxis.ATTN_DP, TokenAxis.ATTN_CP},
        )
        or residual.sharded - residual_to.sharded
        not in (frozenset(), {TokenAxis.ATTN_TP_SCATTER})
    ):
        raise NotImplementedError(f"{produced=} {residual=} {need=}")
    # A residual arriving on each rank's slice is gathered back first.
    gathers_residual = residual != residual_to
    if not gathered:
        if not owes_attention_tp:
            return partial(_mlp_input_norm, residual_ops=residual_ops), ()
        if gathers_residual and residual_joins_sum:
            # Each rank adds its slice of the residual into its share of the
            # sum, so the all-reduce also brings the residual back to every row.
            if not residual_ops.adds_plainly:
                raise NotImplementedError(f"{produced=} {residual=} {need=}")
            return _mlp_input_residual_into_sum, ()
        fused = tuple(f for f in fusions if f.completes is produced.group)
        return (
            partial(
                _mlp_input_without_dp,
                gathers_residual=gathers_residual,
                fusions=tuple(f.run for f in fused),
                residual_ops=residual_ops,
            ),
            fused,
        )
    # The partial order adds the residual on attention-TP rank 0 before the DP
    # gather's collective completes that sum, which only a plain residual add
    # allows.
    # Over attention DP and CP, the DP gather puts each CP rank's shard in its
    # DP group's slot, so the one DP sum gathers both axes.
    places_cp_shards = TokenAxis.ATTN_CP in gathered
    if (
        owes_attention_tp
        and not force_layernorm_before_gather
        and residual_ops.adds_plainly
    ):
        return (
            partial(
                _mlp_input_dp_partial,
                gathers_residual=gathers_residual,
                places_cp_shards=places_cp_shards,
            ),
            (),
        )
    return (
        partial(
            _mlp_input_dp_replicate,
            gathers_residual=gathers_residual,
            reduces_attention_tp=owes_attention_tp,
            places_cp_shards=places_cp_shards,
            residual_ops=residual_ops,
        ),
        (),
    )


def _select_ffn_output_move(
    produced: StageOutput,
    *,
    residual: Layout,
    to: Layout,
    cp_moves: Optional[CpMoves] = None,
    residual_ops: ResidualOps = ADD_AND_NORM,
) -> Tuple[bool, Optional[Callable], bool]:
    """How the FFN output reaches the rows the layer hands on: whether it goes
    back by undoing the attention-DP gather (the FFN exit and postprocess run
    that step), or else the postprocess that moves it, None when there is none
    to choose here; and whether that move also completes the sum the FFN
    leaves."""
    pair = CommunicateSummableTensorPairFn
    if produced.layout == residual:
        if to == residual:
            return False, pair._trivial, False
        if to.sharded == residual.sharded - {TokenAxis.ATTN_TP_SCATTER}:
            # Each rank's slice back to the attention's rows: write the output
            # into the residual, then gather over attention TP.
            return False, partial(pair._gather, residual_ops=residual_ops), False
        raise NotImplementedError(f"{produced=} {residual=} {to=}")
    returned = residual.sharded - produced.layout.sharded
    if returned == {TokenAxis.ATTN_TP_SCATTER} and to in (residual, produced.layout):
        # The residual stays on each rank's slice (MHC on an input-scattered
        # batch): a reduce-scatter onto the slice completes the sum the FFN
        # leaves; a complete output is only sliced.
        sums = produced.leaves_for_reduce_scatter
        return (
            False,
            partial(
                pair._onto_residual_shard,
                sums=sums,
                gathers_back=to != residual,
                residual_ops=residual_ops,
            ),
            sums,
        )
    if to != residual or not produced.layout.sharded <= residual.sharded:
        raise NotImplementedError(f"{produced=} {residual=} {to=}")
    if returned == {TokenAxis.ATTN_CP}:
        if cp_moves is None:
            raise NotImplementedError(f"{produced=} {residual=} {to=}")
        if not produced.leaves_for_reduce_scatter:
            # A complete output: this rank's block of it, nothing summed.
            return False, cp_moves.take_back, False
        # The FFN leaves its sum: only a take-back that sums over the same
        # ranks completes it.
        if cp_moves.reduce_scatter is None or not _same_ranks(
            _sum_group(produced.group), cp_moves.reduce_scatter_group()
        ):
            raise NotImplementedError(f"{produced=} {residual=} {to=}")
        return False, cp_moves.reduce_scatter, True
    if returned == {TokenAxis.ATTN_DP, TokenAxis.ATTN_CP}:
        # This rank's CP shard, from where the DP gather put it.
        return False, CommunicateSummableTensorPairFn._take_back_cp_shard, False
    if returned != {TokenAxis.ATTN_DP}:
        raise NotImplementedError(f"{produced=} {residual=} {to=}")
    return True, None, False


class MlpInputKind(Enum):
    """What the attention-TP -> FFN boundary does, given the two sides' layouts."""

    # The layouts agree: add the residual and normalize.
    NORM = auto()
    # All-reduce over attention TP, normalize, and gather for the FFN group.
    GATHER = auto()
    # The same, then gather over the MoE-CP group (moe_dp_size < attn_cp_size).
    GATHER_MOE_CP = auto()
    # Reduce-scatter over attention TP to this rank's tokens, then normalize.
    SCATTER = auto()
    # All-reduce over attention TP for a dense MLP on that group.
    ATTN_TP_ALL_REDUCE = auto()


def mlp_input_kind(
    modes: LayerScatterModes, context: CommunicateContext
) -> MlpInputKind:
    hidden_in, residual_in = modes.attn_mode, modes.layer_input_mode
    hidden_out, residual_out = modes.mlp_mode, modes.middle_residual_mode
    if (
        context.is_same_layout(hidden_in, hidden_out)
        and context.is_same_layout(residual_in, residual_out)
        and context.attn_tp_size == 1
    ):
        return MlpInputKind.NORM
    if hidden_in == ScatterMode.TP_ATTN_FULL and residual_in in (
        ScatterMode.SCATTERED,
        ScatterMode.TP_ATTN_FULL,
    ):
        kind = {
            (ScatterMode.FULL, ScatterMode.TP_ATTN_FULL): MlpInputKind.GATHER,
            (
                ScatterMode.MOE_FULL,
                ScatterMode.TP_ATTN_FULL,
            ): MlpInputKind.GATHER_MOE_CP,
            (ScatterMode.SCATTERED, ScatterMode.SCATTERED): MlpInputKind.SCATTER,
        }.get((hidden_out, residual_out))
        if kind is not None:
            return kind
        if (
            hidden_out == ScatterMode.TP_ATTN_FULL
            and residual_out == ScatterMode.TP_ATTN_FULL
            and context.attn_tp_size > 1
        ):
            # Used when the dense MLP is tensor-parallelized along the
            # attention TP group (``moe_dense_tp_size > 1``): hidden states
            # need an all-reduce inside the attention TP group before the
            # next layernorm, while staying in TP_ATTN_FULL on both sides.
            return MlpInputKind.ATTN_TP_ALL_REDUCE
    raise NotImplementedError(
        f"{hidden_in=} {residual_in=} {hidden_out=} {residual_out=}"
    )


def _mlp_input_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    residual_ops: ResidualOps = ADD_AND_NORM,
):
    return residual_ops.update_and_read_ffn_input(hidden_states, residual, layernorm)


def _mlp_input_attn_tp_all_reduce(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    residual_ops: ResidualOps = ADD_AND_NORM,
):
    """All-reduce hidden states inside the attention TP group, then layernorm.

    Used when the dense MLP shares the attention TP group
    (``moe_dense_tp_size > 1``): both hidden states and residual stay in
    ``TP_ATTN_FULL`` across the boundary.
    """
    hidden_states = get_parallel().attn_tp_group.all_reduce(hidden_states)
    return residual_ops.update_and_read_ffn_input(hidden_states, residual, layernorm)


def _mlp_input_scatter(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    scatters_residual: bool,
    residual_ops: ResidualOps = ADD_AND_NORM,
):
    hidden_states = _reduce_and_redistribute_output_to_attn_tp_shards(
        hidden_states, context
    )
    if scatters_residual:
        residual = residual_ops.residual_to_attn_tp_shard(residual, context)
    return residual_ops.update_and_read_ffn_input(hidden_states, residual, layernorm)


def _mlp_input_on_residual_shard(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    residual_ops: ResidualOps,
):
    """The residual stays on each rank's slice while the FFN takes the full
    rows: reduce-scatter the attention output onto the slice, which completes
    its sum, write it into the residual and read the FFN input there, then
    gather the input back into the attention output's rows."""
    if hidden_states.shape[0] == 0:
        return hidden_states, hidden_states
    shard = hidden_states.tensor_split(context.tp_size)[context.tp_rank]
    get_parallel().tp_group.reduce_scatter_tensor(shard, hidden_states)
    shard, residual = residual_ops.update_and_read_ffn_input(shard, residual, layernorm)
    attn_tp_all_gather_into_tensor(hidden_states, shard)
    return hidden_states, residual


def _mlp_input_residual_into_sum(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
):
    return _tp_all_reduce_with_scattered_residual(
        hidden_states, residual, layernorm, context
    )


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


def _mlp_input_gather(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    order: Callable,
):
    """Run ``order``, the steps ``_mlp_input_order`` chose at construction,
    unless this batch's attention input is scattered."""
    if get_attn_tp_context().input_scattered:
        return _tp_all_reduce_with_scattered_residual(
            hidden_states, residual, layernorm, context
        )
    return order(hidden_states, residual, forward_batch, layernorm, context)


def _mlp_input_gather_attention_cp(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    gather: Callable,
):
    """DSA and MLA CP: complete this rank's shard, then gather the shards, of
    equal length, over the attention-CP group. The residual stays on the
    shard."""
    hidden_states, residual = gather(
        hidden_states, residual, forward_batch, layernorm, context
    )
    return dsa_cp_gather_hidden_states(hidden_states), residual


def _mlp_input_gather_moe_cp(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    gather: Callable,
):
    """Gather for the FFN, then over the MoE-CP group so each rank holds all
    tokens of its MoE group (moe_dp_size < attn_cp_size). The residual stays at
    TP_ATTN_FULL."""
    # Early return on empty tensor is safe for MOE_CP because:
    # - During CP extend: zigzag split guarantees all CP ranks have non-zero tokens,
    #   so no rank hits this path while others proceed to the allgather.
    # - During decode: moe_cp allgather is skipped (guarded by is_context_parallel_extend).
    # - CUDA graph warmup: not applicable when --cuda-graph-backend-prefill=disabled is used.
    if hidden_states.shape[0] == 0:
        return hidden_states, residual

    hidden_states, residual = gather(
        hidden_states, residual, forward_batch, layernorm, context
    )

    rows = moe_cp_gathered_rows(forward_batch)
    if rows is not None and hidden_states.shape[0] > 0:
        hidden_states = _redistribute_input_to_moe_cp(
            hidden_states, rows, get_moe_cp_size()
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
    hidden_states: torch.Tensor, rows: List[int]
) -> torch.Tensor:
    moe_cp_rank = get_moe_cp_rank()
    # The allgather was padded to max_tokens_per_rank (equal chunks).
    # Extract this rank's actual (non-padded) tokens from its chunk.
    max_tokens_per_rank = max(rows)
    actual_local_tokens = rows[moe_cp_rank]
    return hidden_states.narrow(
        0, moe_cp_rank * max_tokens_per_rank, actual_local_tokens
    ).contiguous()


def _reduce_and_redistribute_output_step(
    forward_batch: ForwardBatch,
    *,
    leaves_for_reduce_scatter: bool,
    leaves_for_reduce_scatterv: bool,
) -> Optional[Callable[[torch.Tensor, torch.Tensor, ForwardBatch], None]]:
    """The reduce-scatter that brings an FFN output gathered over attention
    DP back to this rank's tokens when the FFN leaves its sum to it (see
    StageOutput); None when the FFN reduces the output and only a scatter
    remains."""
    if should_use_dp_reduce_scatterv() and leaves_for_reduce_scatterv:
        return _reduce_and_redistribute_output_varlen
    if (
        leaves_for_reduce_scatter
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
        step = (
            _reduce_and_redistribute_output_step(
                forward_batch,
                leaves_for_reduce_scatter=allow_reduce_scatter,
                # A MoE block leaves its sum to reduce_scatterv whenever it
                # applies (should_skip_post_experts_all_reduce).
                leaves_for_reduce_scatterv=allow_reduce_scatter or is_layer_sparse,
            )
            or _redistribute_output
        )
        return _to_local_tokens(step, forward_batch, hidden_states), residual

    @staticmethod
    def _gather(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        residual_ops: ResidualOps = ADD_AND_NORM,
        **kwargs,
    ):
        hidden_states = residual_ops.update_residual(hidden_states, residual)
        return _redistribute_from_attn_tp_shards(hidden_states), None

    @staticmethod
    def _onto_residual_shard(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        *,
        sums: bool,
        gathers_back: bool,
        residual_ops: ResidualOps,
        **kwargs,
    ):
        """Bring the FFN output onto the slice of the rows the residual is on:
        a reduce-scatter that also completes its sum when ``sums``, else this
        rank's slice of the complete output. With ``gathers_back`` it is
        written into the residual there and the full rows are gathered back."""
        if sums:
            hidden_states, _ = tp_reduce_scatter(hidden_states, None, context)
        else:
            hidden_states = hidden_states.tensor_split(context.tp_size)[context.tp_rank]
        if not gathers_back:
            return hidden_states, residual
        local_states = residual_ops.update_residual(hidden_states, residual)
        hidden_states = local_states.new_empty(
            local_states.shape[0] * context.tp_size, *local_states.shape[1:]
        )
        get_parallel().tp_group.all_gather_into_tensor(hidden_states, local_states)
        return hidden_states, None

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
    def _take_back_attention_cp_shard(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        **kwargs,
    ):
        """DSA and MLA CP: this rank's shard of a complete output gathered in
        equal shards over the attention-CP group."""
        parallel = get_parallel()
        shard = hidden_states.tensor_split(parallel.attn_cp_size)[parallel.attn_cp_rank]
        return shard, residual

    @staticmethod
    def _reduce_scatter_over_cp(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        **kwargs,
    ):
        """DSA and MLA CP: sum the FFN output over the attention-CP group and
        keep this rank's shard."""
        return dsa_cp_reduce_scatter_hidden_states(hidden_states), residual

    @staticmethod
    def _take_back_cp_shard(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        **kwargs,
    ):
        """This rank's CP shard of the rows the DP gather put in its DP group's
        slot, at the shard's padded length with the padding zeroed."""
        held = forward_batch.attn_cp_metadata.per_rank_actual_token
        local_hidden_states = get_local_dp_buffer(_dp_scatter_group())[
            : held[get_parallel().attn_cp_rank]
        ]
        dp_scatter(
            local_hidden_states,
            hidden_states,
            forward_batch,
            _cp_shard_token_rows(forward_batch),
        )
        return local_hidden_states, residual

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
        # Safe w.r.t. empty tensors: same reasoning as _mlp_input_gather_moe_cp
        # — CP extend always has non-zero tokens per rank, and decode skips this path.
        rows = moe_cp_gathered_rows(forward_batch)
        if rows is not None:
            hidden_states = _redistribute_output_from_moe_cp(hidden_states, rows)

        if context.attn_dp_size > 1:
            hidden_states = _to_local_tokens(
                _redistribute_output, forward_batch, hidden_states
            )

        return hidden_states, residual
