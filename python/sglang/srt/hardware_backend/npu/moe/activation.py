from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.language.extra.cann.libdevice as libdevice

from sgl_kernel_npu.utils.triton_utils import get_device_properties

from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.activation import GeluAndMul
from sglang.srt.runtime_context import get_parallel


# =============================================================================
# Abstract base for all activation variants
# =============================================================================
class BaseActivation(ABC):
    @abstractmethod
    def _apply_activation(
        self, *args, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: ...


# =============================================================================
# Concrete activation implementations (unchanged except removed 8.)
# =============================================================================
class NPUSwiglu(BaseActivation):
    def _apply_activation(self, hidden_states: torch.Tensor):
        return torch.ops.npu.npu_swiglu(hidden_states), None


class NPUSwigluQuant(BaseActivation):
    def _apply_activation(self, hidden_states: torch.Tensor):
        hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
            hidden_states,
            quant_mode=1,
            activate_left=True,
        )
        return hidden_states, swiglu_out_scale


class NPUSwigluQuantWithScales(BaseActivation):
    def _apply_activation(
        self,
        hidden_states: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        group_index: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        quant_scale: Optional[torch.Tensor] = None,
        quant_offset: Optional[torch.Tensor] = None,
    ):
        hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
            x=hidden_states,
            weight_scale=weight_scale,
            activation_scale=activation_scale,
            bias=bias,
            quant_scale=quant_scale,
            quant_offset=quant_offset,
            group_index=group_index,
            activate_left=True,
            quant_mode=1,
        )
        return hidden_states, swiglu_out_scale


class NPUSwigluDeepEPKernel(BaseActivation):
    def __init__(self, need_quant: bool = True):
        from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant

        self._kernel = swiglu_quant
        self.need_quant = need_quant

    def _apply_activation(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
    ):
        hidden_states, per_token_scale = self._kernel(
            hidden_states, group_list, group_list_type, need_quant=self.need_quant
        )
        if self.need_quant:
            return hidden_states, per_token_scale
        return hidden_states, None


@triton.jit
def _situ_deepep_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    scale_ptr,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALIGNED: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BETA: tl.constexpr,
    INV_BETA: tl.constexpr,
    DO_LINEAR_BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    INV_LINEAR_BETA: tl.constexpr,
    NEED_QUANT: tl.constexpr,
):
    """Kimi-K3 SiTU over DeepEP's expert-packed rows.

    This is the established 0728 NPU formula.  Only rows covered by the
    DeepEP group list are materialized; the grouped down projection ignores
    padding rows.
    """
    if GROUP_LIST_TYPE == 0:
        total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
    else:
        offsets = tl.arange(0, NUM_EXPERTS_ALIGNED)
        mask = offsets < NUM_EXPERTS
        counts = tl.load(group_list_ptr + offsets, mask=mask, other=0).to(
            tl.int32
        )
        total_rows = tl.sum(counts)

    rows_per_core = (total_rows - 1) // NUM_CORES + 1
    row_begin = tl.program_id(0) * rows_per_core
    if row_begin >= total_rows:
        return
    row_end = tl.minimum(row_begin + rows_per_core, total_rows)

    cols = tl.arange(0, HALF_COLS)
    for row in range(row_begin, row_end):
        row_offset = row.to(tl.int64) * TOTAL_COLS
        gate = tl.load(x_ptr + row_offset + cols).to(tl.float32)
        up = tl.load(x_ptr + row_offset + HALF_COLS + cols).to(tl.float32)
        gate = BETA * libdevice.tanh(gate * INV_BETA) * tl.sigmoid(gate)
        if DO_LINEAR_BETA:
            up = LINEAR_BETA * libdevice.tanh(up * INV_LINEAR_BETA)
        value = gate * up

        if NEED_QUANT:
            scale = tl.maximum(tl.max(tl.abs(value)) / 127.0, 1e-30)
            tl.store(
                scale_ptr + row.to(tl.int64),
                scale.to(scale_ptr.dtype.element_ty),
            )
            for col_begin in range(0, HALF_COLS, COL_BLOCK_SIZE):
                block = al.extract_slice(
                    value,
                    offsets=(col_begin,),
                    sizes=(COL_BLOCK_SIZE,),
                    strides=(1,),
                )
                block = tl.floor(block.to(tl.float32) / scale + 0.5)
                block = tl.clamp(block, -128, 127).to(tl.int8)
                block_cols = col_begin + tl.arange(0, COL_BLOCK_SIZE)
                block_mask = block_cols < HALF_COLS
                tl.store(
                    out_ptr + row.to(tl.int64) * HALF_COLS + block_cols,
                    block.to(out_ptr.dtype.element_ty),
                    mask=block_mask,
                )
        else:
            tl.store(
                out_ptr + row.to(tl.int64) * HALF_COLS + cols,
                value.to(out_ptr.dtype.element_ty),
            )


class NPUSituDeepEPKernel(BaseActivation):
    """SiTU activation and optional INT8 requantization for DeepEP."""

    def __init__(
        self,
        *,
        need_quant: bool,
        beta: float = 4.0,
        linear_beta: Optional[float] = 25.0,
    ):
        self.need_quant = need_quant
        self.beta = float(beta)
        self.linear_beta = (
            None if linear_beta is None else float(linear_beta)
        )

    def _apply_activation(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
    ):
        if group_list_type not in (0, 1):
            raise ValueError(
                f"group_list_type must be 0 or 1, got {group_list_type}"
            )
        if hidden_states.ndim != 2 or hidden_states.shape[1] % 2:
            raise ValueError(
                "DeepEP SiTU input must have shape [tokens, 2 * intermediate]"
            )
        if group_list.dtype == torch.int64:
            num_experts_aligned = (group_list.numel() + 7) // 8 * 8
        elif group_list.dtype == torch.int32:
            num_experts_aligned = (group_list.numel() + 15) // 16 * 16
        else:
            raise ValueError("group_list must use int32 or int64")

        rows, total_cols = hidden_states.shape
        half_cols = total_cols // 2
        out = torch.empty(
            (rows, half_cols),
            dtype=torch.int8 if self.need_quant else hidden_states.dtype,
            device=hidden_states.device,
        )
        scale = torch.empty(rows, dtype=torch.float32, device=hidden_states.device)
        _, num_vector_cores = get_device_properties()
        linear_beta = self.linear_beta if self.linear_beta is not None else 1.0
        _situ_deepep_kernel[(num_vector_cores,)](
            hidden_states,
            group_list,
            out,
            scale,
            TOTAL_COLS=total_cols,
            HALF_COLS=half_cols,
            COL_BLOCK_SIZE=half_cols,
            NUM_EXPERTS=group_list.numel(),
            NUM_EXPERTS_ALIGNED=num_experts_aligned,
            GROUP_LIST_TYPE=group_list_type,
            NUM_CORES=num_vector_cores,
            BETA=self.beta,
            INV_BETA=1.0 / self.beta,
            DO_LINEAR_BETA=self.linear_beta is not None,
            LINEAR_BETA=linear_beta,
            INV_LINEAR_BETA=1.0 / linear_beta,
            NEED_QUANT=self.need_quant,
            multibuffer=True,
        )
        return out, scale if self.need_quant else None


class NPUGeluAndMul(BaseActivation):
    def __init__(self):
        self._gelu = GeluAndMul()

    def _apply_activation(self, hidden_states: torch.Tensor):
        return self._gelu(hidden_states), None


class NPUSwigluOAI(BaseActivation):
    def __init__(self, moe_runner_config=None):
        from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai_triton

        self._kernel = swiglu_oai_triton
        self._moe_runner_config = moe_runner_config

    def _apply_activation(self, hidden_states: torch.Tensor):
        # hidden_states is the output of the grouped matmul with shape
        # [num_tokens, 2 * inter].  The old swiglu_oai kernel derived the
        # gate_up dimension from layer.w13_weight.shape[2], which now fails
        # because w13_weight is stored un-transposed.  Instead we pass
        # the gate_up dimension explicitly from the tensor itself.
        alpha = 1.0
        clamp = None
        if self._moe_runner_config is not None:
            alpha = getattr(self._moe_runner_config, "gemm1_alpha", 1.0)
            clamp = getattr(self._moe_runner_config, "gemm1_clamp_limit", None)

        output = self._kernel(
            hidden_states,
            hidden_states.shape[-1],  # gate_up dim = 2 * inter
            alpha,
            clamp,
        )
        return output, None


class NPUSwigluStepAndMul(BaseActivation):
    def __init__(self, clamp_limit: Optional[float] = None):
        self._clamp_limit = clamp_limit

    def _apply_activation(self, hidden_states: torch.Tensor):
        if self._clamp_limit is not None:
            return self._swiglustep_and_mul(hidden_states, self._clamp_limit), None
        return torch.ops.npu.npu_swiglu(hidden_states), None

    @staticmethod
    def _swiglustep_and_mul(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = F.silu(gate).clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        return gate * up


# =============================================================================
# Generic TP all‑gather wrapper – used by the runner when needed
# =============================================================================
class AllGatherActivationWrapper(BaseActivation):
    """
    Wraps any activation and adds an all‑gather along `dim` if TP > 1.

    This allows the runner to stay TP‑agnostic: the wrapper is applied
    transparently at construction time.
    """

    def __init__(self, inner: BaseActivation, dim: int = -1):
        self.inner = inner
        self.dim = dim

    def _apply_activation(self, *args, **kwargs):
        out, scale = self.inner._apply_activation(*args, **kwargs)
        if get_parallel().tp_size > 1:
            out = tensor_model_parallel_all_gather(out, dim=self.dim)
        return out, scale


# =============================================================================
# Factory (unchanged, returns *base* activations)
# =============================================================================
def get_swiglu_variant(method: str, **kwargs: Any) -> BaseActivation:
    variants: dict[str, type[BaseActivation]] = {
        "standard": NPUSwiglu,
        "dequant_swiglu_quant": NPUSwigluQuant,
        "dequant_swiglu_quant_with_scales": NPUSwigluQuantWithScales,
        "swiglu_quant_deepep_kernel": NPUSwigluDeepEPKernel,
        "gelu_and_mul": NPUGeluAndMul,
    }
    if method == "swiglu_oai":
        # The OAI variant now uses the triton kernel that derives the gate_up
        # dimension from the tensor itself.  No extra parameters are needed.
        return NPUSwigluOAI()
    if method == "swiglustep_and_mul":
        clamp_limit = kwargs.pop("clamp_limit", None)
        return NPUSwigluStepAndMul(clamp_limit=clamp_limit)
    if method not in variants:
        raise ValueError(f"Unknown SwiGLU variant: {method}")
    return variants[method]()
