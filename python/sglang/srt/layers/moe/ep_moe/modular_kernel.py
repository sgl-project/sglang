# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/f9c069c85e029830094ff9abb926ffbf37b7c7e7/vllm/model_executor/layers/fused_moe/modular_kernel.py
from abc import ABC, abstractmethod
import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    try_get_optimal_moe_config, get_config_dtype_str)
from sglang.srt.layers.moe.ep_moe.utils import _resize_cache

from sgl_kernel import gelu_and_mul, silu_and_mul

#
logger = logging.getLogger(__name__)
#
# This file defines a set of base classes used to make MoE kernels more modular.
# The goal is to be able to utilize different communication mechanisms with
# any fused MoE kernel without needing to have combinatoric implementations.
#
# The fused moe kernels are broken down into the following components:
#
# [Router] → [Quantize-Dispatch] → [Permute-Experts-Unpermute] → [Combine]
#
# Each component will be independent of the others except for
# [Quantize-Dispatch] and `[Combine] (see below). The components can then be
# mixed and matched with so that DP+EP can be supported easily for multiple
# MoE kernel implementations.
#
# The following main classes are defined:
# * FusedMoEPrepareAndFinalize - an abstract base class for preparation of MoE
#   inputs (e.g. quantization, distribution) and finalization of Moe outputs.
#   The prepare method must take care of any needed quantization and the
#   finalize method must apply weights and do the final reduction of the output.
# * FusedMoEPermuteExpertsUnpermute - an abstract base class for the main fused
#   MoE operation. One important feature to note is that this class does not
#   apply topk weights or reduce the final output.
#   FusedMoEPrepareAndFinalize and a FusedMoEPermuteExpertsUnpermute to
#   provide the standard fused MoE kernel interface.
#
# [Quantize-Prepare] and [Finalize] functionality are bundled into a single
# class `FusedMoEPrepareAndFinalize` since they could use collective
# communication mechanisms that need to be consistent.
#


def _moe_problem_size(
    a1: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    """
    Extract the MoE problem size from the given tensor arguments:
    - a: The hidden states, input to the MoE layer.
    - w1: The first set of expert weights.
    - w2: The second set of expert weights.
    - topk_ids: The topk ids.

    Note: extracting the problem shape from the weight and activation tensors is
    not obvious.  It needs to be done this way specifically due to subtle issues
    with particular kernels, e.g. the int4 kernels divide the trailing dimension
    by two, so it's not "correct" to extract N or K from the trailing dimension
    of w1 or w2.  Similarly, some kernels transpose the weights, so this needs
    to be kept in mind.
    """
    assert w1.dim() == 3 and w2.dim() == 3
    E, N, _ = w1.size()
    K = w2.size(1)

    if a1.dim() == 2:
        # Make sure we are using the correct a1 (pre-permute).
        assert topk_ids.size(0) == a1.size(0), \
            f"{topk_ids.size(0)} != {a1.size(0)}"
        M = a1.size(0)
    else:
        assert a1.dim() == 3
        assert a1.size(0) == E, f"{a1.size(0)} == {E}"
        M = a1.size(1)  # This is max_num_tokens

    assert topk_ids.dim() == 2
    topk = topk_ids.size(1)

    return E, M, N, K, topk


@triton.jit
def moe_mmk(
        a_ptrs,
        b_ptrs,
        K,
        expert_id,
        a_scale_ptr,
        b_scale_ptr,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_ak,
        stride_bk,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Offsets and masks
        offs_m,
        offs_n,
        mask_m,
        # Block size for block-wise quantization
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        # Meta-parameters
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        compute_type: tl.constexpr,
        use_w8a8: tl.constexpr,
        use_w8a16: tl.constexpr):

    offs_k = tl.arange(0, BLOCK_K)

    if use_w8a16:
        b_scale_ptrs = b_scale_ptr + expert_id * stride_bse + offs_n[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            offs_bsn = offs_n // group_n
            b_scale_ptrs = (b_scale_ptr + expert_id * stride_bse +
                            offs_bsn * stride_bsn)
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + expert_id)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(a_ptrs,
                    mask=mask_m[:, None] & (offs_k[None, :] < K - k * BLOCK_K),
                    other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # We accumulate along the K dimension.
        if use_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                  mask=mask_m,
                                  other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:,
                                                      None] * b_scale[None, :]
            else:
                if use_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if use_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    return accumulator


@triton.jit
def expert_triton_kernel(
        a_ptr,  #[max_tokens, K]
        b_ptr,  #[K, N]
        c_ptr,  #[max_tokens, N]
        expert_id,
        compute_type: tl.constexpr,
        # Dimensions
        M,
        N,
        K,
        # Quantization data
        a_scale_ptr,
        b_scale_ptr,
        b_zp_ptr,
        # strides
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Blockwise quantization data
        group_n,
        group_k,
        # Quantization schemes
        use_fp8_w8a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr,
        # Kernel config
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr):

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) % N
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = moe_mmk(
        a_ptrs,
        b_ptrs,
        K,
        expert_id,
        a_scale_ptr,
        b_scale_ptr,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_ak,
        stride_bk,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Offsets and masks
        offs_m,
        offs_n,
        mask_m,
        # Block size for block-wise quantization
        group_n,
        group_k,
        # Meta-parameters
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a16)

    # store in C
    offs_cn = tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = mask_m[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def batched_triton_kernel(
        a_ptr,  # [E, max_num_tokens, K]
        b_ptr,  # [E, K, N]
        c_ptr,  # [E, max_num_tokens, N]
        expert_num_tokens,  # [E]
        compute_type: tl.constexpr,
        # Dimensions
        max_num_tokens,
        K,
        N,
        # Quantization data
        a_scale_ptr,
        b_scale_ptr,
        b_zp_ptr,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_ae,
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_ce,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Blockwise quantization data
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        # Quantization schemes
        use_fp8_w8a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr,
        # Kernel config
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr):
    expert_id = tl.program_id(axis=0)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)
    if e_num_tokens == 0:
        # Early exit
        return

    pid_mn = tl.program_id(axis=1)
    #num_pid_m = tl.cdiv(max_num_tokens, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    cta_m_start = pid_m * BLOCK_M
    cta_n_start = pid_n * BLOCK_N
    if cta_m_start >= e_num_tokens:
        # Early exit
        return

    cta_m_size = min(BLOCK_M, e_num_tokens - cta_m_start)
    cta_n_size = min(BLOCK_N, N - cta_n_start)

    a_ptr = a_ptr + expert_id * stride_ae + cta_m_start * stride_am
    b_ptr = b_ptr + expert_id * stride_be + cta_n_start * stride_bn
    c_ptr = (c_ptr + expert_id * stride_ce + cta_m_start * stride_cm +
             cta_n_start * stride_cn)

    expert_triton_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        expert_id,
        compute_type,
        cta_m_size,  # M
        cta_n_size,  # N
        K,  # K
        a_scale_ptr,
        b_scale_ptr,
        b_zp_ptr,
        # Strides
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Blockwise quantization data
        group_n,
        group_k,
        # Quantization schemes
        use_fp8_w8a8,
        use_int8_w8a16,
        # Kernel config
        BLOCK_M,
        BLOCK_N,
        BLOCK_K)


def invoke_moe_batched_triton_kernel(
        A: torch.Tensor,  # [E, max_tokens, K]
        B: torch.Tensor,  # [E, K, N]
        C: torch.Tensor,  # [E, max_tokens, N]
        expert_num_tokens: torch.Tensor,  # [E]
        compute_type: tl.dtype,
        # Quantization data
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        B_zp: torch.Tensor,
        # Quantization schemes
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        config: dict[str, int],
        block_shape: Optional[list[int]] = None):

    assert not use_int4_w4a16
    max_num_tokens = A.size(1)
    K = A.size(2)
    N = C.size(2)

    BLOCK_M = config['BLOCK_SIZE_M']
    BLOCK_N = config['BLOCK_SIZE_N']
    BLOCK_K = config['BLOCK_SIZE_K']
    assert (torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing()
            or max_num_tokens % BLOCK_M == 0)

    grid = (expert_num_tokens.size(0), triton.cdiv(max_num_tokens, BLOCK_M) *
            triton.cdiv(B.size(1), BLOCK_N))

    batched_triton_kernel[grid](
        A,
        B,
        C,
        expert_num_tokens,
        compute_type,
        # Dimensions
        max_num_tokens,
        K,
        N,
        # Quantization data
        A_scale,
        B_scale,
        B_zp,
        # Strides
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        # Blockwise quantization data
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        # Quantization schemes
        use_fp8_w8a8,
        use_int8_w8a16,
        # Kernel config
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K)

class FusedMoEPermuteExpertsUnpermute(ABC):
    """
    An abstract base class for the [Permute-Experts-Unpermute] step described
    above.
    """

    @abstractmethod
    def workspace_shapes(
        self,
        a: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int,
    ) -> tuple[int, int, torch.dtype]:
        """
        Compute the number of elements for the temporary outputs of the two
        gemms and activation in the fused expert function.  Since the
        gemms are independent, the workspace for the first gemm can be shared
        with the workspace for the last gemm.

        Returns a tuple of:
        - Number of workspace13 elements: must be large enough to hold the
          result of either expert gemm.
        - Number of workspace2 elements: must be large enough to hold the
          result of the activation function.
        - Workspace type: The dtype to use for the workspace tensors.
        """
        raise NotImplementedError

    def activation(self, activation: str, output: torch.Tensor,
                   input: torch.Tensor) -> None:
        assert output.size(-1) * 2 == input.size(-1)
        if activation == "silu":
            silu_and_mul(input, output)
        elif activation == "gelu":
            gelu_and_mul(input, output)
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    @abstractmethod
    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_num_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        This function computes the intermediate result of a Mixture of Experts
        (MoE) layer using two sets of weights, w1 and w2.

        Parameters:
        - hidden_states: (torch.Tensor): The (quantized) input tensor to the MoE
          layer.
        - w1 (torch.Tensor): The first set of expert weights.
        - w2 (torch.Tensor): The second set of expert weights.
        - topk_ids (torch.Tensor): A map of row to expert id.
        - activation (str): The activation function to apply after the first
          MoE layer.
        - global_num_experts (int): The total number of experts in the global
          expert space.
        - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
          from the global expert space to the local expert space of the expert
          parallel shard.
        - w1_scale (Optional[torch.Tensor]): Optional scale to be used for w1.
        - w2_scale (Optional[torch.Tensor]): Optional scale to be used for w2.
        - w1_zp (Optional[torch.Tensor]): Optional zero points to be used for
          w1.
        - w2_zp (Optional[torch.Tensor]): Optional zero points to be used for
          w2.
        - a1q_scale (Optional[torch.Tensor]): Optional quantized scale to be
          used for a1.
        - a2_scale (Optional[torch.Tensor]): Optional scale to be used for a2.
        - workspace13 (torch.Tensor): A scratch tensor used for gemm outputs
          must be large enough to hold output of either MoE gemm.
        - workspace2 (torch.Tensor): A scratch tensor used for the activation
          function.
        - expert_num_tokens: An optional tensor containing the number of tokens
          assigned to each expert when using batched experts format input.

        Returns:
        - torch.Tensor: The unweighted, unreduced output tensor
        """
        raise NotImplementedError


class BatchedExperts(FusedMoEPermuteExpertsUnpermute):
    """
    A reference MoE expert class that operates on expert batched format,
    i.e. E x max_num_tokens x K.  This is the format that the pplx
    dispatch/combine kernels use.
    """

    def __init__(
        self,
        dp_size: int,
        max_num_tokens: Optional[int] = None,
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        block_shape: Optional[list[int]] = None,
        block_m: Optional[int] = None,
    ):
        super().__init__()
        assert block_shape is None
        assert block_m is None
        assert not use_fp8_w8a8, "NYI"
        assert not use_int8_w8a8, "NYI"
        assert not use_int8_w8a16, "NYI"
        assert not use_int4_w4a16, "NYI"
        self.max_num_tokens = max_num_tokens
        self.dp_size = dp_size

    def workspace_shapes(
        self,
        a: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int,
    ) -> tuple[int, int, torch.dtype]:
        assert a.dim() == 2
        max_num_tokens = a.size(0) if self.max_num_tokens is None else self.max_num_tokens
        #print(f"WORKSPACE {max_num_tokens} {num_dp}")
        workspace13 = num_experts * max_num_tokens * self.dp_size * K
        workspace2 = max_num_tokens * self.dp_size * N
        return (workspace13, workspace2, a.dtype)

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_num_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert hidden_states.dim() == 3
        assert expert_num_tokens is not None
        hidden_dim = hidden_states.size(-1)

        if self.max_num_tokens is None:
            max_num_tokens = hidden_states.size(1)
        else:
            max_num_tokens = self.max_num_tokens

        num_dp = self.dp_size
        num_experts = global_num_experts
        out = _resize_cache(workspace13,
                            (num_experts, max_num_tokens * num_dp, hidden_dim))
        num_local_experts = w1.size(0)
        assert num_local_experts == w1.size(0), (
            f"{num_local_experts} == {w1.size(0)}")

        N = w1.size(1) // 2

        # Not cudagraph friendly
        assert (torch.compiler.is_compiling()
                or torch.cuda.is_current_stream_capturing()
                or torch.all(expert_num_tokens <= max_num_tokens * num_dp)), (
                    f"{expert_num_tokens} <= {max_num_tokens * num_dp}")

        for expert in range(num_local_experts):
            # Indexing expert_num_tokens doesn't work w/cudagraphs or inductor
            if (torch.compiler.is_compiling()
                    or torch.cuda.is_current_stream_capturing()):
                num = max_num_tokens * num_dp
            else:
                num = int(expert_num_tokens[expert].item())
            tmp = _resize_cache(workspace2, (num, N))
            input = hidden_states[expert, :num, :] @ w1[expert].transpose(0, 1)
            self.activation(activation, tmp, input)
            out[expert, :num, :] = tmp @ w2[expert].transpose(0, 1)

        return out


class BatchedTritonExperts(FusedMoEPermuteExpertsUnpermute):
    """
    A Triton based MoE expert class that operates on expert batched format,
    i.e. E x max_num_tokens x K.  This is the format that the pplx
    dispatch/combine kernels use.
    """

    def __init__(
        self,
        dp_size: int,
        max_num_tokens: Optional[int] = None,
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        block_shape: Optional[list[int]] = None,
    ):
        super().__init__()
        self.use_fp8_w8a8 = use_fp8_w8a8
        self.use_int8_w8a8 = use_int8_w8a8
        self.use_int4_w4a16 = use_int4_w4a16
        self.use_int8_w8a16 = use_int8_w8a16
        self.block_shape = block_shape
        self.max_num_tokens = max_num_tokens
        assert not use_int8_w8a8, "NYI"
        assert not use_int4_w4a16, "NYI"
        self.dp_size = dp_size

    def workspace_shapes(
        self,
        a: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int,
    ) -> tuple[int, int, torch.dtype]:
        assert a.dim() == 2
        max_num_tokens = a.size(0) if self.max_num_tokens is None else self.max_num_tokens
        workspace13 = num_experts * max_num_tokens * self.dp_size * max(K, N)
        workspace2 = num_experts * max_num_tokens * self.dp_size * (N // 2)
        return (workspace13, workspace2, a.dtype)

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_num_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Check constraints.
        if self.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), (
                "Hidden size mismatch")
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} "
                f"!= {w1.size(2)}")

        assert hidden_states.is_contiguous(
        ), "Hidden_states must be contiguous"
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn
        ]

        # TODO: num_tokens -> max_num_tokens?
        E, num_tokens, N, K, top_k_num = _moe_problem_size(
            hidden_states, w1, w2, topk_ids)

        assert w1.size(0) == E
        assert w2.size(0) == E

        config_dtype = get_config_dtype_str(use_fp8_w8a8=self.use_fp8_w8a8,
                                            use_int8_w8a16=self.use_int8_w8a16,
                                            use_int4_w4a16=self.use_int4_w4a16,
                                            dtype=hidden_states.dtype)

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            config_dtype,
            num_tokens,
            block_shape=self.block_shape,
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif hidden_states.dtype == torch.float8_e4m3fn:
            compute_type = tl.bfloat16
        else:
            raise ValueError(
                f"Unsupported compute_type: {hidden_states.dtype}")

        #print(f"shape: E={E}, M={num_tokens}, N={N}, K={K}, top_k={top_k_num}")
        # We can reuse the memory between these because by the time we need
        # cache3, we're done with cache1
        intermediate_cache1 = _resize_cache(workspace13, (E, num_tokens, N))
        intermediate_cache2 = _resize_cache(workspace2,
                                            (E, num_tokens, N // 2))
        intermediate_cache3 = _resize_cache(workspace13, (E, num_tokens, K))

        # MM1
        invoke_moe_batched_triton_kernel(A=hidden_states,
                                         B=w1,
                                         C=intermediate_cache1,
                                         expert_num_tokens=expert_num_tokens,
                                         compute_type=compute_type,
                                         A_scale=a1q_scale,
                                         B_scale=w1_scale,
                                         B_zp=w1_zp,
                                         use_fp8_w8a8=self.use_fp8_w8a8,
                                         use_int8_w8a16=self.use_int8_w8a16,
                                         use_int4_w4a16=self.use_int4_w4a16,
                                         config=config,
                                         block_shape=self.block_shape)

        # TODO: would be nice to use expert_num_tokens here to reduce
        # garbage compute
        self.activation(activation, intermediate_cache2.view(-1, N // 2),
                        intermediate_cache1.view(-1, N))

        #qintermediate_cache2 = intermediate_cache2
        a2q_scale = a2_scale
        # TODO (varun) : support w8a8
        assert not self.use_fp8_w8a8
        #if self.use_fp8_w8a8:
        #    qintermediate_cache2, a2q_scale = _fp8_quantize(
        #        intermediate_cache2, a2_scale, self.block_shape)

        invoke_moe_batched_triton_kernel(A=intermediate_cache2,
                                         B=w2,
                                         C=intermediate_cache3,
                                         expert_num_tokens=expert_num_tokens,
                                         compute_type=compute_type,
                                         A_scale=a2q_scale,
                                         B_scale=w2_scale,
                                         B_zp=w2_zp,
                                         use_fp8_w8a8=self.use_fp8_w8a8,
                                         use_int8_w8a16=self.use_int8_w8a16,
                                         use_int4_w4a16=self.use_int4_w4a16,
                                         config=config,
                                         block_shape=self.block_shape)

        return intermediate_cache3
