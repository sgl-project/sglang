from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch


class CutlassMoEType(Enum):
    """
    Enum for the different types of cutlass moe operations
    that are currently supported in SGLang.
    """

    BlockscaledFP8 = auto()
    BlockscaledFP4 = auto()


@dataclass
class CutlassMoEParams:
    """
    Parameters for the cutlass moe operation.
    """

    #  Type as defined above
    cutlass_moe_type: CutlassMoEType

    # Strides for activations, weights and output in logical number of elements.
    # The activations & output stride is the number of elements to the next row.
    # The weights stride is the number of elements to the next row per expert.
    # For example, if the weight is [e, n, k], then the b_stride is a tensor of
    # shape [e] with each element being k. Similarly for activations, if the
    # shape is [m, k], then the a_stride has shape [e] with each value k.
    # Similarly for output, if the output is [m, n], then the c_stride is a
    # tensor of shape [e] with each element being k.

    # Note: cutlass_fp4_group_mm is designed to accept the strides of
    # activations and weights to be the same, so it is passed in as a single
    # tensor.
    # ab_strides_13: [e] dtype: int64 [Gemm 1: Activation / Weight strides]
    # ab_strides_2: [e] dtype: int64 [Gemm 2: Activation / Weight strides]
    # c_strides_13: [e] dtype: int64 [Gemm 1: Output Strides]
    # c_strides_2: [e] dtype: int64 [Gemm 2: Output Strides]
    ab_strides_13: torch.Tensor
    ab_strides_2: torch.Tensor
    c_strides_13: torch.Tensor
    c_strides_2: torch.Tensor

    # m: Total number of tokens
    # n: intermediate size per partition
    # k: hidden size per expert
    # e: Number of experts
    # device: Device to run computation on and store tensors
    m: int
    intermediate_size_per_partition: int
    hidden_size: int
    num_experts: int
    device: torch.device

    # Pointers container for calculating offsets of the input activations for each expert
    # a_ptrs: [e] dtype: int64
    a_ptrs: torch.Tensor

    # Pointers container for calculating offsets of the input weights for each expert
    # b_ptrs: [e] dtype: int64
    b_ptrs: torch.Tensor

    # Pointers container for calculating offsets of the output activations for each expert
    # out_ptrs: [e] dtype: int64
    out_ptrs: torch.Tensor
    # Pointers container for calculating offsets of the input scales for each expert
    # a_scales_ptrs: [e] dtype: int64
    # b_scales_ptrs: [e] dtype: int64
    a_scales_ptrs: torch.Tensor
    b_scales_ptrs: torch.Tensor

    # Offsets that mark at which token index each expert begins its computation
    # The number of tokens computed with expert E is expert_offsets[E + 1] - expert_offsets[E]
    # expert_offsets: [e+1] dtype: int32
    expert_offsets: torch.Tensor

    # Problem size: (num_experts, (m,2n,k)) for first GEMM
    # problem_sizes1: [e, 3] dtype: int32
    # Problem size: (num_experts, (m,n,k)) for second GEMM
    # problem_sizes2: [e, 3] dtype: int32
    problem_sizes1: torch.Tensor
    problem_sizes2: torch.Tensor
    # Similar to expert_offsets, but for blockscales for FP4 blockscaled Group GEMM
    blockscale_offsets: Optional[torch.Tensor] = None

    def __init__(
        self,
        cutlass_moe_type: CutlassMoEType,
        device: torch.device,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_size: int,
    ):
        self.cutlass_moe_type = cutlass_moe_type
        self.device = device
        self.num_experts = num_experts
        self.intermediate_size_per_partition = intermediate_size_per_partition
        self.hidden_size = hidden_size
        self.n = self.intermediate_size_per_partition
        self.k = self.hidden_size
        self.e = self.num_experts
        self.ab_strides_13 = torch.full(
            (self.e,), self.k, dtype=torch.int64, device=self.device
        )
        self.ab_strides_2 = torch.full(
            (self.e,), self.n, dtype=torch.int64, device=self.device
        )
        self.c_strides_13 = torch.full(
            (self.e,), 2 * self.n, dtype=torch.int64, device=self.device
        )
        self.c_strides_2 = torch.full(
            (self.e,), self.k, dtype=torch.int64, device=self.device
        )
        self.expert_offsets = torch.empty(
            (self.e + 1,), dtype=torch.int32, device=self.device
        )
        self.problem_sizes1 = torch.empty(
            (self.e, 3), dtype=torch.int32, device=self.device
        )
        self.problem_sizes2 = torch.empty(
            (self.e, 3), dtype=torch.int32, device=self.device
        )
        if self.cutlass_moe_type == CutlassMoEType.BlockscaledFP4:
            self.blockscale_offsets = torch.empty(
                (self.e + 1,), dtype=torch.int32, device=self.device
            )
        else:
            self.blockscale_offsets = None
        self.a_ptrs = torch.empty((self.e,), dtype=torch.int64, device=self.device)
        self.b_ptrs = torch.empty((self.e,), dtype=torch.int64, device=self.device)
        self.out_ptrs = torch.empty((self.e,), dtype=torch.int64, device=self.device)
        self.a_scales_ptrs = torch.empty(
            (self.e,), dtype=torch.int64, device=self.device
        )
        self.b_scales_ptrs = torch.empty(
            (self.e,), dtype=torch.int64, device=self.device
        )

    def to_gemm1_args(self) -> dict:
        return {
            "ab_strides": self.ab_strides_13,
            "c_strides": self.c_strides_13,
            "problem_sizes": self.problem_sizes1,
            "expert_offsets": self.expert_offsets[:-1],
            "blockscale_offsets": self.blockscale_offsets[:-1],
            #    "a_ptrs": self.a_ptrs,
            #    "b_ptrs": self.b_ptrs,
            #    "out_ptrs": self.out_ptrs,
            #    "a_scales_ptrs": self.a_scales_ptrs,
            #    "b_scales_ptrs": self.b_scales_ptrs,
        }

    def to_gemm2_args(self) -> dict:
        return {
            "ab_strides": self.ab_strides_2,
            "c_strides": self.c_strides_2,
            "problem_sizes": self.problem_sizes2,
            "expert_offsets": self.expert_offsets[:-1],
            "blockscale_offsets": self.blockscale_offsets[:-1],
            #    "a_ptrs": self.a_ptrs,
            #    "b_ptrs": self.b_ptrs,
            #    "out_ptrs": self.out_ptrs,
            #    "a_scales_ptrs": self.a_scales_ptrs,
            #    "b_scales_ptrs": self.b_scales_ptrs,
        }
