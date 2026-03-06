from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch


class CutlassMoEQuantType(Enum):
    """
    Enum for the different quantization types supported by cutlass moe operations.
    """

    # Quantization types
    W4A8 = auto()
    BlockscaledFP8 = auto()
    BlockscaledFP4 = auto()


class CutlassMoEType(Enum):
    """
    Enum for the different execution modes supported by cutlass moe operations.
    """

    # DeepEP distributed execution modes
    DeepEP_LL = auto()
    DeepEP_Normal = auto()


@dataclass
class CutlassMoEParams:
    """
    Parameters for the cutlass moe operation.
    """

    # Quantization type as defined above
    quant_type: CutlassMoEQuantType

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
    ab_strides_13: Optional[torch.Tensor] = None
    ab_strides_2: Optional[torch.Tensor] = None
    c_strides_13: Optional[torch.Tensor] = None
    c_strides_2: Optional[torch.Tensor] = None

    # m: Total number of tokens
    # n: intermediate size per partition
    # k: hidden size per expert
    # e: Number of experts
    # device: Device to run computation on and store tensors
    m: int = 0
    intermediate_size_per_partition: int = 0
    hidden_size: int = 0
    num_experts: int = 0
    device: Optional[torch.device] = None

    # Pointers container for calculating offsets of the input activations for each expert
    # a_ptrs: [e] dtype: int64
    a_ptrs: Optional[torch.Tensor] = None

    # Pointers container for calculating offsets of the input weights for each expert
    # b_ptrs: [e] dtype: int64
    b_ptrs: Optional[torch.Tensor] = None

    # Pointers container for calculating offsets of the output activations for each expert
    # out_ptrs: [e] dtype: int64
    out_ptrs: Optional[torch.Tensor] = None
    # Pointers container for calculating offsets of the input scales for each expert
    # a_scales_ptrs: [e] dtype: int64
    # b_scales_ptrs: [e] dtype: int64
    a_scales_ptrs: Optional[torch.Tensor] = None
    b_scales_ptrs: Optional[torch.Tensor] = None

    # Offsets that mark at which token index each expert begins its computation
    # The number of tokens computed with expert E is expert_offsets[E + 1] - expert_offsets[E]
    # expert_offsets: [e+1] dtype: int32
    expert_offsets: Optional[torch.Tensor] = None

    # Problem size: (num_experts, (m,2n,k)) for first GEMM
    # problem_sizes1: [e, 3] dtype: int32
    # Problem size: (num_experts, (m,n,k)) for second GEMM
    # problem_sizes2: [e, 3] dtype: int32
    problem_sizes1: Optional[torch.Tensor] = None
    problem_sizes2: Optional[torch.Tensor] = None
    # Similar to expert_offsets, but for blockscales for FP4 blockscaled Group GEMM
    blockscale_offsets: Optional[torch.Tensor] = None

    # W4A8 specific fields
    a_strides1: Optional[torch.Tensor] = None
    b_strides1: Optional[torch.Tensor] = None
    c_strides1: Optional[torch.Tensor] = None
    a_strides2: Optional[torch.Tensor] = None
    b_strides2: Optional[torch.Tensor] = None
    c_strides2: Optional[torch.Tensor] = None
    s_strides13: Optional[torch.Tensor] = None
    s_strides2: Optional[torch.Tensor] = None
    workspace: Optional[torch.Tensor] = None

    def __init__(
        self,
        quant_type: CutlassMoEQuantType,
        device: torch.device,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_size: int,
    ):
        self.quant_type = quant_type
        self.device = device
        self.num_experts = num_experts
        self.intermediate_size_per_partition = intermediate_size_per_partition
        self.hidden_size = hidden_size
        self.n = self.intermediate_size_per_partition
        self.k = self.hidden_size
        self.e = self.num_experts

        if self.quant_type == CutlassMoEQuantType.W4A8:
            self.a_strides1 = torch.full(
                (self.e, 3), self.k, dtype=torch.int64, device=self.device
            )
            self.c_strides1 = torch.full(
                (self.e, 3), 2 * self.n, dtype=torch.int64, device=self.device
            )
            self.a_strides2 = torch.full(
                (self.e, 3), self.n, dtype=torch.int64, device=self.device
            )
            self.c_strides2 = torch.full(
                (self.e, 3), self.k, dtype=torch.int64, device=self.device
            )
            self.b_strides1 = self.a_strides1
            self.s_strides13 = self.c_strides1
            self.b_strides2 = self.a_strides2
            self.s_strides2 = self.c_strides2
        else:
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

        if self.quant_type == CutlassMoEQuantType.BlockscaledFP4:
            self.blockscale_offsets = torch.empty(
                (self.e + 1,), dtype=torch.int32, device=self.device
            )
        else:
            self.blockscale_offsets = None

        if self.quant_type == CutlassMoEQuantType.BlockscaledFP8:
            self.workspace = torch.empty(90000, device=self.device, dtype=torch.uint8)
            self.a_ptrs = torch.empty((self.e,), dtype=torch.int64, device=self.device)
            self.b_ptrs = torch.empty((self.e,), dtype=torch.int64, device=self.device)
            self.out_ptrs = torch.empty(
                (self.e,), dtype=torch.int64, device=self.device
            )
            self.a_scales_ptrs = torch.empty(
                (self.e,), dtype=torch.int64, device=self.device
            )
            self.b_scales_ptrs = torch.empty(
                (self.e,), dtype=torch.int64, device=self.device
            )
        else:
            self.workspace = None
            self.a_ptrs = None
            self.b_ptrs = None
            self.out_ptrs = None
            self.a_scales_ptrs = None
            self.b_scales_ptrs = None

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
