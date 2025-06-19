import torch

def get_cutlass_moe_mm_data(
        topk_ids: torch.Tensor, expert_offsets: torch.Tensor,
        problem_sizes1: torch.Tensor, problem_sizes2: torch.Tensor,
        input_permutation: torch.Tensor, output_permutation: torch.Tensor,
        num_experts: int, n: int, k: int):
    """
    Prepare data necessary to perform CUTLASS grouped matrix multiplications
    used in CUTLASS-based fused MoE.

    The function takes in topk_ids (token-expert mapping) and uses it to
    compute:
    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation after the input is sorted with
                      input_permutation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes1, problem_sizes2: MxNxK sizes of each expert's
                                      multiplication in two grouped MMs used in
                                      the fused MoE operation.
    - input_permutation: Permutation that must be used to shuffle the input
                         before executing the MMs.
    - output_permutation: Permutation that must be used to shuffle the output
                          after executing the MMs.
    """
    torch.ops.sgl_kernel.get_cutlass_moe_mm_data.default(topk_ids, expert_offsets,
                                         problem_sizes1, problem_sizes2,
                                         input_permutation, output_permutation,
                                         num_experts, n, k)


def cutlass_moe_mm(out_tensors: torch.Tensor, a_tensors: torch.Tensor,
                   b_tensors: torch.Tensor, a_scales: torch.Tensor,
                   b_scales: torch.Tensor, expert_offsets: torch.Tensor,
                   problem_sizes: torch.Tensor, a_strides: torch.Tensor,
                   b_strides: torch.Tensor, c_strides: torch.Tensor):
    """
    A single grouped matrix multiplication used in CUTLASS-based fused MoE.
    The function executes fp8-quantized OUT = AB matrix multiplication.

    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes: MxNxK sizes of each expert's multiplication in two grouped
                     MMs used in the fused MoE operation.
    - a/b/c_strides: The data strides passed to grouped matrix multiplication.
    """
    torch.ops.sgl_kernel.cutlass_moe_mm.default(out_tensors, a_tensors, b_tensors, a_scales,
                                        b_scales, expert_offsets, problem_sizes,
                                        a_strides, b_strides, c_strides)


