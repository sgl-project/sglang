import torch
import triton
import triton.language as tl

from sglang.srt.lora.backend.base_backend import BaseLoraBackend
from sglang.srt.lora.lora import LoraBatchInfo


@triton.jit
def _sgemm_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Matrix dimensions
    N,
    K,
    # Strides
    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,
    # Information on sequence lengths and weight id
    seg_lens,
    seg_indptr,
    weight_indices,
    # Meta parameters
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):

    # x: (s, K), s is the sum of sequence lengths
    # weights: (num_lora, N, K)
    # output: (s, N)

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    batch_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    w_index = tl.load(weight_indices + batch_id)
    seg_start = tl.load(seg_indptr + batch_id)

    # The tile in output matrix will have (pid_s, pid_n) as id
    # FIXME: might be replaced by super-grouping
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create pointers for the first block of x and weights[batch_id]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    x_ptrs = (x + seg_start * x_stride_0) + (
        s_offset[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iteate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len)
            and (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) and (n_offset[None, :] < N),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = (output + seg_start * output_stride_0) + (
        s_offset[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset[:, None] < seg_len) and (n_offset[None, :] < N)
    tl.store(output_ptr, partial_sum, mask=output_mask)


class TritonLoraBackend(BaseLoraBackend):

    def __init__(self, name: str, batch_info: LoraBatchInfo = None):
        super().__init__(name, batch_info)

    def run_sgemm(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:

        assert x.is_contiguous()
        assert weights.is_contiguous()
        assert len(x.shape) == 2
        assert len(weights.shape) == 3

        # x: (s, k)
        # weights: (num_lora, n, k)
        # output: (s, n)
        S = x.shape[0]
        N = weights.shape[-2]
        K = weights.shape[-1]
        assert x.shape[-1] == K

        # Block shapes
        # Autotuning tried but not effective
        BLOCK_S = 16
        BLOCK_K = 16
        BLOCK_N = 16

        grid = (
            triton.cdiv(self.batch_info.max_len, BLOCK_S) * triton.cdiv(N, BLOCK_N),
            self.batch_info.bs,
        )

        output = torch.zeros((S, N), device=x.device, dtype=x.dtype)
        _sgemm_kernel[grid](
            x,
            weights,
            output,
            N,
            K,
            x.stride(0),
            x.stride(1),
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            output.stride(0),
            output.stride(1),
            self.batch_info.seg_lens,
            self.batch_info.seg_indptr,
            self.batch_info.weight_indices,
            BLOCK_S,
            BLOCK_N,
            BLOCK_K,
        )
        return output

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        q_lora_b: torch.Tensor,
        kv_lora_b: torch.Tensor,
    ) -> torch.Tensor:

        # x: (s, input_dim)
        # qkv_lora_a: (num_lora, 3 * r, input_dim)
        # q_lora_b: (1, num_lora, output_dim_q, r)
        # kv_lora_b: (2, num_lora, output_dim_kv, r)

        # Shape of lora_a_output: (s, 3 * r)
        lora_a_output = self.run_sgemm(x=x, weights=qkv_lora_a)

        lora_rank = kv_lora_b.shape[-1]
        output_dim_q = q_lora_b.shape[-2]
        output_dim_kv = kv_lora_b.shape[-2]
        lora_output = torch.empty(
            (x.shape[0], output_dim_q + 2 * output_dim_kv),
            device=x.device,
            dtype=x.dtype,
        )

        # FIXME parallelize qkv
        # q
        lora_output[:, :output_dim_q] = self.run_sgemm(
            x=lora_a_output[:, :lora_rank].contiguous(), weights=q_lora_b[0]
        )

        # kv
        lora_output[:, output_dim_q : output_dim_q + output_dim_kv] = self.run_sgemm(
            x=lora_a_output[:, lora_rank : 2 * lora_rank].contiguous(),
            weights=kv_lora_b[0],
        )

        lora_output[
            :, output_dim_q + output_dim_kv : output_dim_q + 2 * output_dim_kv
        ] = self.run_sgemm(
            x=lora_a_output[:, 2 * lora_rank : 3 * lora_rank].contiguous(),
            weights=kv_lora_b[1],
        )

        return lora_output
