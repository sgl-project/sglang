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


@triton.jit
def _qkv_lora_b_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Parameters of size
    K,  # K = R
    N_Q,
    N_KV,
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
    # Offsets of q/k/v slice on output dimension
    n_indptr,
    # Meta parameters
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # This kernel packs 3 sgemms (q/k/v) into a single kernel.

    # x: (s, 3 * K), s is the sum of sequence lengths, K equals to lora rank
    # weights: (num_lora, N_Q + 2 * N_KV, K)
    # output: (s, N_Q + 2 * N_KV)

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len.
    # qkv_id decides which of q,k,v to compute (0: q, 1: k, 2: v)
    batch_id = tl.program_id(axis=2)
    qkv_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    w_index = tl.load(weight_indices + batch_id)
    seg_start = tl.load(seg_indptr + batch_id)
    n_start = tl.load(n_indptr + qkv_id)
    n_size = tl.load(n_indptr + qkv_id + 1) - n_start

    # The tile in output matrix will have (pid_s, pid_n) as id
    # FIXME: might be replaced by super-grouping
    num_pid_n = tl.cdiv(max(N_Q, N_KV), BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create pointers for the first block of x and weights[batch_id][n_start: n_end][:]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    x_ptrs = (x + seg_start * x_stride_0 + (qkv_id * K) * x_stride_1) + (
        s_offset[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0 + n_start * w_stride_1) + (
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
            mask=(k_offset[:, None] < K - k * BLOCK_K) and (n_offset[None, :] < n_size),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = (output + seg_start * output_stride_0 + n_start * output_stride_1) + (
        s_offset[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset[:, None] < seg_len) and (n_offset[None, :] < n_size)
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

        output = torch.empty((S, N), device=x.device, dtype=x.dtype)
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

        assert x.is_contiguous()
        assert qkv_lora_a.is_contiguous()
        assert q_lora_b.is_contiguous()
        assert kv_lora_b.is_contiguous
        assert len(x.shape) == 2
        assert len(qkv_lora_a.shape) == 3
        assert len(q_lora_b.shape) == 4
        assert len(kv_lora_b.shape) == 4

        # Get dims
        s = x.shape[0]
        input_dim = x.shape[1]
        output_dim_q = q_lora_b.shape[-2]
        output_dim_kv = kv_lora_b.shape[-2]
        r = q_lora_b.shape[-1]
        middle_dim = 3 * r
        output_dim = output_dim_q + 2 * output_dim_kv
        assert qkv_lora_a.shape[-2] == middle_dim
        assert qkv_lora_a.shape[-1] == input_dim
        assert kv_lora_b.shape[-1] == r

        # Compute lora_a_output = sgemm(x, qkv_lora_a)
        # shape of lora_a_output: (s, middle_dim)
        BLOCK_S = 16
        BLOCK_IN = 64
        BLOCK_R = 16

        grid_a = (
            triton.cdiv(self.batch_info.max_len, BLOCK_S)
            * triton.cdiv(middle_dim, BLOCK_R),
            self.batch_info.bs,
        )

        lora_a_output = torch.empty((s, middle_dim), device=x.device, dtype=x.dtype)
        _sgemm_kernel[grid_a](
            x,
            qkv_lora_a,
            lora_a_output,
            middle_dim,
            input_dim,
            x.stride(0),
            x.stride(1),
            qkv_lora_a.stride(0),
            qkv_lora_a.stride(1),
            qkv_lora_a.stride(2),
            lora_a_output.stride(0),
            lora_a_output.stride(1),
            self.batch_info.seg_lens,
            self.batch_info.seg_indptr,
            self.batch_info.weight_indices,
            BLOCK_S,
            BLOCK_R,
            BLOCK_IN,
        )

        # Compute lora_output with shape (s, output_dim) as follows:
        # lora_output[:, :output_dim_q] = sgemm(lora_output_a[:, :r], q_lora_b[0])
        # lora_output[:, output_dim_q: output_dim_q + output_dim_kv]
        #      = sgemm(lora_output_a[:, r: 2 * r], kv_lora_b[0])
        # lora_output[:, output_dim_q + output_dim_kv: ]
        #      = sgemm(lora_output_a[:, 2 * r: 3 * r], kv_lora_b[1])
        BLOCK_S = 16
        BLOCK_R = 16
        BLOCK_OUT = 64

        grid_b = (
            triton.cdiv(self.batch_info.max_len, BLOCK_S)
            * triton.cdiv(max(output_dim_q, output_dim_kv), BLOCK_OUT),
            3,  # this dimension decides current block computes on q, k or v
            self.batch_info.bs,
        )

        # w_lora_b with shape (num_lora, output_dim_q + 2 * output_dim_kv, r) is passed to kernel
        w_lora_b = torch.cat(
            (q_lora_b[0], kv_lora_b[0], kv_lora_b[1]), dim=-2
        ).contiguous()
        lora_output = torch.empty((s, output_dim), device=x.device, dtype=x.dtype)
        n_indptr = torch.tensor(
            [
                0,
                output_dim_q,
                output_dim_q + output_dim_kv,
                output_dim_q + 2 * output_dim_kv,
            ],
            dtype=torch.int32,
            device=x.device,
        )

        _qkv_lora_b_kernel[grid_b](
            lora_a_output,
            w_lora_b,
            lora_output,
            r,
            output_dim_q,
            output_dim_kv,
            lora_a_output.stride(0),
            lora_a_output.stride(1),
            w_lora_b.stride(0),
            w_lora_b.stride(1),
            w_lora_b.stride(2),
            lora_output.stride(0),
            lora_output.stride(1),
            self.batch_info.seg_lens,
            self.batch_info.seg_indptr,
            self.batch_info.weight_indices,
            n_indptr,
            BLOCK_S,
            BLOCK_OUT,
            BLOCK_R,
        )

        return lora_output
