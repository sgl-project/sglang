import random
import time
import unittest

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd,
    redundant_attention,
)
from sglang.srt.layers.attention.triton_ops.extend_attention_int4kv import (
    extend_attention_fwd_int4kv,
)
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)


@triton.jit
def _fwd_kernel_destindex_copy_quantize_int4_kv(
    K,
    Out,
    Out_scale,
    stride_k_bs,
    stride_k_h,
    stride_k_g,
    stride_k_d,
    stride_o_bs,
    stride_o_h,
    stride_o_g,
    stride_o_d,
    stride_os_bs,
    stride_os_h,
    stride_os_g,
    group_size,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM // 2)

    dest_index = cur_index

    src_data_0 = tl.load(
        K
        + cur_index * stride_k_bs
        + cur_head * stride_k_h
        + offs_g[:, None] * stride_k_g
        + offs_d[None, :] * 2,
        mask=offs_g[:, None] < group_size,
        other=0.0,
    )
    src_data_1 = tl.load(
        K
        + cur_index * stride_k_bs
        + cur_head * stride_k_h
        + offs_g[:, None] * stride_k_g
        + offs_d[None, :] * 2
        + 1,
        mask=offs_g[:, None] < group_size,
        other=0.0,
    )

    abs_data_0 = tl.abs(src_data_0)
    abs_data_1 = tl.abs(src_data_1)

    data_scale = (
        tl.maximum(tl.max(abs_data_0, axis=1), tl.max(abs_data_1, axis=1)) / 7.0
    ).to(Out_scale.dtype.element_ty)
    q_src_data_0 = (src_data_0 / data_scale[:, None]).to(tl.int8)
    q_src_data_0 = tl.where(q_src_data_0 > 7, 7, q_src_data_0)
    q_src_data_0 = tl.where(q_src_data_0 < -7, -7, q_src_data_0)
    q_src_data_0 = q_src_data_0 + 8  # easy for dequant

    q_src_data_1 = (src_data_1 / data_scale[:, None]).to(tl.int8)
    q_src_data_1 = tl.where(q_src_data_1 > 7, 7, q_src_data_1)
    q_src_data_1 = tl.where(q_src_data_1 < -7, -7, q_src_data_1)
    q_src_data_1 = q_src_data_1 + 8  # easy for dequant

    low_4 = q_src_data_0 & 0xF
    high_4 = (q_src_data_1 & 0xF) << 4

    out_data = low_4 | high_4

    o_ptrs = (
        Out
        + dest_index * stride_o_bs
        + cur_head * stride_o_h
        + offs_g[:, None] * stride_o_g
        + offs_d[None, :]
    )
    os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g
    tl.store(o_ptrs, out_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, data_scale, mask=offs_g < group_size)
    return


@torch.no_grad()
def destindex_copy_quantize_int4kv(K, Out, Out_scale, quant_group_dim):
    bs_seq = K.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]

    assert (
        head_dim % quant_group_dim == 0
    ), "error head dim, can not been supported to copy quant kv"
    grid = (bs_seq, head_num)
    num_warps = 1

    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim

    K = K.view((K.shape[0], K.shape[1], group_size, group_dim))
    Out = Out.view(
        Out.shape[0], Out.shape[1], group_size, group_dim // 2
    )  # OUt 是 int8 类型， 两个int4组一个int8，所以 group_dim // 2

    _fwd_kernel_destindex_copy_quantize_int4_kv[grid](
        K,
        Out,
        Out_scale,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out.stride(3),
        Out_scale.stride(0),
        Out_scale.stride(1),
        Out_scale.stride(2),
        group_size,
        BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _bwd_kernel_destindex_dequantize_int4_kv(
    Quantized,
    Scale,
    Out,
    stride_q_bs,
    stride_q_h,
    stride_q_g,
    stride_q_d,
    stride_s_bs,
    stride_s_h,
    stride_s_g,
    stride_o_bs,
    stride_o_h,
    stride_o_g,
    stride_o_d,
    group_size,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM // 2)

    dest_index = cur_index

    # 加载量化数据
    q_data = tl.load(
        Quantized
        + cur_index * stride_q_bs
        + cur_head * stride_q_h
        + offs_g[:, None] * stride_q_g
        + offs_d[None, :],
        mask=offs_g[:, None] < group_size,
        other=0.0,
    )

    # 分离 int8 的低 4 位（int4 数据 0）和高 4 位（int4 数据 1）
    low_4 = q_data & 0xF
    high_4 = (q_data >> 4) & 0xF

    # 恢复 int4 到 [-7, 7] 的范围
    src_data_0 = low_4.to(tl.int8) - 8
    src_data_1 = high_4.to(tl.int8) - 8

    # 加载反量化比例因子（scale）
    scale = tl.load(
        Scale + dest_index * stride_s_bs + cur_head * stride_s_h + offs_g,
        mask=offs_g < group_size,
    )

    # 反量化
    dequant_data_0 = src_data_0 * scale[:, None]
    dequant_data_1 = src_data_1 * scale[:, None]

    # 存储反量化的 float 数据
    o_ptrs_0 = (
        Out
        + dest_index * stride_o_bs
        + cur_head * stride_o_h
        + offs_g[:, None] * stride_o_g
        + offs_d[None, :] * 2
    )
    o_ptrs_1 = (
        Out
        + dest_index * stride_o_bs
        + cur_head * stride_o_h
        + offs_g[:, None] * stride_o_g
        + offs_d[None, :] * 2
        + 1
    )

    tl.store(o_ptrs_0, dequant_data_0, mask=offs_g[:, None] < group_size)
    tl.store(o_ptrs_1, dequant_data_1, mask=offs_g[:, None] < group_size)
    return


@torch.no_grad()
def destindex_dequantize_int4kv(Quantized, Scale, Out, quant_group_dim):
    bs_seq = Quantized.shape[0]
    head_num = Quantized.shape[1]
    head_dim = Out.shape[2]

    assert (
        head_dim % quant_group_dim == 0
    ), "error head dim, can not been supported to copy dequant kv"
    grid = (bs_seq, head_num)
    num_warps = 1

    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim

    Quantized = Quantized.view(
        (Quantized.shape[0], Quantized.shape[1], group_size, group_dim // 2)
    )
    Scale = Scale.view((Scale.shape[0], Scale.shape[1], group_size))
    Out = Out.view(
        Out.shape[0], Out.shape[1], group_size, group_dim
    )  # Out 是 float16 类型，解压缩时需要两个 int4 恢复成 float16，所以 group_dim

    _bwd_kernel_destindex_dequantize_int4_kv[grid](
        Quantized,
        Scale,
        Out,
        Quantized.stride(0),
        Quantized.stride(1),
        Quantized.stride(2),
        Quantized.stride(3),
        Scale.stride(0),
        Scale.stride(1),
        Scale.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out.stride(3),
        group_size,
        BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


class TestExtendAttention(unittest.TestCase):

    def _set_all_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setUp(self):
        # Set seeds before each test method
        self._set_all_seeds(42)

    def _test_extend_attention_once(self, B, N_CTX_PRE, N_CTX_EXT, H_Q, H_KV, D):
        dtype = torch.float16

        if N_CTX_PRE == 0:
            b_seq_len_prefix = torch.zeros((B,), dtype=torch.int32, device="cuda")
        else:
            b_seq_len_prefix = torch.full(
                (B,), N_CTX_PRE, dtype=torch.int32, device="cuda"
            )

        b_seq_len_extend = torch.full((B,), N_CTX_EXT, dtype=torch.int32, device="cuda")

        b_seq_len = b_seq_len_prefix + b_seq_len_extend
        max_len_in_batch = torch.max(b_seq_len, 0)[0].item()

        b_req_idx = torch.arange(B, dtype=torch.int32, device="cuda")
        req_to_tokens = torch.empty(
            (B, max_len_in_batch), dtype=torch.int32, device="cuda"
        )
        b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
        b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device="cuda")
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
        for i in range(B):
            req_to_tokens[i, : b_seq_len[i]] = torch.arange(
                b_start_loc[i], b_start_loc[i] + b_seq_len[i]
            )

        total_token_num = torch.sum(b_seq_len).item()
        extend_token_num = torch.sum(b_seq_len_extend).item()
        k_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device="cuda"
        ).normal_(mean=0.1, std=0.2)
        v_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device="cuda"
        ).normal_(mean=0.1, std=0.2)

        k_buffer_int4 = torch.empty(
            (total_token_num, H_KV, D // 2), dtype=torch.int8, device="cuda"
        )
        v_buffer_int4 = torch.empty(
            (total_token_num, H_KV, D // 2), dtype=torch.int8, device="cuda"
        )

        quant_group_size = 32

        k_buffer_scales = torch.empty(
            (total_token_num, H_KV, D // quant_group_size), dtype=dtype, device="cuda"
        )
        v_buffer_scales = torch.empty(
            (total_token_num, H_KV, D // quant_group_size), dtype=dtype, device="cuda"
        )

        destindex_copy_quantize_int4kv(
            k_buffer,
            k_buffer_int4,
            k_buffer_scales,
            quant_group_size,
        )
        destindex_copy_quantize_int4kv(
            v_buffer,
            v_buffer_int4,
            v_buffer_scales,
            quant_group_size,
        )

        k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
        v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
        q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
        for i in range(B):
            extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
            extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
            extend_start = b_start_loc_extend[i]
            extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
            k_extend[extend_start:extend_end] = k_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            v_extend[extend_start:extend_end] = v_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            q_extend[extend_start:extend_end] = torch.empty(
                (b_seq_len_extend[i], H_Q, D), dtype=dtype, device="cuda"
            ).normal_(mean=0.1, std=0.2)

        o_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
        o_redundant = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device="cuda"
        )

        b_seq_len_extend = b_seq_len - b_seq_len_prefix
        b_start_loc_extend = torch.zeros_like(b_seq_len)
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
        max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()

        extend_attention_fwd_int4kv(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer_int4,
            v_buffer_int4,
            k_buffer_scales,
            v_buffer_scales,
            req_to_tokens,
            b_req_idx,
            b_seq_len,
            b_seq_len_extend,
            b_start_loc_extend,
            max_len_extend,
            quant_group_size,
        )

        redundant_attention(
            q_extend,
            o_redundant,
            k_buffer,
            v_buffer,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            b_seq_len_prefix,
            max_len_in_batch,
        )

        self.assertTrue(torch.allclose(o_extend, o_redundant, atol=5e-2, rtol=5e-2))

    def test_extend_attention(self):

        configs = [
            (2, 8, 16, 32, 32, 128),
        ]

        for B, SEQ_PRE, SEQ_EXT, H_Q, H_KV, D in configs:
            print(
                f"B, SEQ_PRE, SEQ_EXT, H_Q, H_KV, D: {B} {SEQ_PRE} {SEQ_EXT} {H_Q} {H_KV} {D}"
            )
            self._test_extend_attention_once(B, SEQ_PRE, SEQ_EXT, H_Q, H_KV, D)


if __name__ == "__main__":
    unittest.main()
