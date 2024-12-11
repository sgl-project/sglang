import itertools

import torch
import triton
import triton.language as tl
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd


def decode_attention_sglang(
    q, kv_data, batch_size, kv_len, head_num_q, head_num_kv, head_dim, num_kv_splits
):

    k_buffer = kv_data[:, 0].view(-1, head_num_kv, head_dim).contiguous()
    v_buffer = kv_data[:, 1].view(-1, head_num_kv, head_dim).contiguous()
    o = torch.empty_like(q)
    total_tokens = batch_size * kv_len
    req_to_token = torch.arange(0, total_tokens).to(0).int().view(batch_size, kv_len)
    b_req_idx = torch.arange(0, batch_size).to(0).int()
    b_seq_len = torch.full((batch_size,), kv_len, dtype=torch.int32, device="cuda")
    max_len_in_batch = kv_len
    sm_scale = 1.0 / (head_dim**0.5)

    attn_logits = torch.empty(
        (batch_size, head_num_q, num_kv_splits, head_dim + 1),
        dtype=torch.float32,
        device="cuda",
    )

    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o,
        req_to_token,
        b_req_idx,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
    )

    return o


def decode_attention_flashinfer(
    q, kv_data, batch_size, kv_len, head_num_q, head_num_kv, head_dim, dtype
):

    total_tokens = batch_size * kv_len
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len
    kv_indices = torch.arange(0, total_tokens).to(0).int()
    kv_last_page_len = torch.full((batch_size,), 1, dtype=torch.int32, device="cuda")

    flashinfer_decode_wrapper.end_forward()
    flashinfer_decode_wrapper.begin_forward(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        head_num_q,
        head_num_kv,
        head_dim,
        1,
        pos_encoding_mode="NONE",
        data_type=dtype,
    )
    o = flashinfer_decode_wrapper.forward(
        q.contiguous().view(-1, head_num_q, head_dim), kv_data
    )

    return o


def calculate_diff():

    dtype = torch.bfloat16
    batch_size = 4
    kv_len = 16
    head_num_q = 32
    head_num_kv = 32
    head_dim = 128

    q = torch.randn(batch_size, head_num_q, head_dim, dtype=dtype, device="cuda")
    kv_data = torch.randn(
        batch_size * kv_len, 2, head_num_kv, head_dim, dtype=dtype, device="cuda"
    )

    output_sglang = decode_attention_sglang(
        q,
        kv_data,
        batch_size,
        kv_len,
        head_num_q,
        head_num_kv,
        head_dim,
        num_kv_splits=8,
    )
    output_flashinfer = decode_attention_flashinfer(
        q, kv_data, batch_size, kv_len, head_num_q, head_num_kv, head_dim, dtype=dtype
    )

    print(f"SGLang output={output_sglang}")
    print(f"FlashInfer output={output_flashinfer}")
    if torch.allclose(output_sglang, output_flashinfer, atol=1e-2, rtol=1e-2):
        print("✅ SGLang[Triton] and FlashInfer match")
    else:
        print("❌ SGLang[Triton] and FlashInfer differ")


head_dim = 128
dtype = torch.float16
batch_size_range = [2**i for i in range(0, 8, 2)]
kv_len_range = [2**i for i in range(6, 13, 1)]
head_num_range = [32, 64]
configs = list(itertools.product(head_num_range, batch_size_range, kv_len_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["head_num", "batch_size", "kv_len"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["sglang_triton", "flashinfer"],
        line_names=["SGLang[triton]", "FlashInfer"],
        styles=[("green", "-"), ("red", "-")],
        ylabel="us",
        plot_name="decode-attention-performance",
        args={},
    )
)
def benchmark(head_num, batch_size, kv_len, provider):
    head_num_q = head_num_kv = head_num
    q = torch.randn(batch_size, head_num_q, head_dim, dtype=dtype, device="cuda")
    kv_data = torch.randn(
        batch_size * kv_len, 2, head_num_kv, head_dim, dtype=dtype, device="cuda"
    )
    quantiles = [0.5, 0.2, 0.8]
    if provider == "sglang_triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: decode_attention_sglang(
                q,
                kv_data,
                batch_size,
                kv_len,
                head_num_q,
                head_num_kv,
                head_dim,
                num_kv_splits=8,
            ),
            quantiles=quantiles,
        )
    if provider == "flashinfer":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: decode_attention_flashinfer(
                q, kv_data, batch_size, kv_len, head_num_q, head_num_kv, head_dim, dtype
            ),
            quantiles=quantiles,
        )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    global flashinfer_decode_wrapper
    flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=False
    )

    calculate_diff()

    benchmark.run(print_data=True)
