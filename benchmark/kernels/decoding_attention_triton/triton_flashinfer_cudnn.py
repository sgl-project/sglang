import itertools
import math

import cudnn
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


def decode_attention_flashinfer():
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=False
    )

    class FlashinferAttention(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q,
            kv_data,
            batch_size,
            kv_len,
            head_num_q,
            head_num_kv,
            head_dim,
            dtype,
        ):
            total_tokens = batch_size * kv_len
            kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len
            kv_indices = torch.arange(0, total_tokens).to(0).int()
            kv_last_page_len = torch.full(
                (batch_size,), 1, dtype=torch.int32, device="cuda"
            )

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

    return FlashinferAttention


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    else:
        raise ValueError("Unsupported tensor data type.")


def decode_attention_cudnn():

    cache = {}

    class CuDNNAttention(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q,
            kv_data,
            batch_size,
            kv_len,
            head_num_q,
            head_num_kv,
            head_dim,
            dtype,
        ):
            # Prepare data: continuous q,k,v
            dims_q = (batch_size, head_num_q, 1, head_dim)
            strides_q = (head_num_q * head_dim, head_dim, head_num_q * head_dim, 1)
            q_gpu = q.as_strided(dims_q, strides_q)
            o_gpu = (
                torch.empty(batch_size * head_num_q * head_dim)
                .half()
                .cuda()
                .as_strided(dims_q, strides_q)
            )

            dims_kv = (batch_size, head_num_kv, kv_len, head_dim)
            strides_kv = (
                kv_len * head_num_kv * head_dim,
                head_dim,
                head_num_kv * head_dim,
                1,
            )
            k_gpu = (
                kv_data[:, 0]
                .view(batch_size, kv_len, head_num_kv, head_dim)
                .contiguous()
                .as_strided(dims_kv, strides_kv)
            )
            v_gpu = (
                kv_data[:, 1]
                .view(batch_size, kv_len, head_num_kv, head_dim)
                .contiguous()
                .as_strided(dims_kv, strides_kv)
            )

            seq_len_q_gpu = torch.full((batch_size, 1, 1, 1), 1, device="cuda")
            seq_len_kv_gpu = torch.full((batch_size, 1, 1, 1), kv_len, device="cuda")
            attn_scale = 1.0 / (head_dim**0.5)

            # Prepare data: paged k,v
            block_size = 64
            blocks_per_batch = math.ceil(kv_len / block_size)
            # [num_blocks, head_num_kv, block_size, head_dim], num_blocks = batch_size * blocks_per_batch
            container_k_gpu = torch.cat(k_gpu.chunk(blocks_per_batch, dim=2), dim=0)
            container_v_gpu = torch.cat(v_gpu.chunk(blocks_per_batch, dim=2), dim=0)
            page_table_k_gpu = (
                torch.linspace(
                    0,
                    batch_size * blocks_per_batch - 1,
                    batch_size * blocks_per_batch,
                    device="cuda",
                    dtype=torch.int32,
                )
                .reshape(blocks_per_batch, 1, batch_size, 1)
                .transpose(0, 2)
            )
            page_table_v_gpu = page_table_k_gpu.clone()

            if "compiled_graph" not in cache:
                graph = cudnn.pygraph(
                    io_data_type=convert_to_cudnn_type(dtype),
                    intermediate_data_type=cudnn.data_type.FLOAT,
                    compute_data_type=cudnn.data_type.FLOAT,
                )

                q = graph.tensor_like(q_gpu)
                container_k = graph.tensor_like(container_k_gpu)
                container_v = graph.tensor_like(container_v_gpu)
                page_table_k = graph.tensor_like(page_table_k_gpu)
                page_table_v = graph.tensor_like(page_table_v_gpu)

                seq_len_q = graph.tensor_like(seq_len_q_gpu)
                seq_len_kv = graph.tensor_like(seq_len_kv_gpu)

                o, _ = graph.sdpa(
                    name="sdpa",
                    q=q,
                    k=container_k,  # Container K: non contiguous container with K blocks
                    v=container_v,  # Container V: non contiguous container with V blocks
                    is_inference=True,
                    attn_scale=attn_scale,
                    use_causal_mask=False,
                    use_padding_mask=True,
                    seq_len_q=seq_len_q,
                    seq_len_kv=seq_len_kv,
                    paged_attention_k_table=page_table_k,  # Page Table K: Tensor containing offsets to the container with K blocks
                    paged_attention_v_table=page_table_v,  # Page Table V: Tensor containing offsets to the container with V blocks
                    paged_attention_max_seq_len_kv=kv_len,  # The maximum sequence length for K caches (this is optional, but recommended)
                )

                o.set_output(True).set_dim(dims_q).set_stride(strides_q)

                graph.validate()
                graph.build_operation_graph()
                graph.create_execution_plans([cudnn.heur_mode.A])
                graph.check_support()
                graph.build_plans()

                cache["compiled_graph"] = graph

                workspace = torch.empty(
                    graph.get_workspace_size(), device="cuda", dtype=torch.uint8
                )
                cache["workspace"] = workspace

                cache["q"] = q
                cache["container_k"] = container_k
                cache["container_v"] = container_v
                cache["page_table_k"] = page_table_k
                cache["page_table_v"] = page_table_v
                cache["seq_len_q"] = seq_len_q
                cache["seq_len_kv"] = seq_len_kv
                cache["o"] = o

            variant_pack = {
                cache["q"]: q_gpu,
                cache["container_k"]: container_k_gpu,
                cache["container_v"]: container_v_gpu,
                cache["page_table_k"]: page_table_k_gpu,
                cache["page_table_v"]: page_table_v_gpu,
                cache["seq_len_q"]: seq_len_q_gpu,
                cache["seq_len_kv"]: seq_len_kv_gpu,
                cache["o"]: o_gpu,
            }

            cache["compiled_graph"].execute(variant_pack, cache["workspace"])

            return o_gpu.squeeze(dim=2)

    return CuDNNAttention


def calculate_diff():

    dtype = torch.float16
    batch_size = 4
    kv_len = 1024
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

    attn_flashinfer = decode_attention_flashinfer().apply
    output_flashinfer = attn_flashinfer(
        q, kv_data, batch_size, kv_len, head_num_q, head_num_kv, head_dim, dtype
    )

    attn_cudnn = decode_attention_cudnn().apply
    output_cudnn = attn_cudnn(
        q, kv_data, batch_size, kv_len, head_num_q, head_num_kv, head_dim, dtype
    )

    print(f"SGLang output={output_sglang}")
    print(f"FlashInfer output={output_flashinfer}")
    print(f"cuDNN output={output_cudnn}")
    if torch.allclose(output_sglang, output_flashinfer, atol=1e-2, rtol=1e-2):
        print("✅ SGLang[Triton] and FlashInfer match")
    else:
        print("❌ SGLang[Triton] and FlashInfer differ")

    if torch.allclose(output_sglang, output_cudnn, atol=1e-2, rtol=1e-2):
        print("✅ SGLang[Triton] and cuDNN match")
    else:
        print("❌ SGLang[Triton] and cuDNN differ")


head_dim = 128
dtype = torch.float16
batch_size_range = [2**i for i in range(0, 8, 2)]
kv_len_range = [2**i for i in range(6, 10, 1)]
head_num_range = [32, 64]
configs = list(itertools.product(head_num_range, batch_size_range, kv_len_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["head_num", "batch_size", "kv_len"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["sglang_triton", "flashinfer", "cudnn"],
        line_names=["SGLang[triton]", "FlashInfer", "cuDNN"],
        styles=[("green", "-"), ("red", "-"), ("blue", "-")],
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
    attn_flashinfer = decode_attention_flashinfer().apply
    attn_cudnn = decode_attention_cudnn().apply
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
            lambda: attn_flashinfer(
                q, kv_data, batch_size, kv_len, head_num_q, head_num_kv, head_dim, dtype
            ),
            quantiles=quantiles,
        )
    if provider == "cudnn":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: attn_cudnn(
                q, kv_data, batch_size, kv_len, head_num_q, head_num_kv, head_dim, dtype
            ),
            quantiles=quantiles,
        )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    calculate_diff()
    benchmark.run(print_data=True)
