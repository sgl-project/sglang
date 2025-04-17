import gc
import math
import os
import sys
import time
from dataclasses import dataclass

import cudnn
import numpy as np
import psutil
import torch

# 添加项目根目录到Python路径
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(project_root)

from sglang.srt.configs.model_config import AttentionArch, ModelConfig

# 只导入需要的外部backend
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs

global_graph_time = 0
global_forward_time = 0


class CuDNNBackend:
    def __init__(self):
        super().__init__()
        self.forward_metadata = None

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        pass

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert query.shape[0] == seq_lens.shape[0], "batch size must be the same"

        start_time = time.perf_counter()

        max_seq_len = k_cache.shape[0]
        # Convert into CuDNN Query format (B, H, S, D)
        # where B is number of queries and S is sequence per query (1 in decoding)
        # [num_tokens, num_heads, head_size] -> [num_token, num_heads, 1,  head_size]
        query = query.unsqueeze(1).movedim(1, 2)

        # heads, tokens, head size
        # The tokens of queries are indexed by req_to_token
        s, h, d = k_cache.shape
        # Block Size of Paged Cache, 1 since only one token per block
        b = 1

        # Radix Attention use KVCache of Block Size 1

        # TODO: determine data type
        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.HALF,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        print(graph)

        # query contains num_tokens queries batched togather
        q_gpu = query
        q = graph.tensor_like(q_gpu)

        # get the request id of each query up to t
        # per_req_tokens = req_to_token[req_pool_indices, :seq_len_kv]

        # get the token location in kvcache, only up to seq_len_kv is valid
        # cudnn required shape: (num_block, 1, ceil(s/num_block), 1)
        print(
            "req index shape: ",
            req_pool_indices.shape,
            "req to token shape: ",
            req_to_token.shape,
        )
        per_req_tokens = req_to_token[req_pool_indices, :]
        print("per req token shape: ", per_req_tokens.shape)

        # get the kv cache with request id
        # container: num_blocks, num heads, tokens_per_block, dim
        # TODO: permute for correctness
        container_k_gpu = k_cache.view(s, h, b, d)
        print("cache shape: ", container_k_gpu.shape)
        container_v_gpu = v_cache.view(s, h, b, d)

        container_k = graph.tensor_like(container_k_gpu)
        container_v = graph.tensor_like(container_v_gpu)

        page_table_k_gpu = per_req_tokens.view(
            per_req_tokens.shape[0], 1, per_req_tokens.shape[1], 1
        )
        print("paged table k shape: ", page_table_k_gpu.shape)
        page_table_v_gpu = per_req_tokens.view(
            per_req_tokens.shape[0], 1, per_req_tokens.shape[1], 1
        )
        print("page table v shape: ", page_table_v_gpu.shape)
        page_table_k = graph.tensor_like(page_table_k_gpu)
        page_table_v = graph.tensor_like(page_table_v_gpu)

        seq_lens_kv = seq_lens.view(seq_lens.shape[0], 1, 1, 1)

        seq_lens_q = torch.ones_like(seq_lens_kv)

        seq_len_q_tensor_info = graph.tensor_like(seq_lens_q)
        seq_len_kv_tensor_info = graph.tensor_like(seq_lens_kv)

        o, _ = graph.sdpa(
            name="sdpa",
            q=q,
            k=container_k,  # Container K: non contiguous container with K blocks
            v=container_v,  # Container V: non contiguous container with V blocks
            is_inference=True,
            attn_scale=scaling,
            use_causal_mask=causal,
            use_padding_mask=True,
            seq_len_q=seq_len_q_tensor_info,
            seq_len_kv=seq_len_kv_tensor_info,
            paged_attention_k_table=page_table_k,  # Page Table K: Tensor containing offsets to the container with K blocks
            paged_attention_v_table=page_table_v,  # Page Table V: Tensor containing offsets to the container with V blocks
            paged_attention_max_seq_len_kv=max_seq_len,  # The maximum sequence length for K caches (this is optional, but recommended)
        )
        logging.info(graph)

        output = output.view(*query.shape)
        dims = output.shape
        strides = output.stride()

        o.set_output(True).set_dim(dims).set_stride(strides)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        build_graph_time = time.perf_counter()

        variant_pack = {
            q: q_gpu,
            container_k: container_k_gpu,
            container_v: container_v_gpu,
            page_table_k: page_table_k_gpu,
            page_table_v: page_table_v_gpu,
            seq_len_q_tensor_info: seq_lens_q,
            seq_len_kv_tensor_info: seq_lens_kv,
            o: output,
        }

        workspace = torch.empty(
            graph.get_workspace_size(), device="cuda", dtype=torch.uint8
        )
        graph.execute(variant_pack, workspace)
        print(output.shape)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        global_graph_time = build_graph_time - start_time
        global_forward_time = end_time - build_graph_time

        print(f"Graph Construction Time: {global_graph_time}")
        print(f"Forward Time: {global_forward_time}")

        return output


@dataclass
class InputParameters:
    """Small configuration parameters"""

    num_token: int = 32
    num_heads: int = 32
    head_size: int = 128
    max_total_num_tokens: int = 32768
    max_num_reqs: int = 32
    num_seqs: int = 32
    max_context_lenght: int = 1024


@dataclass
class InputParametersLarge:
    """Large configuration parameters"""

    num_token: int = 512
    num_heads: int = 32
    head_size: int = 128
    max_total_num_tokens: int = 32768
    max_num_reqs: int = 32
    num_seqs: int = 32
    max_context_lenght: int = 8192


def measure_memory_usage():
    """Measure current GPU memory usage"""
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB


def run_backend(backend, inputs, name=""):
    """Run a single backend and measure performance"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Measure total time
    start_total = time.perf_counter()

    # Run the backend
    output = backend._run_sdpa_forward_extend(**inputs)

    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start_total) * 1000  # ms

    # Get memory usage
    memory_used = measure_memory_usage()

    # For CuDNN, we expect graph construction and forward times
    if name == "CuDNN" and isinstance(output, dict):
        graph_time = output.get("graph_construction_time", 0) * 1000
        forward_time = output.get("forward_time", 0) * 1000
    else:
        # For other backends, we just split the total time
        graph_time = 0
        forward_time = total_time

    return {
        "graph_time": graph_time,
        "forward_time": forward_time,
        "total_time": total_time,
        "memory": memory_used,
    }


def create_model_runner():
    # Create basic configuration
    model_config = ModelConfig(
        model_path="facebook/opt-125m",
    )

    # Create server arguments
    server_args = ServerArgs(
        model_path="facebook/opt-125m",
        tokenizer_path="facebook/opt-125m",
        device="cuda",
        attention_backend="flashinfer",  # Initialize with flashinfer backend
        disable_cuda_graph=True,  # Disable CUDA graph to simplify testing
        mem_fraction_static=0.95,  # Use most memory for static allocation
    )

    # Initialize ModelRunner
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=0.95,
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
        nccl_port=29500,
        server_args=server_args,
    )

    return runner


def get_backends():
    runner = create_model_runner()
    return {
        "CuDNN": CuDNNBackend(),
        "Flash Infer": FlashInferAttnBackend(runner),
        "Torch Native": TorchNativeAttnBackend(),
    }


def benchmark_performance_metrics():
    """Comprehensive performance evaluation of different backends"""

    configs = {"Small": InputParameters(), "Large": InputParametersLarge()}

    backends = get_backends()

    metrics = {
        "Graph Construction Time": {"Small": [], "Large": []},
        "Forward Time": {"Small": [], "Large": []},
        "Total Latency": {"Small": [], "Large": []},
        "Memory Usage (MB)": {"Small": [], "Large": []},
    }

    num_runs = 10
    warmup_runs = 3

    for config_name, params in configs.items():
        print(f"\n=== Benchmarking {config_name} Configuration ===")

        # Initialize input tensors
        query = (
            torch.randn([params.num_token, params.num_heads, params.head_size])
            .half()
            .cuda()
        )
        output = (
            torch.randn([params.num_token, params.num_heads, params.head_size])
            .half()
            .cuda()
        )
        k_cache = (
            torch.randn(
                [params.max_total_num_tokens, params.num_heads, params.head_size]
            )
            .half()
            .cuda()
        )
        v_cache = (
            torch.randn(
                [params.max_total_num_tokens, params.num_heads, params.head_size]
            )
            .half()
            .cuda()
        )

        req_pool_indices = torch.randint(
            low=0, high=params.max_num_reqs, size=[params.num_seqs], dtype=torch.int32
        ).cuda()
        extend_prefix_lens = torch.randint(
            low=0, high=5, size=[params.num_seqs], dtype=torch.int32
        ).cuda()
        extend_seq_lens = torch.ones(params.num_seqs, dtype=torch.int32).cuda() * (
            params.num_token // params.num_seqs
        )
        req_to_token = torch.randint(
            low=0,
            high=params.num_token,
            size=[params.max_num_reqs, params.max_context_lenght],
            dtype=torch.int32,
        ).cuda()
        seq_lens = (extend_prefix_lens + extend_seq_lens).cuda()
        scaling = 1 / math.sqrt(params.head_size)

        inputs = {
            "query": query,
            "output": output,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "req_to_token": req_to_token,
            "req_pool_indices": req_pool_indices,
            "seq_lens": seq_lens,
            "extend_prefix_lens": extend_prefix_lens,
            "extend_seq_lens": extend_seq_lens,
            "scaling": scaling,
        }

        # Warmup
        print("Performing warmup runs...")
        for backend_name, backend in backends.items():
            for _ in range(warmup_runs):
                _ = backend._run_sdpa_forward_extend(**inputs)
                torch.cuda.synchronize()

        # Benchmark runs
        print(f"\nRunning {num_runs} iterations for each backend...")

        for backend_name, backend in backends.items():
            print(f"\nTesting {backend_name}...")
            for i in range(num_runs):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Run and measure
                result = backend._run_sdpa_forward_extend(**inputs)
                torch.cuda.synchronize()

                # Get timing information
                if isinstance(result, dict) and "graph_construction_time" in result:
                    graph_time = global_graph_time * 1000  # convert to ms
                    forward_time = global_forward_time * 1000
                else:
                    # For non-CuDNN backends
                    graph_time = 0
                    forward_time = 0  # Need to implement timing for other backends

                metrics["Graph Construction Time"][config_name].append(graph_time)
                metrics["Forward Time"][config_name].append(forward_time)
                metrics["Total Latency"][config_name].append(graph_time + forward_time)
                metrics["Memory Usage (MB)"][config_name].append(
                    torch.cuda.max_memory_allocated() / 1024**2
                )

    # Print results
    print("\n=== Performance Evaluation Results ===")
    print(f"Results averaged over {num_runs} runs")
    print("-" * 80)

    for metric in metrics:
        print(f"\n{metric}:")
        print("-" * 60)
        print(f"{'Backend':<20} {'Small':>15} {'Large':>15}")
        print("-" * 60)

        for backend_name in backends:
            small_avg = np.mean(metrics[metric]["Small"])
            large_avg = np.mean(metrics[metric]["Large"])
            print(f"{backend_name:<20} {small_avg:>15.2f} {large_avg:>15.2f}")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability()[0] >= 1

    print("Starting benchmark...")
    benchmark_performance_metrics()
