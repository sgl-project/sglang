# python3 benchmark/fbgemm/benchmark_fbgemm_grouped_gemm.py --model Qwen/Qwen2-57B-A14B-Instruct --tp-size 4 --use-fp8-w8a8
import argparse

import torch
import triton
from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
    quantize_fp8_row,
    triton_quantize_fp8_row,
)
from fbgemm_gpu.experimental.gemm.triton_gemm.grouped_gemm import (
    grouped_gemm as fbgemm_grouped_gemm,
)
from fbgemm_gpu.experimental.gemm.triton_gemm.grouped_gemm import (
    grouped_gemm_fp8_rowwise as fbgemm_grouped_gemm_fp8_rowwise,
)
from transformers import AutoConfig

from sglang.srt.layers.moe.ep_moe.kernels import (
    grouped_gemm_triton as sglang_grouped_gemm,
)


def get_model_config(model_name: str, tp_size: int):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    if config.architectures[0] == "DbrxForCausalLM":
        num_groups = config.ffn_config.moe_num_experts
        intermediate_size = config.ffn_config.ffn_hidden_size
    elif config.architectures[0] == "JambaForCausalLM":
        num_groups = config.num_experts
        intermediate_size = config.intermediate_size
    elif config.architectures[0] == "Qwen2MoeForCausalLM":
        num_groups = config.num_experts
        intermediate_size = config.moe_intermediate_size
    elif config.architectures[0] == "Qwen3MoeForCausalLM":
        num_groups = config.num_experts
        intermediate_size = config.moe_intermediate_size
    elif config.architectures[0] in [
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
    ]:
        num_groups = config.n_routed_experts
        intermediate_size = config.moe_intermediate_size
    elif config.architectures[0] == "Llama4ForConditionalGeneration":
        num_groups = config.text_config.num_local_experts
        intermediate_size = config.text_config.intermediate_size
    elif config.architectures[0] in [
        "Grok1ForCausalLM",
        "Grok1ImgGen",
        "Grok1AForCausalLM",
    ]:
        num_groups = config.num_local_experts
        intermediate_size = config.moe_intermediate_size
    else:
        num_groups = config.num_local_experts
        intermediate_size = config.intermediate_size

    shape_configs = {
        "num_groups": num_groups,
        "hidden_size": config.hidden_size,
        "intermediate_size": intermediate_size,
        "dtype": config.torch_dtype,
    }
    print(f"{shape_configs=}")
    return shape_configs


def create_test_data(batch_size, num_groups, hidden_size, intermediate_size):
    torch.manual_seed(42)

    tokens_per_group = batch_size // num_groups
    m_sizes = torch.full(
        (num_groups,), tokens_per_group, dtype=torch.int32, device="cuda"
    )

    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="cuda")

    base_weights = torch.randn(
        num_groups, intermediate_size, hidden_size, dtype=torch.bfloat16, device="cuda"
    )

    w_fbgemm = base_weights.reshape(num_groups * intermediate_size, hidden_size)
    w_sglang = base_weights

    c_fbgemm = torch.empty(
        batch_size, intermediate_size, dtype=torch.bfloat16, device="cuda"
    )
    c_sglang = torch.empty(
        batch_size, intermediate_size, dtype=torch.bfloat16, device="cuda"
    )

    seg_indptr = torch.zeros(num_groups + 1, dtype=torch.int32, device="cuda")
    for i in range(1, num_groups + 1):
        seg_indptr[i] = seg_indptr[i - 1] + tokens_per_group

    weight_indices = torch.arange(num_groups, dtype=torch.int32, device="cuda")

    return (
        x,
        w_fbgemm,
        w_sglang,
        c_fbgemm,
        c_sglang,
        m_sizes,
        seg_indptr,
        weight_indices,
    )


def create_fp8_test_data(
    batch_size, num_groups, hidden_size, intermediate_size, backend="triton"
):
    """
    Create test data for FP8 grouped GEMM operations.

    Args:
        batch_size: Total batch size
        num_groups: Number of groups
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate dimension size
        backend: "triton" for Triton GEMM, "cutlass" for CUTLASS GEMM

    Returns:
        For triton: (x_fp8, w_fp8, m_sizes, x_scale, w_scale)
        For cutlass: (x, wq, w_scale, m_sizes)
    """
    torch.manual_seed(42)

    tokens_per_group = batch_size // num_groups

    # Create weight matrices for each group
    w_list = []
    for _ in range(num_groups):
        w = torch.randn(
            intermediate_size, hidden_size, dtype=torch.float16, device="cuda"
        )
        w_list.append(w)

    # Quantize weights using quantize_fp8_row for each group
    wq_list, w_scale_list = zip(*[quantize_fp8_row(w) for w in w_list])

    if backend == "triton":
        # Triton format: concatenated weights
        w_fp8 = torch.concat(wq_list, dim=0).contiguous()
        w_scale = torch.concat(w_scale_list, dim=0).contiguous()

        # Create m_sizes as int32 for triton
        m_sizes = torch.full(
            (num_groups,), tokens_per_group, dtype=torch.int32, device="cuda"
        )

        # Create and quantize input
        x_fp16 = torch.randn(
            batch_size, hidden_size, dtype=torch.float16, device="cuda"
        )
        x_fp8, x_scale = triton_quantize_fp8_row(x_fp16)
        x_scale = x_scale.view(batch_size, -1)

        return x_fp8, w_fp8, m_sizes, x_scale, w_scale

    elif backend == "cutlass":
        # CUTLASS format: stacked weights
        wq = torch.stack(wq_list, dim=0).contiguous()
        w_scale = torch.stack(w_scale_list, dim=0).contiguous()

        # Create m_sizes as int64 for cutlass
        m_values = [tokens_per_group] * num_groups
        m_sizes = torch.tensor(m_values).to(dtype=torch.int64, device="cuda")

        # Create input data - separate for each group then concat
        x_list = []
        for _ in range(num_groups):
            x = torch.randn(
                tokens_per_group, hidden_size, dtype=torch.float16, device="cuda"
            )
            x_list.append(x)

        # Concatenate inputs into single tensor
        x = torch.concat(x_list, dim=0).contiguous()

        return x, wq, w_scale, m_sizes

    else:
        raise ValueError(f"Unsupported backend: {backend}")


def calculate_memory_bandwidth(m_sizes, hidden_size, intermediate_size, dtype):
    """
    Calculate memory bandwidth based on accessed expert weights.

    Args:
        m_sizes: Tensor containing batch sizes for each group
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate dimension size
        dtype: Data type of weights

    Returns:
        Memory size in bytes for accessed expert weights
    """
    # Count non-zero groups (active experts)
    if hasattr(m_sizes, "cpu"):
        active_experts = torch.count_nonzero(m_sizes).item()
    else:
        active_experts = sum(1 for m in m_sizes if m > 0)

    # Calculate bytes per element based on dtype
    if dtype in [torch.float16, torch.bfloat16]:
        bytes_per_element = 2
    elif dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        bytes_per_element = 1
    elif dtype == torch.float32:
        bytes_per_element = 4
    else:
        # Default to 2 bytes for unknown dtypes
        bytes_per_element = 2

    # Memory per expert weight matrix
    memory_per_expert = hidden_size * intermediate_size * bytes_per_element

    # Total memory for active experts
    total_memory_bytes = active_experts * memory_per_expert

    return total_memory_bytes


def get_benchmark_config(use_fp8_w8a8=False):
    if use_fp8_w8a8:
        return {
            "line_vals": [
                "fbgemm_triton_grouped_gemm_fp8",
                "fbgemm_cutlass_f8f8bf16_rowwise",
                "sglang_grouped_gemm",
            ],
            "line_names": [
                "FBGEMM Triton Grouped GEMM FP8",
                "FBGEMM CUTLASS F8F8BF16 Rowwise",
                "SGLang Grouped GEMM FP8",
            ],
            "styles": [("blue", "-"), ("orange", "-"), ("red", "-")],
        }
    else:
        return {
            "line_vals": ["fbgemm_triton_grouped_gemm", "sglang_grouped_gemm"],
            "line_names": [
                "FBGEMM Triton Grouped GEMM BF16",
                "SGLang Grouped GEMM BF16",
            ],
            "styles": [("blue", "-"), ("green", "-")],
        }


def run_benchmark(
    model_config, use_fp8_w8a8=False, save_path="./benchmark_grouped_gemm/"
):
    config = get_benchmark_config(use_fp8_w8a8)

    benchmark_config = triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[256, 512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=config["line_vals"],
        line_names=config["line_names"],
        styles=config["styles"],
        ylabel="Bandwidth (GB/s)",
        plot_name="grouped-gemm-performance",
        args={},
    )

    @triton.testing.perf_report(benchmark_config)
    def dynamic_benchmark(batch_size, provider, model_config, use_fp8_w8a8=False):
        print(f"Benchmarking {provider} with batch_size={batch_size}")
        torch.cuda.manual_seed_all(0)

        num_groups = model_config["num_groups"]
        hidden_size = model_config["hidden_size"]
        intermediate_size = model_config["intermediate_size"]

        if provider == "fbgemm_triton_grouped_gemm_fp8":
            try:
                test_data = create_fp8_test_data(
                    batch_size,
                    num_groups,
                    hidden_size,
                    intermediate_size,
                    backend="triton",
                )
                x_fp8, w_fp8, m_sizes, x_scale, w_scale = test_data

                # Calculate memory bandwidth
                memory_bytes = calculate_memory_bandwidth(
                    m_sizes, hidden_size, intermediate_size, torch.float8_e4m3fn
                )

                def run_func():
                    return fbgemm_grouped_gemm_fp8_rowwise(
                        x_fp8, w_fp8, m_sizes, x_scale, w_scale, use_fast_accum=True
                    )

            except Exception as e:
                print(f"FP8 not supported, skipping: {e}")
                return float("inf"), float("inf"), float("inf")

        elif provider == "fbgemm_cutlass_f8f8bf16_rowwise":
            try:
                test_data = create_fp8_test_data(
                    batch_size,
                    num_groups,
                    hidden_size,
                    intermediate_size,
                    backend="cutlass",
                )
                x, wq, w_scale, m_sizes = test_data

                # Calculate memory bandwidth
                memory_bytes = calculate_memory_bandwidth(
                    m_sizes, hidden_size, intermediate_size, torch.float8_e4m3fn
                )

                # Quantize input using triton_quantize_fp8_row
                xq, x_scale = triton_quantize_fp8_row(x)
                x_scale = x_scale.view(batch_size, -1)

                def run_func():
                    return torch.ops.fbgemm.f8f8bf16_rowwise_grouped_stacked(
                        xq, wq, x_scale, w_scale, m_sizes
                    )

            except Exception as e:
                print(
                    f"CUTLASS f8f8bf16_rowwise_grouped_stacked not supported, "
                    f"skipping: {e}"
                )
                return float("inf"), float("inf"), float("inf")
        else:
            test_data = create_test_data(
                batch_size, num_groups, hidden_size, intermediate_size
            )
            (
                x,
                w_fbgemm,
                w_sglang,
                c_fbgemm,
                c_sglang,
                m_sizes,
                seg_indptr,
                weight_indices,
            ) = test_data

            # Calculate memory bandwidth for BF16 operations
            memory_bytes = calculate_memory_bandwidth(
                m_sizes, hidden_size, intermediate_size, torch.bfloat16
            )

            if provider == "fbgemm_triton_grouped_gemm":

                def run_func():
                    return fbgemm_grouped_gemm(
                        x, w_fbgemm, m_sizes, use_fast_accum=True
                    )

            else:

                def run_func():
                    return sglang_grouped_gemm(
                        x,
                        w_sglang,
                        c_sglang,
                        num_groups,
                        weight_column_major=True,
                        seg_indptr=seg_indptr,
                        weight_indices=weight_indices,
                        c_dtype=c_sglang.dtype,
                    )

        for _ in range(10):
            try:
                run_func()
            except Exception as e:
                print(f"Error during warmup for {provider}: {e}")
                return float("inf"), float("inf"), float("inf")

        torch.cuda.synchronize()

        try:
            quantiles = [0.5, 0.2, 0.8]
            ms, min_ms, max_ms = triton.testing.do_bench(run_func, quantiles=quantiles)

            # Convert time (ms) to bandwidth (GB/s)
            # Bandwidth = Memory (bytes) / Time (seconds)
            # Convert ms to seconds and bytes to GB (1e9)
            gb_per_s = (memory_bytes / 1e9) / (ms / 1000)
            # min bandwidth = max time, max bandwidth = min time
            min_gb_per_s = (memory_bytes / 1e9) / (max_ms / 1000)
            max_gb_per_s = (memory_bytes / 1e9) / (min_ms / 1000)

            return gb_per_s, min_gb_per_s, max_gb_per_s
        except Exception as e:
            print(f"Error during benchmarking for {provider}: {e}")
            return 0.0, 0.0, 0.0

    dynamic_benchmark.run(
        show_plots=True,
        print_data=True,
        save_path=save_path,
        model_config=model_config,
        use_fp8_w8a8=use_fp8_w8a8,
    )


def verify_correctness(model_config):
    print("Verifying correctness...")
    batch_size = 128
    num_groups = model_config["num_groups"]
    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config["intermediate_size"]

    test_data = create_test_data(batch_size, num_groups, hidden_size, intermediate_size)
    (
        x,
        w_fbgemm,
        w_sglang,
        c_fbgemm,
        c_sglang,
        m_sizes,
        seg_indptr,
        weight_indices,
    ) = test_data

    result_fbgemm = fbgemm_grouped_gemm(x, w_fbgemm, m_sizes, use_fast_accum=True)

    result_sglang = sglang_grouped_gemm(
        x,
        w_sglang,
        c_sglang,
        num_groups,
        weight_column_major=True,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        c_dtype=c_sglang.dtype,
    )

    if torch.allclose(result_fbgemm, result_sglang, rtol=1e-3, atol=1e-3):
        print("✓ BF16 Correctness verification passed!")
    else:
        max_diff = torch.max(torch.abs(result_fbgemm - result_sglang))
        print(f"✗ BF16 Correctness verification failed! Max diff: {max_diff}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FBGEMM vs SGLang Grouped GEMM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Model name to get configuration from",
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallelism size"
    )
    parser.add_argument(
        "--use-fp8-w8a8", action="store_true", help="Enable FP8 W8A8 benchmark"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./benchmark_grouped_gemm/",
        help="Path to save benchmark results",
    )
    parser.add_argument(
        "--verify-correctness",
        action="store_true",
        help="Verify correctness before benchmarking",
    )

    args = parser.parse_args()

    try:
        model_config = get_model_config(args.model, args.tp_size)
    except Exception as e:
        print(f"Failed to get model config: {e}")
        print("Using default configuration...")
        model_config = {
            "num_groups": 8,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "dtype": torch.bfloat16,
        }

    print("Running benchmark with:")
    print(f"  num_groups: {model_config['num_groups']}")
    print(f"  hidden_size: {model_config['hidden_size']}")
    print(f"  intermediate_size: {model_config['intermediate_size']}")
    print(f"  use_fp8_w8a8: {args.use_fp8_w8a8}")

    if args.verify_correctness:
        if not verify_correctness(model_config):
            print("Correctness verification failed. Exiting...")
            return

    try:
        run_benchmark(
            model_config=model_config,
            use_fp8_w8a8=args.use_fp8_w8a8,
            save_path=args.save_path,
        )
    except Exception as e:
        print(f"Benchmark failed: {e}")


if __name__ == "__main__":
    main()
