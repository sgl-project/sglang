# python3 benchmark/kernels/fbgemm/benchmark_fbgemm_grouped_gemm.py --model Qwen/Qwen2-57B-A14B-Instruct --tp-size 4 --use-fp8-w8a8
import argparse

import torch
import triton
from fbgemm_grouped_gemm import grouped_gemm as fbgemm_grouped_gemm
from fbgemm_grouped_gemm import (
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
    elif config.architectures[0] in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]:
        num_groups = (
            config.n_routed_experts + 1
            if config.architectures[0] in ["DeepseekV3ForCausalLM"]
            else config.n_routed_experts
        )
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
        (num_groups,), tokens_per_group, dtype=torch.int64, device="cuda"
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

    seg_indptr = torch.zeros(num_groups + 1, dtype=torch.int64, device="cuda")
    for i in range(1, num_groups + 1):
        seg_indptr[i] = seg_indptr[i - 1] + tokens_per_group

    weight_indices = torch.arange(num_groups, dtype=torch.int64, device="cuda")

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


def create_fp8_test_data(batch_size, num_groups, hidden_size, intermediate_size):
    torch.manual_seed(42)

    tokens_per_group = batch_size // num_groups
    m_sizes = torch.full(
        (num_groups,), tokens_per_group, dtype=torch.int64, device="cuda"
    )

    x_fp16 = torch.randn(batch_size, hidden_size, dtype=torch.float16, device="cuda")
    w_fp16 = torch.randn(
        num_groups * intermediate_size, hidden_size, dtype=torch.float16, device="cuda"
    )

    x_fp8 = x_fp16.to(torch.float8_e4m3fn)
    w_fp8 = w_fp16.to(torch.float8_e4m3fn)

    x_scale = torch.randn(batch_size, dtype=torch.float32, device="cuda").abs() + 1e-4
    w_scale = torch.randn(num_groups, dtype=torch.float32, device="cuda").abs() + 1e-4

    return x_fp8, w_fp8, m_sizes, x_scale, w_scale


def get_benchmark_config(use_fp8_w8a8=False):
    if use_fp8_w8a8:
        return {
            "line_vals": ["fbgemm_grouped_gemm_fp8", "sglang_grouped_gemm"],
            "line_names": ["FBGEMM Grouped GEMM FP8", "SGLang Grouped GEMM FP8"],
            "styles": [("blue", "-"), ("red", "-")],
        }
    else:
        return {
            "line_vals": ["fbgemm_grouped_gemm", "sglang_grouped_gemm"],
            "line_names": ["FBGEMM Grouped GEMM BF16", "SGLang Grouped GEMM BF16"],
            "styles": [("blue", "-"), ("green", "-")],
        }


def run_benchmark(
    model_config, use_fp8_w8a8=False, save_path="./benchmark_grouped_gemm/"
):
    config = get_benchmark_config(use_fp8_w8a8)

    benchmark_config = triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=config["line_vals"],
        line_names=config["line_names"],
        styles=config["styles"],
        ylabel="Time (ms)",
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

        if provider == "fbgemm_grouped_gemm_fp8":
            try:
                test_data = create_fp8_test_data(
                    batch_size, num_groups, hidden_size, intermediate_size
                )
                x_fp8, w_fp8, m_sizes, x_scale, w_scale = test_data

                def run_func():
                    return fbgemm_grouped_gemm_fp8_rowwise(
                        x_fp8, w_fp8, m_sizes, x_scale, w_scale, use_fast_accum=True
                    )

            except Exception as e:
                print(f"FP8 not supported, skipping: {e}")
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

            if provider == "fbgemm_grouped_gemm":

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
            return ms, min_ms, max_ms
        except Exception as e:
            print(f"Error during benchmarking for {provider}: {e}")
            return float("inf"), float("inf"), float("inf")

    dynamic_benchmark.run(
        show_plots=True,
        print_data=True,
        save_path=save_path,
        model_config=model_config,
        use_fp8_w8a8=use_fp8_w8a8,
    )


def verify_correctness(model_config, use_fp8_w8a8):
    print("Verifying correctness...")
    batch_size = 128
    num_groups = model_config["num_groups"]
    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config["intermediate_size"]

    test_data = create_test_data(batch_size, num_groups, hidden_size, intermediate_size)
    (x, w_fbgemm, w_sglang, c_fbgemm, c_sglang, m_sizes, seg_indptr, weight_indices) = (
        test_data
    )

    try:
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

        if use_fp8_w8a8:
            try:
                fp8_data = create_fp8_test_data(
                    batch_size, num_groups, hidden_size, intermediate_size
                )
                x_fp8, w_fp8, m_sizes_fp8, x_scale, w_scale = fp8_data

                result_fp8 = fbgemm_grouped_gemm_fp8_rowwise(
                    x_fp8, w_fp8, m_sizes_fp8, x_scale, w_scale, use_fast_accum=True
                )

                assert result_fp8.shape == (batch_size, intermediate_size)
                print("✓ FP8 functionality test passed!")
            except Exception as e:
                print(f"FP8 test failed (possibly unsupported): {e}")
                return False

        return True

    except Exception as e:
        print(f"✗ Error during correctness verification: {e}")
        return False


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
        if not verify_correctness(model_config, args.use_fp8_w8a8):
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
