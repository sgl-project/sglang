# python3 benchmark/kernels/fused_moe_triton/sglang_fused_moe_triton.py --model /DeepSeek-V3/ --tp-size 8
import argparse

import torch
import triton
from transformers import AutoConfig

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_moe as fused_moe_sglang,
)
from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
    triton_kernel_moe_forward,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.topk import TopK, TopKConfig, select_experts


def get_model_config(model_name: str, tp_size: int):
    """Get model configuration parameters"""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    if config.architectures[0] == "Qwen2MoeForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // tp_size
    elif config.architectures[0] == "Qwen3MoeForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // tp_size
    elif config.architectures[0] in [
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "Glm4MoeForCausalLM",
    ]:
        E = (
            config.n_routed_experts + 1
            if config.architectures[0] in ["DeepseekV3ForCausalLM"]
            else config.n_routed_experts
        )
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // tp_size
    else:
        # Default: Mixtral
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // tp_size

    block_shape = None
    if (
        hasattr(config, "quantization_config")
        and "weight_block_size" in config.quantization_config
    ):
        block_shape = config.quantization_config["weight_block_size"]
        assert len(block_shape) == 2

    shape_configs = {
        "num_experts": E,
        "topk": topk,
        "hidden_size": config.hidden_size,
        "shard_intermediate_size": shard_intermediate_size,
        "dtype": config.torch_dtype,
        "block_shape": block_shape,
    }
    print(f"{shape_configs=}")
    return shape_configs


def fused_moe_triton_api(
    x,
    w1,
    w2,
    input_gating,
    topk,
):
    topk_op = TopK(
        top_k=topk,
        renormalize=False,
        use_grouped_topk=False,
    )
    topk_op.use_triton_kernels = True
    triton_topk_output = topk_op.forward_cuda(
        hidden_states=x,
        router_logits=input_gating,
    )

    moe_runner_config = MoeRunnerConfig(
        inplace=False,
    )
    return triton_kernel_moe_forward(
        x,
        w1,
        w2,
        triton_topk_output,
        moe_runner_config,
    )


def fused_moe_sglang_api(
    x,
    w1,
    w2,
    input_gating,
    topk,
    use_fp8_w8a8=False,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
    block_shape=None,
):
    topk_output = select_experts(
        hidden_states=x,
        router_logits=input_gating,
        topk_config=TopKConfig(top_k=topk, renormalize=False),
    )
    return fused_moe_sglang(
        x,
        w1,
        w2,
        topk_output,
        use_fp8_w8a8=use_fp8_w8a8,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=list([128, 256, 512, 1024, 2048, 4096, 8192]),
        line_arg="provider",
        line_vals=[
            "sglang_fused_moe_triton_v340",
            "sglang_fused_moe_triton",
        ],
        line_names=[
            "sglang_fused_moe_triton_v340",
            "sglang_fused_moe_triton",
        ],
        styles=[
            ("blue", "-"),
            ("green", "-"),
        ],
        ylabel="Time (ms)",
        plot_name="fused-moe-performance",
        args={},
    )
)
def benchmark(
    batch_size,
    provider,
    model_config,
    use_fp8_w8a8=False,
    use_cuda_graph: bool = False,
):
    print(f"benchmark {provider} with batch_size={batch_size}")
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)

    num_tokens = batch_size
    num_experts = model_config["num_experts"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    topk = model_config["topk"]
    dtype = model_config["dtype"]
    block_shape = model_config["block_shape"]

    x = torch.randn(num_tokens, hidden_size, dtype=dtype)

    w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, dtype=dtype)
    w2 = torch.randn(
        num_experts, hidden_size, shard_intermediate_size // 2, dtype=dtype
    )

    w1_tri = w1.clone()
    w2_tri = w2.clone()
    w1_tri = w1_tri.transpose(-2, -1).contiguous()
    w2_tri = w2_tri.transpose(-2, -1).contiguous()

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)

    if provider == "sglang_fused_moe_triton_v340":
        api_func = fused_moe_triton_api
        api_kwargs = {
            "x": x,
            "w1": w1_tri,
            "w2": w2_tri,
            "input_gating": input_gating,
            "topk": topk,
        }
    else:
        api_func = fused_moe_sglang_api
        api_kwargs = {
            "x": x,
            "w1": w1,
            "w2": w2,
            "input_gating": input_gating,
            "topk": topk,
            "use_fp8_w8a8": use_fp8_w8a8,
            "block_shape": block_shape,
        }

    # Warmup
    for _ in range(10):
        _ = api_func(**api_kwargs)
    torch.cuda.synchronize()

    if use_cuda_graph:
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            api_func(**api_kwargs)
        torch.cuda.synchronize()

        bench_lambda = lambda: graph.replay()
    else:
        bench_lambda = lambda: api_func(**api_kwargs)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(bench_lambda, quantiles=quantiles)
    return ms, min_ms, max_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--use-fp8-w8a8", action="store_true")
    parser.add_argument(
        "--use-cuda-graph", action="store_true", help="Enable CUDA Graph capture/replay"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/benchmark_ops/sglang_fused_moe/",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    try:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="tcp://127.0.0.1:23456",
                world_size=1,
                rank=0,
            )

        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:23456",
            local_rank=0,
            backend="nccl" if torch.cuda.is_available() else "gloo",
        )

        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        model_config = get_model_config(args.model, args.tp_size)
        benchmark.run(
            show_plots=True,
            print_data=True,
            save_path=args.save_path,
            model_config=model_config,
            use_fp8_w8a8=args.use_fp8_w8a8,
            use_cuda_graph=args.use_cuda_graph,
        )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
