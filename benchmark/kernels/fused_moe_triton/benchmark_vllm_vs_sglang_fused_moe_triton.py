# python3 benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py --model /DeepSeek-V3/ --tp-size 8 --use-fp8-w8a8
import argparse

import torch
import triton
import vllm
from transformers import AutoConfig
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe as fused_moe_vllm

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_moe as fused_moe_sglang,
)


def get_model_config(model_name: str, tp_size: int):
    """Get model configuration parameters"""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        shard_intermediate_size = 2 * intermediate_size // tp_size
    elif config.architectures[0] == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // tp_size
    elif config.architectures[0] == "Qwen2MoeForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // tp_size
    elif config.architectures[0] == "Qwen3MoeForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // tp_size
    elif config.architectures[0] in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]:
        E = (
            config.n_routed_experts + 1
            if config.architectures[0] in ["DeepseekV3ForCausalLM"]
            else config.n_routed_experts
        )
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // tp_size
    elif config.architectures[0] == "Llama4ForConditionalGeneration":
        E = config.text_config.num_local_experts
        topk = config.text_config.num_experts_per_tok
        intermediate_size = config.text_config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // tp_size
    elif config.architectures[0] in [
        "Grok1ForCausalLM",
        "Grok1ImgGen",
        "Grok1AForCausalLM",
    ]:
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // tp_size
    else:
        # Default: Mixtral
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // tp_size

    vllm_version_num = (
        vllm.__version_tuple__[0] * 100
        + vllm.__version_tuple__[1] * 10
        + vllm.__version_tuple__[2]
    )
    block_shape = None
    if (
        hasattr(config, "quantization_config")
        and "weight_block_size" in config.quantization_config
    ):
        block_shape = config.quantization_config["weight_block_size"]
        assert len(block_shape) == 2
        assert (
            vllm_version_num >= 66
        ), "Block-wise quantized fp8 fused_moe is only supported for VLLM>=0.6.6.post1"

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


def fused_moe_vllm_api(
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
    if block_shape is not None:
        return fused_moe_vllm(
            x,
            w1,
            w2,
            input_gating,
            topk,
            renormalize=True,
            inplace=True,
            use_fp8_w8a8=use_fp8_w8a8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
        )
    else:
        return fused_moe_vllm(
            x,
            w1,
            w2,
            input_gating,
            topk,
            renormalize=True,
            inplace=True,
            use_fp8_w8a8=use_fp8_w8a8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
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
    return fused_moe_sglang(
        x,
        w1,
        w2,
        input_gating,
        topk,
        renormalize=True,
        inplace=True,
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
        x_vals=list(range(1, 513)),
        line_arg="provider",
        line_vals=[
            "vllm_fused_moe_triton",
            "sglang_fused_moe_triton",
        ],
        line_names=[
            "vllm_fused_moe_triton",
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
def benchmark(batch_size, provider, model_config, use_fp8_w8a8=False):
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
    w1_scale = w2_scale = a1_scale = a2_scale = None

    if use_fp8_w8a8:
        init_dtype = dtype
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype
        )
        w1 = w1.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)

        if block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)
            a1_scale = torch.randn(1, dtype=torch.float32)
            a2_scale = torch.randn(1, dtype=torch.float32)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand(
                (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32
            )
            w2_scale = torch.rand(
                (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32
            )
    else:
        w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, dtype=dtype)
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=dtype
        )

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)

    # Warmup
    api_func = (
        fused_moe_vllm_api
        if provider == "vllm_fused_moe_triton"
        else fused_moe_sglang_api
    )
    for _ in range(10):
        y = api_func(
            x,
            w1,
            w2,
            input_gating,
            topk,
            use_fp8_w8a8=use_fp8_w8a8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
        )
    torch.cuda.synchronize()

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: api_func(
            x,
            w1,
            w2,
            input_gating,
            topk,
            use_fp8_w8a8=use_fp8_w8a8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
        )[0],
        quantiles=quantiles,
    )
    return ms, min_ms, max_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--use-fp8-w8a8", action="store_true")
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/benchmark_ops/vllm_sglang_fused_moe/",
    )
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
        )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
