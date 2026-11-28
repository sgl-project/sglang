import itertools
from unittest import mock

import torch
import torch.nn.functional as F
import triton
import triton.testing

from sglang.kernels.jit.benchmark.utils import (
    get_benchmark_range,
    run_benchmark,
    run_benchmark_no_cudagraph,
)
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptNvFp4EmbeddingMethod,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

GROUP_SIZE = 16
VOCAB_SIZE = 131072

NUM_TOKENS = get_benchmark_range([1, 8, 32, 256, 2048, 8192], ci_range=[1, 256])
# 160 is the Qwen3.8-Flash-Next PLE width; the others are token-embedding widths.
HIDDEN_SIZES = get_benchmark_range([160, 2560, 7168], ci_range=[160])
CUDA_GRAPH = get_benchmark_range([True, False], ci_range=[True])

BENCHMARK_CONFIGS = list(itertools.product(NUM_TOKENS, HIDDEN_SIZES, CUDA_GRAPH))


def _make_nvfp4_layer(hidden_size: int):
    method = ModelOptNvFp4EmbeddingMethod(
        ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=GROUP_SIZE)
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=hidden_size,
        output_partition_sizes=[VOCAB_SIZE],
        input_size=hidden_size,
        output_size=VOCAB_SIZE,
        params_dtype=torch.bfloat16,
    )
    layer = layer.cuda()
    layer.weight.random_(0, 256)
    layer.weight_scale.copy_(
        torch.rand(layer.weight_scale.shape, device="cuda").to(torch.float8_e4m3fn)
    )
    layer.weight_scale_2.fill_(0.375)
    method.process_weights_after_loading(layer)
    assert layer.use_fused_nvfp4_embedding, "This GPU cannot run the fused kernel."
    return method, layer


def _torch_nvfp4_embedding(method, layer, ids):
    with mock.patch.object(
        modelopt_quant, "_can_use_fused_nvfp4_embedding", return_value=False
    ):
        return method.embedding(layer, ids)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "hidden_size", "cuda_graph"],
        x_vals=BENCHMARK_CONFIGS,
        line_arg="provider",
        line_vals=["fused", "torch", "bf16"],
        line_names=[
            "Fused Triton NVFP4",
            "Torch NVFP4 (previous path)",
            "Resident BF16 F.embedding",
        ],
        styles=[("blue", "-"), ("red", "--"), ("green", ":")],
        ylabel="us",
        plot_name="nvfp4-embedding-performance",
        args={},
    )
)
@torch.no_grad()
def benchmark(num_tokens: int, hidden_size: int, cuda_graph: bool, provider: str):
    method, layer = _make_nvfp4_layer(hidden_size)
    ids = torch.randint(0, VOCAB_SIZE, (num_tokens,), device="cuda")

    fused = method.embedding(layer, ids)
    torch.testing.assert_close(
        fused, _torch_nvfp4_embedding(method, layer, ids), rtol=0, atol=0
    )

    if provider == "fused":
        fn = lambda: method.embedding(layer, ids)
    elif provider == "torch":
        fn = lambda: _torch_nvfp4_embedding(method, layer, ids)
    elif provider == "bf16":
        dense = method.embedding(layer, torch.arange(VOCAB_SIZE, device="cuda"))
        fn = lambda: F.embedding(ids, dense)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if cuda_graph:
        return run_benchmark(fn)
    return run_benchmark_no_cudagraph(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
