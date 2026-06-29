# python3 benchmark/kernels/fused_moe_triton/sglang_fused_moe_triton.py --model /DeepSeek-V3/ --tp-size 8
import argparse
import contextlib

import torch
import triton
from common_utils import get_model_config

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils import (
    fused_moe as _sglang_fused_moe_mod,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    fused_moe as fused_moe_sglang,
)

# flashinfer (used by sglang.benchmark.bench_utils) is CUDA-only; fall back to
# Triton's do_bench on other accelerators (e.g. XPU).
try:
    from sglang.benchmark.bench_utils import run_bench
except Exception:
    run_bench = None

# The v340 provider depends on the `triton_kernels` package (CUDA/Hopper-Blackwell
# specific). Make it optional so the script still runs where it is unavailable.
try:
    from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
        triton_kernel_moe_forward,
    )

    _HAS_TRITON_KERNELS = True
except Exception:
    triton_kernel_moe_forward = None
    _HAS_TRITON_KERNELS = False
from sglang.srt.layers.moe.topk import (
    TopK,
    TopKConfig,
    TopKOutputFormat,
    select_experts,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def _get_accel_device() -> str:
    """Pick the active accelerator: cuda, else xpu, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


_ACCEL = _get_accel_device()
# torch.cuda / torch.xpu expose the same synchronize / Stream / manual_seed_all API.
_accel_mod = torch.cuda if _ACCEL == "cuda" else getattr(torch, _ACCEL, torch.cuda)
# Graph capture is available on CUDA (CUDAGraph) and on XPU builds that expose
# torch.xpu.XPUGraph / torch.xpu.graph; fall back to eager otherwise.
_supports_graph = _ACCEL == "cuda" or (
    _ACCEL == "xpu" and hasattr(torch.xpu, "XPUGraph")
)


def _accel_synchronize() -> None:
    _accel_mod.synchronize()


def _accel_manual_seed_all(seed: int) -> None:
    _accel_mod.manual_seed_all(seed)


def _accel_new_graph():
    """Return a new capture graph object for the active accelerator."""
    return torch.cuda.CUDAGraph() if _ACCEL == "cuda" else torch.xpu.XPUGraph()


def _accel_graph_ctx(graph):
    """Return the graph-capture context manager for the active accelerator."""
    return torch.cuda.graph(graph) if _ACCEL == "cuda" else torch.xpu.graph(graph)


def _run_bench(fn, quantiles):
    """Device-agnostic timing: flashinfer on CUDA, Triton do_bench elsewhere."""
    if _ACCEL == "cuda" and run_bench is not None:
        return run_bench(fn, quantiles=quantiles)
    return triton.testing.do_bench(fn, quantiles=list(quantiles))


@contextlib.contextmanager
def _force_down_tma(value: bool):
    """Force the resolved down-proj config's USE_TMA flag for benchmarking.

    The production path reads USE_TMA from the tuned down-proj config JSON, so
    the same shapes can only be A/B'd by overriding that flag. This wraps the
    config resolver and stamps the desired USE_TMA onto the down config; the
    kernel still requires hardware tensor-descriptor support to actually use it.

    When no tuned down-proj config exists for the current shape/device,
    ``try_get_optimal_moe_config`` returns ``down_config=None`` (it only fills
    one from a JSON file). Stamping onto ``None`` is a silent no-op, so the
    "tma" provider would run the identical kernel as "no_tma". To make the
    override actually take effect we synthesize a down config from the up
    ``config`` -- it is a complete, valid config and already shares the up
    config's BLOCK_SIZE_M (the constraint both kernels must satisfy, since they
    share one moe_align_block_size sort) -- and stamp USE_TMA onto that.
    """
    orig = _sglang_fused_moe_mod.try_get_optimal_moe_config

    def wrapper(*args, **kwargs):
        out = orig(*args, **kwargs)
        if kwargs.get("return_down_config") and isinstance(out, tuple):
            config, (down_config, max_block_m) = out
            if down_config is None:
                # No tuned down config: synthesize one from the up config so the
                # USE_TMA override below is not a no-op.
                down_config = dict(config)
                if max_block_m is None:
                    max_block_m = config["BLOCK_SIZE_M"]
            down_config["USE_TMA"] = value
            return config, (down_config, max_block_m)
        return out

    _sglang_fused_moe_mod.try_get_optimal_moe_config = wrapper
    try:
        yield
    finally:
        _sglang_fused_moe_mod.try_get_optimal_moe_config = orig


# Providers comparing the sglang fused-MoE path with TMA forced off vs on. The
# v340 provider (separate kernel) is only available where `triton_kernels`
# imported cleanly (CUDA).
_TMA_PROVIDERS = {
    "sglang_fused_moe_triton_no_tma": False,
    "sglang_fused_moe_triton_tma": True,
}
_BATCH_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]
_PROVIDERS = ["sglang_fused_moe_triton_no_tma", "sglang_fused_moe_triton_tma"]
_STYLES = [("green", "-"), ("red", "-")]
if _HAS_TRITON_KERNELS:
    _PROVIDERS.insert(0, "sglang_fused_moe_triton_v340")
    _STYLES.insert(0, ("blue", "-"))


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
        output_format=TopKOutputFormat.TRITON_KERNEL,
    )
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
    # The fused topk_softmax sgl_kernel op is not implemented for float32 on XPU;
    # use the native torch routing path there.
    topk_output = select_experts(
        hidden_states=x,
        router_logits=input_gating,
        topk_config=TopKConfig(
            top_k=topk,
            renormalize=False,
            torch_native=_ACCEL != "cuda",
        ),
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
        x_vals=list(_BATCH_SIZES),
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=_PROVIDERS,
        styles=_STYLES,
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
    torch.set_default_device(_ACCEL)
    _accel_manual_seed_all(0)

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
        force_tma_ctx = contextlib.nullcontext()
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
        force_tma_ctx = _force_down_tma(_TMA_PROVIDERS[provider])

    # Keep USE_TMA forced through warmup, graph capture, and the timed runs so
    # config resolution (which happens on every call) sees the chosen value.
    with force_tma_ctx:
        # Warmup
        for _ in range(10):
            _ = api_func(**api_kwargs)
        _accel_synchronize()

        if use_cuda_graph and _supports_graph:
            graph = _accel_new_graph()
            with _accel_graph_ctx(graph):
                api_func(**api_kwargs)
            _accel_synchronize()

            bench_lambda = lambda: graph.replay()
        else:
            bench_lambda = lambda: api_func(**api_kwargs)

        quantiles = (0.5, 0.2, 0.8)
        ms, min_ms, max_ms = _run_bench(bench_lambda, quantiles=quantiles)
    return ms, min_ms, max_ms


def _prewarm_all_shapes(model_config, use_fp8_w8a8=False):
    """Pre-touch every (batch_size, USE_TMA) combination before the timed sweep.

    The first call for each shape pays one-time costs (Triton JIT compilation,
    config-file cache fill, lazy runtime init) that the timing utility's warmup
    cannot subtract. Running each shape once up front -- for both TMA off and on
    -- moves that cost out of the measured numbers so the first data point is
    not an outlier.
    """
    torch.set_default_device(_ACCEL)
    _accel_manual_seed_all(0)

    num_experts = model_config["num_experts"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    topk = model_config["topk"]
    dtype = model_config["dtype"]
    block_shape = model_config["block_shape"]

    w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, dtype=dtype)
    w2 = torch.randn(
        num_experts, hidden_size, shard_intermediate_size // 2, dtype=dtype
    )

    print("Pre-warming all shapes (TMA off and on)...")
    for batch_size in _BATCH_SIZES:
        x = torch.randn(batch_size, hidden_size, dtype=dtype)
        input_gating = torch.randn(batch_size, num_experts, dtype=torch.float32)
        for use_tma in (False, True):
            with _force_down_tma(use_tma):
                fused_moe_sglang_api(
                    x=x,
                    w1=w1,
                    w2=w2,
                    input_gating=input_gating,
                    topk=topk,
                    use_fp8_w8a8=use_fp8_w8a8,
                    block_shape=block_shape,
                )
    _accel_synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", "--tp", type=int, default=2)
    parser.add_argument("--ep-size", "--ep", type=int, default=1)
    parser.add_argument("--use-fp8-w8a8", action="store_true")
    parser.add_argument(
        "--use-cuda-graph", action="store_true", help="Enable CUDA Graph capture/replay"
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display the matplotlib plot window (blocks until closed). Off by "
        "default so headless/SSH runs do not hang; the plot is still saved to "
        "--save-path.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/benchmark_ops/sglang_fused_moe/",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    # Initialize global server args (required by SGLang MoE kernels)
    server_args = ServerArgs(model_path=args.model)
    set_global_server_args_for_scheduler(server_args)

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
            expert_model_parallel_size=1,
        )

        model_config = get_model_config(args.model, args.tp_size, args.ep_size)
        _prewarm_all_shapes(model_config, use_fp8_w8a8=args.use_fp8_w8a8)
        benchmark.run(
            show_plots=args.show_plots,
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
