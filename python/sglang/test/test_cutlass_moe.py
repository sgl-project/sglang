import argparse

import torch
import triton  # Added import
import triton.testing  # Added import
from transformers import AutoConfig

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.cutlass import CutlassMoeQuantInfo, CutlassMoEType
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend


# Copy from: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/utils.py
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def get_model_config(tp_size: int):
    config = AutoConfig.from_pretrained(
        "deepseek-ai/Deepseek-R1", trust_remote_code=True
    )
    E = config.n_routed_experts
    topk = config.num_experts_per_tok
    intermediate_size = config.moe_intermediate_size
    shard_intermediate_size = 2 * intermediate_size // tp_size

    return {
        "num_experts": E,
        "topk": topk,
        "hidden_size": config.hidden_size,
        "shard_intermediate_size": shard_intermediate_size,
        "dtype": config.dtype,
        "block_shape": config.quantization_config["weight_block_size"],
    }


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    """Converts tensor to FP8 E4M3, scaling values to fit the range."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    # Calculate max absolute value safely
    max_val = torch.max(torch.abs(tensor))
    # Avoid division by zero if tensor is all zeros
    if max_val == 0:
        scale_factor = 1.0
    else:
        # Scale factor to bring the max value to finfo.max
        scale_factor = finfo.max / max_val

    # Apply scaling
    scaled_tensor = tensor * scale_factor

    # Clamp and convert
    fp8_tensor = scaled_tensor.clamp(min=finfo.min, max=finfo.max).to(
        dtype=torch.float8_e4m3fn
    )
    return fp8_tensor


def run_test(tp_size, batch_size, model_config, check=False):
    print(f"\n--- Batch Size: {batch_size} ---")
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(42)  # For reproducible random numbers

    E = model_config["num_experts"]
    topk = model_config["topk"]
    H = model_config["hidden_size"]
    I = model_config["shard_intermediate_size"]
    block_shape = model_config["block_shape"]  # Tuple (BLOCK_N, BLOCK_K)
    dtype = model_config["dtype"]  # e.g., torch.bfloat16

    print(
        f"Config: E={E}, topk={topk}, H={H}, I_shard={I}, dtype={dtype}, block_shape={block_shape}"
    )

    # --- Input Data ---
    # Use bf16/fp16 for input activation based on model config
    x = torch.randn((batch_size, H), device="cuda", dtype=dtype)
    # --- Weights (Generate in higher precision, then convert to FP8) ---
    # Generate weights suitable for FP8 conversion (e.g., scaled appropriately)
    w1_hp = torch.randn((E, I, H), device="cuda", dtype=torch.float32)
    w2_hp = torch.randn((E, H, I // 2), device="cuda", dtype=torch.float32)

    w1 = to_fp8(w1_hp)
    w2 = to_fp8(w2_hp)

    # --- Scales for FP8 Weights ---
    block_n, block_k = block_shape
    # Calculate number of blocks needed
    w1_blocks_dim1 = (I + block_n - 1) // block_n
    w1_blocks_dim2 = (H + block_k - 1) // block_k
    w2_blocks_dim1 = (H + block_n - 1) // block_n
    w2_blocks_dim2 = (I // 2 + block_k - 1) // block_k

    # Scales are typically float32 or float16/bfloat16
    scale_dtype = torch.float32  # Or dtype if scales match model dtype
    w1_scale = torch.full(
        (E, w1_blocks_dim1, w1_blocks_dim2), 1, device="cuda", dtype=scale_dtype
    )  # Avoid zero scales
    w2_scale = torch.full(
        (E, w2_blocks_dim1, w2_blocks_dim2), 1, device="cuda", dtype=scale_dtype
    )  # Avoid zero scales

    # --- Routing Information ---
    topk_weights = torch.softmax(
        torch.rand(batch_size, topk, device="cuda", dtype=dtype), dim=-1
    )
    topk_ids = torch.randint(0, E, (batch_size, topk), dtype=torch.int32, device="cuda")
    router_logits = torch.randn((batch_size, E), device="cuda", dtype=dtype)

    a1_strides = torch.full((E,), H, dtype=torch.int64, device="cuda")
    c1_strides = torch.full((E,), I, dtype=torch.int64, device="cuda")
    a2_strides = torch.full((E,), I // 2, dtype=torch.int64, device="cuda")
    c2_strides = torch.full((E,), H, dtype=torch.int64, device="cuda")

    workspace = torch.empty(
        (7182 * 1024), device="cuda", dtype=torch.uint8
    )  # Allocate sufficient workspace
    # Pointer arrays (often filled by the kernel or a prep step, but needed as args)
    a_ptrs = torch.empty((E,), dtype=torch.int64, device="cuda")
    b_ptrs = torch.empty((E,), dtype=torch.int64, device="cuda")
    out_ptrs = torch.empty((E,), dtype=torch.int64, device="cuda")
    a_scales_ptrs = torch.empty((E,), dtype=torch.int64, device="cuda")
    b_scales_ptrs = torch.empty((E,), dtype=torch.int64, device="cuda")
    expert_offsets = torch.empty((E + 1,), dtype=torch.int32, device="cuda")
    problem_sizes1 = torch.empty((E, 3), dtype=torch.int32, device="cuda")
    problem_sizes2 = torch.empty((E, 3), dtype=torch.int32, device="cuda")

    # --- Setup for refactored MoeRunner ---
    moe_runner_config = MoeRunnerConfig(
        num_experts=E,
        top_k=topk,
        hidden_size=H,
        intermediate_size_per_partition=I,
        params_dtype=dtype,
        activation="silu",
        inplace=False,
    )

    # Create dispatch output
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
    )
    dispatch_output = StandardDispatchOutput(
        hidden_states=x,
        topk_output=topk_output,
    )

    # CUTLASS runner setup
    cutlass_runner = MoeRunner(MoeRunnerBackend.CUTLASS, moe_runner_config)
    cutlass_quant_info = CutlassMoeQuantInfo(
        moe_type=CutlassMoEType.BlockscaledFP8,
        w13_weight=w1.transpose(1, 2),
        w2_weight=w2.transpose(1, 2),
        w13_scale=w1_scale.transpose(1, 2),
        w2_scale=w2_scale.transpose(1, 2),
        expert_offsets=expert_offsets,
        problem_sizes1=problem_sizes1,
        problem_sizes2=problem_sizes2,
        ab_strides_13=a1_strides,
        ab_strides_2=a2_strides,
        c_strides_13=c1_strides,
        c_strides_2=c2_strides,
        workspace=workspace,
        a_ptrs=a_ptrs,
        b_ptrs=b_ptrs,
        out_ptrs=out_ptrs,
        a_scales_ptrs=a_scales_ptrs,
        b_scales_ptrs=b_scales_ptrs,
    )

    # TRITON runner setup
    # Note: Triton expects non-transposed weights
    triton_runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)
    triton_quant_info = TritonMoeQuantInfo(
        w13_weight=w1,
        w2_weight=w2,
        use_fp8_w8a8=True,
        w13_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )

    # Lambdas using refactored runner paths
    cutlass_lambda = lambda: cutlass_runner.run(
        dispatch_output, cutlass_quant_info
    ).hidden_states

    triton_lambda = lambda: triton_runner.run(
        dispatch_output, triton_quant_info
    ).hidden_states

    # --- Warmup ---
    print("Warming up...")
    for _ in range(10):
        _ = cutlass_lambda()
        _ = triton_lambda()
    torch.cuda.synchronize()

    # --- Benchmarking ---
    quantiles = [0.5, 0.2, 0.8]
    print(f"Benchmarking Cutlass fused_experts...")
    cutlass_ms, cutlass_min, cutlass_max = triton.testing.do_bench_cudagraph(
        cutlass_lambda, rep=1000, quantiles=quantiles
    )

    print(f"Benchmarking Triton fused_experts...")
    triton_ms, triton_min, triton_max = triton.testing.do_bench_cudagraph(
        triton_lambda, rep=1000, quantiles=quantiles
    )
    print(
        f"Cutlass fused_experts time: {cutlass_ms:.3f} ms (median) [{cutlass_min:.3f} - {cutlass_max:.3f}]"
    )
    print(
        f"Triton  fused_experts time: {triton_ms:.3f} ms (median) [{triton_min:.3f} - {triton_max:.3f}]"
    )

    # --- Correctness Check ---
    if check:
        print("Running correctness check...")
        with torch.no_grad():
            # Run CUTLASS version using refactored runner
            y_cutlass = cutlass_runner.run(
                dispatch_output, cutlass_quant_info
            ).hidden_states

            # Run Triton version using refactored runner
            y_triton = triton_runner.run(
                dispatch_output, triton_quant_info
            ).hidden_states

        diff = calc_diff(y_cutlass, y_triton)
        print(f"Diff: {diff:.6f}")

        # Tolerance might need adjustment based on FP8 specifics and kernel differences
        # FP8 comparisons often require higher tolerance than FP16/BF16
        assert diff < 1e-4, f"Diff too high! {diff}"
        print("Correctness check passed.")


def main(tp_size=8, batch_sizes=[1, 4, 8, 16, 32, 64, 128, 256, 512], check=False):
    model_config = get_model_config(tp_size)
    print("Model Config:", model_config)
    for batch_size in batch_sizes:
        run_test(tp_size, batch_size, model_config, check)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=8, help="Tensor Parallel size")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[
            1,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
        ],  # Adjusted default
        help="List of batch sizes to test",
    )
    parser.add_argument("--check", action="store_true", help="Enable check mode")
    args = parser.parse_args()

    print(f"Running benchmarks with TP size: {args.tp_size}")
    print(f"Testing batch sizes: {args.batch_sizes}")

    main(tp_size=args.tp_size, batch_sizes=args.batch_sizes, check=args.check)
