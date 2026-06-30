import argparse
from typing import Tuple

import torch
import triton  # Added import
import triton.testing  # Added import
from transformers import AutoConfig

from sglang.jit_kernel.mxfp8 import es_sm100_mxfp8_blockscaled_grouped_quant
from sglang.srt.layers.moe.cutlass_moe import cutlass_fused_experts_fp8
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import StandardTopKOutput


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


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def per_block_cast_to_fp8(
    x: torch.Tensor, block_m: int, block_n: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, block_m) * block_m, ceil_div(n, block_n) * block_n),
        dtype=x.dtype,
        device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, block_m, x_padded.size(1) // block_n, block_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


def to_fp8(tensor: torch.Tensor, block_m: int, block_n: int) -> torch.Tensor:
    E = tensor.shape[0]
    q_list = []
    sf_list = []
    for e in range(E):
        q, sf = per_block_cast_to_fp8(tensor[e, :, :], block_m, block_n)
        q_list.append(q)
        sf_list.append(sf)
    return torch.stack(q_list, dim=0), torch.stack(sf_list, dim=0)


def run_test(tp_size, batch_size, model_config, check=False, use_mxfp8=False):
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

    w1, w1_scale = to_fp8(w1_hp, *block_shape)
    w2, w2_scale = to_fp8(w2_hp, *block_shape)

    if use_mxfp8:
        assert H % 128 == 0, "H should align to 128 when use_mxfp8 == True"
        assert (
            I // 2
        ) % 128 == 0, "(I // 2) should align to 128 when use_mxfp8 == True"
        _w1_quant_ng = torch.ones((E,), dtype=torch.int32, device=w1_hp.device) * I
        _w1_quant_offset = torch.concat(
            [torch.tensor([0], dtype=torch.int32), torch.cumsum(_w1_quant_ng, 0)]
        )[:-1]
        mxfp8_w1 = torch.empty_like(w1).view(-1, H)
        mxfp8_w1_scale = torch.full(
            (E * I, H // 32), 0, device=w1_hp.device, dtype=torch.uint8
        )
        es_sm100_mxfp8_blockscaled_grouped_quant(
            w1_hp.to(dtype).view(-1, H),
            _w1_quant_ng,
            _w1_quant_offset.to(torch.int32),
            _w1_quant_offset.to(torch.int32),
            mxfp8_w1,
            mxfp8_w1_scale,
        )
        mxfp8_w1 = mxfp8_w1.view_as(w1)
        mxfp8_w1_scale = mxfp8_w1_scale.view(E, I, H // 32)

        _w2_quant_ng = torch.ones((E,), dtype=torch.int32, device=w1_hp.device) * H
        _w2_quant_offset = torch.concat(
            [torch.tensor([0], dtype=torch.int32), torch.cumsum(_w2_quant_ng, 0)]
        )[:-1]
        mxfp8_w2 = torch.empty_like(w2).view(-1, I // 2)
        mxfp8_w2_scale = torch.full(
            (E * H, I // 64), 0, device=w2_hp.device, dtype=torch.uint8
        )
        es_sm100_mxfp8_blockscaled_grouped_quant(
            w2_hp.to(dtype).view(-1, I // 2),
            _w2_quant_ng,
            _w2_quant_offset.to(torch.int32),
            _w2_quant_offset.to(torch.int32),
            mxfp8_w2,
            mxfp8_w2_scale,
        )
        mxfp8_w2 = mxfp8_w2.view_as(w2)
        mxfp8_w2_scale = mxfp8_w2_scale.view(E, H, I // 64)

    # --- Routing Information ---
    topk_weights = torch.softmax(
        torch.rand(batch_size, topk, device="cuda", dtype=dtype), dim=-1
    )
    topk_ids = torch.randint(0, E, (batch_size, topk), dtype=torch.int32, device="cuda")

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

    enable_es = (False, False)
    if torch.cuda.get_device_name(torch.cuda.current_device()) == "NVIDIA H200":
        enable_es = (False, True)
    elif torch.cuda.get_device_name(torch.cuda.current_device()) == "NVIDIA H20":
        enable_es = (True, True)
    elif use_mxfp8:
        enable_es = (True, True)

    # --- Lambdas for Benchmarking ---
    cutlass_lambda = lambda: cutlass_fused_experts_fp8(
        x,
        mxfp8_w1.transpose(1, 2) if use_mxfp8 else w1.transpose(1, 2),  # Transposed
        mxfp8_w2.transpose(1, 2) if use_mxfp8 else w2.transpose(1, 2),  # Transposed
        mxfp8_w1_scale.transpose(1, 2) if use_mxfp8 else w1_scale.transpose(1, 2),
        mxfp8_w2_scale.transpose(1, 2) if use_mxfp8 else w2_scale.transpose(1, 2),
        topk_weights,
        topk_ids,
        a1_strides,
        c1_strides,
        a2_strides,
        c2_strides,
        workspace,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        enable_es=enable_es,
        use_mxfp8=use_mxfp8,
    )

    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=torch.randn(
            (batch_size, topk), device=topk_weights.device, dtype=dtype
        ),
    )

    moe_runner_config = MoeRunnerConfig(
        num_experts=E,
        top_k=topk,
        hidden_size=H,
        intermediate_size_per_partition=I,
        params_dtype=dtype,
        activation="silu",
        inplace=False,
    )

    # Note: Triton expects non-transposed weights
    triton_lambda = lambda: fused_experts(
        x,
        w1,
        w2,
        topk_output,
        moe_runner_config,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )

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
            # Run CUTLASS version (requires transposed weights)
            y_cutlass = cutlass_fused_experts_fp8(
                x,
                (
                    mxfp8_w1.transpose(1, 2) if use_mxfp8 else w1.transpose(1, 2)
                ),  # Transposed
                (
                    mxfp8_w2.transpose(1, 2) if use_mxfp8 else w2.transpose(1, 2)
                ),  # Transposed
                (
                    mxfp8_w1_scale.transpose(1, 2)
                    if use_mxfp8
                    else w1_scale.transpose(1, 2)
                ),
                (
                    mxfp8_w2_scale.transpose(1, 2)
                    if use_mxfp8
                    else w2_scale.transpose(1, 2)
                ),
                topk_weights,
                topk_ids,
                a1_strides,
                c1_strides,
                a2_strides,
                c2_strides,
                workspace,
                a_ptrs,
                b_ptrs,
                out_ptrs,
                a_scales_ptrs,
                b_scales_ptrs,
                expert_offsets,
                problem_sizes1,
                problem_sizes2,
                enable_es=enable_es,
                use_mxfp8=use_mxfp8,
            )

            # Run Triton version (requires original shape weights, use inplace=False)
            y_triton = fused_experts(
                x,
                w1_hp.to(dtype),  # Original shape
                w2_hp.to(dtype),  # Original shape
                topk_output,
                moe_runner_config,
            )

        diff = calc_diff(y_cutlass, y_triton)
        print(f"Diff: {diff:.6f}")
        # Tolerance might need adjustment based on FP8 specifics and kernel differences
        # FP8 comparisons often require higher tolerance than FP16/BF16
        assert diff < 0.001, f"Diff too high! {diff}"
        print("Correctness check passed.")


def main(
    tp_size=8,
    batch_sizes=[1, 4, 8, 16, 32, 64, 128, 256, 512],
    check=False,
    use_mxfp8=False,
):
    model_config = get_model_config(tp_size)
    print("Model Config:", model_config)
    for batch_size in batch_sizes:
        run_test(tp_size, batch_size, model_config, check, use_mxfp8=use_mxfp8)


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
    parser.add_argument("--mxfp8", action="store_true", help="Enable MXFP8")
    args = parser.parse_args()

    print(f"Running benchmarks with TP size: {args.tp_size}")
    print(f"Testing batch sizes: {args.batch_sizes}")

    main(
        tp_size=args.tp_size,
        batch_sizes=args.batch_sizes,
        check=args.check,
        use_mxfp8=args.mxfp8,
    )
