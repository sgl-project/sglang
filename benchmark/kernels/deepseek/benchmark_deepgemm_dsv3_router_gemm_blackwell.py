import argparse

import torch 
from flashinfer.gemm.routergemm_dsv3 import mm_M1_16_K7168_N256 

from sgl_kernel import dsv3_router_gemm

def dsv3_router_gemm_flashinfer(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    launch_with_pdl=False
):
    """Flashinfer implementation of dsv3 router gemm"""
    num_tokens, num_experts = hidden_states.shape[0], router_weights.shape[0]
    # output = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.bfloat16)
    output = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32).contiguous()

    print(f"hidden_states.shape: {hidden_states.shape}")    
    print(f"num_tokens: {num_tokens}, num_experts: {num_experts}")
    print(f"router_weights.shape: {router_weights.shape}")    
    print(f"output.shape: {output.shape}")    

    mm_M1_16_K7168_N256(
        hidden_states,
        router_weights.t(),
        output,
        launch_with_pdl=launch_with_pdl
    )
    return output


def dsv3_router_gemm_sgl(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
):
    """SGLang implementation of dsv3 router gemm"""
    output = dsv3_router_gemm(
        hidden_states,
        router_weights,
        out_dtype=torch.float32,
    )
    return output


def check_accuracy(a, b, atol, rtol, percent):
    """Unified accuracy checking function with detailed error reporting."""
    if not torch.isfinite(a).all():
        print("Non-finite values in reference output")
        return False
    if not torch.isfinite(b).all():
        print("Non-finite values in actual output")
        return False
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    close = torch.isclose(a, b, atol=atol, rtol=rtol)
    match_ratio = close.float().mean()
    if match_ratio >= percent:
        return True

    mismatch_percent = 1.0 - match_ratio.item()
    if mismatch_percent > 1 - percent:
        print(
            f"Mismatch percentage is {mismatch_percent:.4f} for rtol {rtol} "
            f"(threshold: {1 - percent:.4f})"
        )
        return False

def calculate_diff(num_tokens: int, num_experts: int, hidden_dim: int):
    hidden_states = torch.randn((num_tokens, hidden_dim), device="cuda", dtype=torch.bfloat16).contiguous()
    router_weights = torch.randn((num_experts, hidden_dim), device="cuda", dtype=torch.bfloat16).contiguous()

    out_flashinfer = dsv3_router_gemm_flashinfer(
        hidden_states,
        router_weights,
        False,
    )

    out_sgl = dsv3_router_gemm_sgl(
        hidden_states,
        router_weights,
    )

    print(f"Shape m={num_tokens}, n={num_experts}, k={hidden_dim}:")
    print(f"Flashinfer output: {out_flashinfer[0, 0:5]}")
    print(f"DeepGEMM output: {out_sgl[0, 0:5]}")

    flashinfer_deepgemm_match = check_accuracy(
        out_flashinfer, out_sgl, 0.1, 0.6, 0.95
    )
    print("Correctness check:")
    print(f"  - Flashinfer vs DeepGEMM: {'✅' if flashinfer_deepgemm_match else '❌'}")


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        print("Skipping benchmark because the device is not supported")
        exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/benchmark_ops/dsv3_router_gemm/",
        help="Path to save dsv3 router gemm benchmark results",
    )
    parser.add_argument(
        "--run-correctness",
        action="store_true",
        default=True,
        help="Whether to run correctness test",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size to benchmark (default: 1)",
    )
    parser.add_argument(
        "--plot-friendly",
        action="store_true",
        default=False,
        help="Plot x axis as the config index instead of the m",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Run correctness tests on a few examples
    if args.run_correctness:
        print("Running correctness tests...")
        calculate_diff(1, 256, 7168)  # Small test
        calculate_diff(8, 256, 7168)  # Medium test
        calculate_diff(16, 256, 7168)  # Large test

