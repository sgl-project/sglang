import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Add the parent directory to the path to import block_sparse_attn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vsa import block_sparse_attn

from test.utils import (
    create_full_mask_from_block_mask,
    generate_block_sparse_mask_for_function,
)

BLOCK_M = 64
BLOCK_N = 64


def pytorch_test(Q, K, V, block_sparse_mask, dO):
    q_ = Q.clone().float().requires_grad_()
    k_ = K.clone().float().requires_grad_()
    v_ = V.clone().float().requires_grad_()

    QK = torch.matmul(q_, k_.transpose(-2, -1))
    QK /= q_.size(-1) ** 0.5
    QK = QK.masked_fill(~block_sparse_mask.unsqueeze(0), float("-inf"))

    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v_)

    dO_ = dO
    output.backward(dO_)
    return (
        output.to(torch.bfloat16),
        q_.grad.to(torch.bfloat16),
        k_.grad.to(torch.bfloat16),
        v_.grad.to(torch.bfloat16),
    )


def block_sparse_kernel_test(
    Q, K, V, block_sparse_mask, variable_block_sizes, non_pad_index, dO
):
    Q = Q.detach().requires_grad_()
    K = K.detach().requires_grad_()
    V = V.detach().requires_grad_()

    q_padded = vsa_pad(Q, non_pad_index, variable_block_sizes.shape[0], BLOCK_M)
    k_padded = vsa_pad(K, non_pad_index, variable_block_sizes.shape[0], BLOCK_M)
    v_padded = vsa_pad(V, non_pad_index, variable_block_sizes.shape[0], BLOCK_M)
    output, _ = block_sparse_attn(
        q_padded, k_padded, v_padded, block_sparse_mask, variable_block_sizes
    )
    output = output[:, :, non_pad_index, :]
    output.backward(dO)
    return output, Q.grad, K.grad, V.grad


def get_non_pad_index(
    vid_len: torch.LongTensor,
    n_win: int,
    win_size: int,
):
    device = vid_len.device
    starts_pad = torch.arange(n_win, device=device) * win_size
    index_pad = starts_pad[:, None] + torch.arange(win_size, device=device)[None, :]
    index_mask = torch.arange(win_size, device=device)[None, :] < vid_len[:, None]

    return index_pad[index_mask]


def generate_tensor(shape, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)
    return tensor


def generate_variable_block_sizes(num_blocks, min_size=32, max_size=64, device="cuda"):
    return torch.randint(
        min_size, max_size + 1, (num_blocks,), device=device, dtype=torch.int32
    )


def vsa_pad(x, non_pad_index, num_blocks, block_size):
    padded_x = torch.zeros(
        (1, x.shape[1], num_blocks * BLOCK_M, x.shape[3]),
        device=x.device,
        dtype=x.dtype,
    )
    padded_x[:, :, non_pad_index, :] = x
    return padded_x


def check_correctness(h, d, num_blocks, k, num_iterations=20, error_mode="all"):
    results = {
        "gO": {"sum_diff": 0.0, "sum_abs": 0.0, "max_diff": 0.0},
        "gQ": {"sum_diff": 0.0, "sum_abs": 0.0, "max_diff": 0.0},
        "gK": {"sum_diff": 0.0, "sum_abs": 0.0, "max_diff": 0.0},
        "gV": {"sum_diff": 0.0, "sum_abs": 0.0, "max_diff": 0.0},
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    variable_block_sizes = generate_variable_block_sizes(num_blocks, device=device)
    S = int(variable_block_sizes.sum().item())
    padded_S = num_blocks * BLOCK_M
    non_pad_index = get_non_pad_index(variable_block_sizes, num_blocks, BLOCK_M)
    block_mask = generate_block_sparse_mask_for_function(h, num_blocks, k, device)
    full_mask = create_full_mask_from_block_mask(
        block_mask, variable_block_sizes, device
    )
    for _ in range(num_iterations):
        Q = generate_tensor((1, h, S, d), torch.bfloat16, device)
        K = generate_tensor((1, h, S, d), torch.bfloat16, device)
        V = generate_tensor((1, h, S, d), torch.bfloat16, device)
        dO = generate_tensor((1, h, S, d), torch.bfloat16, device)

        # dO_padded = torch.zeros_like(dO_padded)
        # dO_padded[:, :, non_pad_index, :] = dO

        pt_o, pt_qg, pt_kg, pt_vg = pytorch_test(Q, K, V, full_mask, dO)
        bs_o, bs_qg, bs_kg, bs_vg = block_sparse_kernel_test(
            Q, K, V, block_mask.unsqueeze(0), variable_block_sizes, non_pad_index, dO
        )
        for name, (pt, bs) in zip(
            ["gQ", "gK", "gV", "gO"],
            [(pt_qg, bs_qg), (pt_kg, bs_kg), (pt_vg, bs_vg), (pt_o, bs_o)],
        ):
            if bs is not None:
                diff = pt - bs
                abs_diff = torch.abs(diff)
                results[name]["sum_diff"] += torch.sum(abs_diff).item()
                results[name]["sum_abs"] += torch.sum(torch.abs(pt)).item()
                rel_max_diff = torch.max(abs_diff) / torch.mean(torch.abs(pt))
                results[name]["max_diff"] = max(
                    results[name]["max_diff"], rel_max_diff.item()
                )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elements = h * S * d * num_iterations
    for name, data in results.items():
        avg_diff = data["sum_diff"] / total_elements
        max_diff = data["max_diff"]
        results[name] = {"avg_diff": avg_diff, "max_diff": max_diff}

    return results


def generate_error_graphs(h, d, error_mode="all"):
    test_configs = [
        {"num_blocks": 16, "k": 2, "description": "Small sequence"},
        {"num_blocks": 32, "k": 4, "description": "Medium sequence"},
        {"num_blocks": 53, "k": 6, "description": "Large sequence"},
    ]

    print(f"\nError Analysis for h={h}, d={d}, mode={error_mode}")
    print("=" * 150)
    print(
        f"{'Config':<20} {'Blocks':<8} {'K':<4} "
        f"{'gQ Avg':<12} {'Rel gQ Max':<12} "
        f"{'gK Avg':<12} {'Rel gK Max':<12} "
        f"{'gV Avg':<12} {'Rel gV Max':<12} "
        f"{'gO Avg':<12} {'Rel gO Max':<12}"
    )
    print("-" * 150)

    for config in test_configs:
        num_blocks = config["num_blocks"]
        k = config["k"]
        description = config["description"]
        results = check_correctness(h, d, num_blocks, k, error_mode=error_mode)
        print(
            f"{description:<20} {num_blocks:<8} {k:<4} "
            f"{results['gQ']['avg_diff']:<12.6e} {results['gQ']['max_diff']:<12.6e} "
            f"{results['gK']['avg_diff']:<12.6e} {results['gK']['max_diff']:<12.6e} "
            f"{results['gV']['avg_diff']:<12.6e} {results['gV']['max_diff']:<12.6e} "
            f"{results['gO']['avg_diff']:<12.6e} {results['gO']['max_diff']:<12.6e}"
        )

    print("-" * 150)


if __name__ == "__main__":
    h, d = 16, 128
    print("Block Sparse Attention with Variable Block Sizes Analysis")
    print("=" * 60)
    for mode in ["backward"]:
        generate_error_graphs(h, d, error_mode=mode)
    print("\nAnalysis completed for all modes.")
