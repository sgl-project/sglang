from itertools import product

import torch
from flag_gems import silu_and_mul as flag_gems_silu_and_mul
from flashinfer.activation import silu_and_mul as flashinfer_silu_and_mul
from torch.utils.benchmark import Timer
from vllm import _custom_ops as ops


def forward_vllm(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=torch.float16, device=x.device)
    ops.silu_and_mul(out, x)
    return out


def forward_flashinfer(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out = torch.empty((*x.shape[:-1], d), dtype=torch.float16, device=x.device)
    flashinfer_silu_and_mul(out, x)
    return out


def forward_flag_gems(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return flag_gems_silu_and_mul(x[..., :d], x[..., d:])


def test_consistency():
    x = torch.randn(2, 4, 2 * d, dtype=torch.float16, device=device)
    out_vllm = forward_vllm(x)
    out_flashinfer = forward_flashinfer(x)
    out_flag_gems = forward_flag_gems(x)
    assert torch.allclose(out_vllm, out_flashinfer, atol=1e-3, rtol=1e-3)
    assert torch.allclose(out_vllm, out_flag_gems, atol=1e-3, rtol=1e-3)
    assert torch.allclose(out_flashinfer, out_flag_gems, atol=1e-3, rtol=1e-3)
    print("Consistency test passed!")


device = torch.device("cuda")
d = 4096

test_consistency()

results = []
sizes = [2, 8, 32, 128, 512]

for batch_size, seq_length in product(sizes, sizes):
    label = "SiLU and Mul"
    sub_label = f"[{batch_size}, {seq_length}]"

    input_tensor = torch.randn(
        batch_size, seq_length, 2 * d, dtype=torch.float16, device=device
    )

    min_run_time = max(0.1, min(1, batch_size * seq_length / 1e6))

    for num_threads in [1, 4, 16, 32]:
        results.append(
            Timer(
                stmt="forward_vllm(input_tensor)",
                setup="from __main__ import forward_vllm",
                globals={"input_tensor": input_tensor},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description="vLLM",
            ).blocked_autorange(min_run_time=min_run_time)
        )

        results.append(
            Timer(
                stmt="forward_flashinfer(input_tensor)",
                setup="from __main__ import forward_flashinfer",
                globals={"input_tensor": input_tensor},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description="FlashInfer",
            ).blocked_autorange(min_run_time=min_run_time)
        )

        results.append(
            Timer(
                stmt="forward_flag_gems(input_tensor)",
                setup="from __main__ import forward_flag_gems",
                globals={"input_tensor": input_tensor},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description="Flag_gems",
            ).blocked_autorange(min_run_time=min_run_time)
        )

compare = torch.utils.benchmark.Compare(results)
compare.print()
