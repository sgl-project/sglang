"""Benchmark SGLang's Triton and cuDNN Frontend GDN prefill backends.

This uses the production backend adapters and Qwen3.5-4B's TP1 GDN shape:
16 Q/K heads, 32 value heads, and 128 dimensions per head. The benchmark
includes the state-pool gather/scatter performed by the cuDNN adapter.

Examples:
    python benchmark/bench_linear_attention/bench_gdn_prefill_cudnn.py
    python benchmark/bench_linear_attention/bench_gdn_prefill_cudnn.py \
        --batch-size 1 4 --seq-len 1024 4096 8192
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math

import torch
import triton.testing

from sglang.srt.layers.attention.linear.kernels.gdn_cudnn import CudnnGDNKernel
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _rms_ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float()
    expected = expected.float()
    error = (actual - expected).square().mean().sqrt()
    scale = expected.square().mean().sqrt().clamp_min(1e-12)
    return (error / scale).item()


def _make_inputs(
    batch_size: int,
    seq_len: int,
    q_heads: int,
    value_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    total = batch_size * seq_len
    device = "cuda"
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(42)

    q = torch.randn(
        1,
        total,
        q_heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1,
        total,
        value_heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    gate_logits = torch.randn(
        1,
        total,
        value_heads,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    g = -torch.nn.functional.softplus(gate_logits)
    beta = torch.rand(
        1,
        total,
        value_heads,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    query_start_loc = torch.arange(
        0,
        total + 1,
        seq_len,
        device=device,
        dtype=torch.int32,
    )
    cache_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
    states = torch.zeros(
        batch_size + 1,
        value_heads,
        head_dim,
        head_dim,
        device=device,
        dtype=torch.float32,
    )
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        states=states,
    )


def _run(kernel, inputs, states):
    return kernel.extend(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        ssm_states=states,
        cache_indices=inputs["cache_indices"],
        query_start_loc=inputs["query_start_loc"],
    )[0]


def _check_checkpoint_layout(
    triton_kernel: TritonGDNKernel,
    cudnn_kernel: CudnnGDNKernel,
    batch_size: int,
    seq_len: int,
    q_heads: int,
    value_heads: int,
    head_dim: int,
) -> float:
    if seq_len < 64:
        raise ValueError("checkpoint validation requires --seq-len >= 64")
    inputs = _make_inputs(batch_size, seq_len, q_heads, value_heads, head_dim)
    triton_result = triton_kernel.extend(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        ssm_states=inputs["states"].clone(),
        cache_indices=inputs["cache_indices"],
        query_start_loc=inputs["query_start_loc"],
    )
    cudnn_result = cudnn_kernel.extend(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        ssm_states=inputs["states"].clone(),
        cache_indices=inputs["cache_indices"],
        query_start_loc=inputs["query_start_loc"],
        state_checkpoint_every_n_tokens=64,
    )
    triton_h = triton_result[2]
    cudnn_h = cudnn_result[2]
    # cuDNN allocates a shape-derived upper bound; valid per-sequence rows are
    # packed first and match Triton's exact-size checkpoint tensor.
    cudnn_valid_h = cudnn_h[:, : triton_h.shape[1]]
    checkpoint_error = _rms_ratio(cudnn_valid_h, triton_h)
    if checkpoint_error > 2e-2:
        raise AssertionError(
            f"cuDNN/Triton checkpoint mismatch: rms={checkpoint_error:.3e}"
        )
    return checkpoint_error


def _bench_case(
    triton_kernel: TritonGDNKernel,
    cudnn_kernel: CudnnGDNKernel,
    batch_size: int,
    seq_len: int,
    q_heads: int,
    value_heads: int,
    head_dim: int,
    check_correctness: bool,
) -> tuple[float, float, float, float]:
    inputs = _make_inputs(batch_size, seq_len, q_heads, value_heads, head_dim)
    initial_states = inputs["states"]

    output_error = state_error = math.nan
    if check_correctness:
        triton_states = initial_states.clone()
        cudnn_states = initial_states.clone()
        triton_output = _run(triton_kernel, inputs, triton_states)
        cudnn_output = _run(cudnn_kernel, inputs, cudnn_states)
        torch.cuda.synchronize()
        output_error = _rms_ratio(cudnn_output, triton_output)
        state_error = _rms_ratio(cudnn_states[:batch_size], triton_states[:batch_size])
        if output_error > 2e-2 or state_error > 2e-2:
            raise AssertionError(
                "cuDNN/Triton GDN mismatch: "
                f"output_rms={output_error:.3e}, state_rms={state_error:.3e}"
            )

    triton_states = initial_states.clone()
    cudnn_states = initial_states.clone()
    triton_fn = lambda: _run(triton_kernel, inputs, triton_states)
    cudnn_fn = lambda: _run(cudnn_kernel, inputs, cudnn_states)

    # The first cuDNN call plans the graph and JIT-compiles the FROST kernel.
    triton_fn()
    cudnn_fn()
    torch.cuda.synchronize()
    triton_ms = triton.testing.do_bench(triton_fn, warmup=100, rep=500)
    cudnn_ms = triton.testing.do_bench(cudnn_fn, warmup=100, rep=500)
    return triton_ms, cudnn_ms, output_error, state_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton vs cuDNN Frontend GDN prefill."
    )
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seq-len", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--check-checkpoints",
        action="store_true",
        help="Also validate cuDNN's Radix-cache checkpoint layout on the first case.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU.")

    print(f"device: {torch.cuda.get_device_name()}")
    print(
        "versions: "
        f"torch={torch.__version__}, "
        f"cudnn-frontend={_version('nvidia-cudnn-frontend')}, "
        f"cutlass-dsl={_version('nvidia-cutlass-dsl')}"
    )
    print(
        f"shape: q_heads={args.q_heads}, value_heads={args.value_heads}, "
        f"head_dim={args.head_dim} (Qwen3.5-4B TP1)"
    )
    print(
        "batch  seq_len  tokens | triton_ms  cudnn_ms  speedup | output_rms  state_rms"
    )

    triton_kernel = TritonGDNKernel()
    cudnn_kernel = CudnnGDNKernel()
    if args.check_checkpoints:
        checkpoint_error = _check_checkpoint_layout(
            triton_kernel,
            cudnn_kernel,
            args.batch_size[0],
            args.seq_len[0],
            args.q_heads,
            args.value_heads,
            args.head_dim,
        )
        print(f"checkpoint_rms: {checkpoint_error:.3e}")
    for batch_size in args.batch_size:
        for seq_len in args.seq_len:
            triton_ms, cudnn_ms, output_error, state_error = _bench_case(
                triton_kernel,
                cudnn_kernel,
                batch_size,
                seq_len,
                args.q_heads,
                args.value_heads,
                args.head_dim,
                not args.skip_correctness,
            )
            print(
                f"{batch_size:>5}  {seq_len:>7}  {batch_size * seq_len:>6} | "
                f"{triton_ms:>9.3f}  {cudnn_ms:>8.3f}  "
                f"{triton_ms / cudnn_ms:>7.2f}x | "
                f"{output_error:>10.3e}  {state_error:>9.3e}"
            )


if __name__ == "__main__":
    main()
