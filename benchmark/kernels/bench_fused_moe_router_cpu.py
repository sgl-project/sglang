#!/usr/bin/env python3
import argparse
import statistics
import time

import sgl_kernel  # noqa: F401
import torch


GROK_SHAPES = [
    # (tokens, hidden_size, num_experts, topk)
    (1, 6144, 8, 2),
    (8, 6144, 8, 2),
    (32, 6144, 8, 2),
    (128, 6144, 8, 2),
    (512, 6144, 8, 2),
    (1024, 6144, 8, 2),
]


def reference_router(hidden_states, gating_output, topk, softcap, correction_bias=None):
    logits = hidden_states.float() @ gating_output.float().t()
    if softcap != 0:
        logits = torch.tanh(logits / softcap) * softcap
    if correction_bias is not None:
        logits = logits + correction_bias.float()
    return torch.topk(torch.softmax(logits, dim=-1, dtype=torch.float32), topk, dim=-1)


def xeon_router(hidden_states, gating_output, topk, softcap, correction_bias=None):
    return torch.ops.sgl_kernel.fused_moe_router_cpu(
        hidden_states, gating_output, topk, softcap, correction_bias
    )


def make_inputs(tokens, hidden_size, num_experts, dtype):
    hidden_states = (torch.randn((tokens, hidden_size), dtype=dtype) / 16).contiguous()
    gating_output = (
        torch.randn((num_experts, hidden_size), dtype=torch.float32) / 16
    ).contiguous()
    return hidden_states, gating_output


def time_ms(fn, args, warmup, repeat):
    for _ in range(warmup):
        fn(*args)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def parse_shapes(spec):
    if not spec:
        return GROK_SHAPES
    shapes = []
    for item in spec.split(","):
        tokens, hidden_size, num_experts, topk = (int(part) for part in item.split("x"))
        shapes.append((tokens, hidden_size, num_experts, topk))
    return shapes


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Xeon AVX512 fused_moe_router_cpu against PyTorch and torch.compile."
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--softcap", type=float, default=30.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--shapes",
        default="",
        help="Comma-separated tokensxhiddenxexpertsxtopk entries. Default: Grok router shapes.",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    shapes = parse_shapes(args.shapes)

    compiled_router = None
    compile_error = None
    try:
        compiled_router = torch.compile(reference_router, dynamic=False, fullgraph=True)
    except Exception as exc:  # pragma: no cover - depends on installed torch/inductor
        compile_error = repr(exc)

    print(
        "dtype,tokens,hidden_size,num_experts,topk,pytorch_ms,torch_compile_ms,xeon_avx512_ms,"
        "speedup_vs_pytorch,speedup_vs_compile"
    )

    for tokens, hidden_size, num_experts, topk in shapes:
        torch.manual_seed(tokens + hidden_size + num_experts + topk)
        hidden_states, gating_output = make_inputs(tokens, hidden_size, num_experts, dtype)
        bench_args = (hidden_states, gating_output, topk, args.softcap, None)

        ref_weights, ref_ids = reference_router(*bench_args)
        out_weights, out_ids = xeon_router(*bench_args)
        torch.testing.assert_close(out_ids, ref_ids.to(torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(out_weights, ref_weights, rtol=5e-4, atol=5e-4)

        pytorch_ms = time_ms(reference_router, bench_args, args.warmup, args.repeat)
        xeon_ms = time_ms(xeon_router, bench_args, args.warmup, args.repeat)

        compile_ms = float("nan")
        if compiled_router is not None:
            try:
                c_weights, c_ids = compiled_router(*bench_args)
                torch.testing.assert_close(c_ids, ref_ids, rtol=0, atol=0)
                torch.testing.assert_close(c_weights, ref_weights, rtol=5e-4, atol=5e-4)
                compile_ms = time_ms(compiled_router, bench_args, args.warmup, args.repeat)
            except Exception as exc:  # pragma: no cover - depends on installed torch/inductor
                if compile_error is None:
                    compile_error = repr(exc)
                compiled_router = None

        speedup_vs_pytorch = pytorch_ms / xeon_ms
        speedup_vs_compile = compile_ms / xeon_ms if compile_ms == compile_ms else float("nan")
        print(
            f"{args.dtype},{tokens},{hidden_size},{num_experts},{topk},"
            f"{pytorch_ms:.4f},{compile_ms:.4f},{xeon_ms:.4f},"
            f"{speedup_vs_pytorch:.2f},{speedup_vs_compile:.2f}"
        )

    if compile_error is not None:
        print(f"torch_compile_error={compile_error}")


if __name__ == "__main__":
    main()
