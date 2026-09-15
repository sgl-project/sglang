"""Reproduce CUDA block-FP8 K-group unrolling measurements, without model weights.

python benchmark/kernels/quantization/bench_block_fp8_k_groups.py

Optional: --baseline-module /path/to/original/fp8_kernel.py compares against a
previous implementation. The default compares both paths in the current module.
These fixed H200 configs are examples, not defaults or a claim of global optima.
"""

import argparse
import importlib.util
import json
import statistics

import torch
import triton

import sglang.kernels.ops.quantization.fp8_kernel as fp8


def config(bk, bn, stages, gm=1):
    return dict(
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=bk,
        GROUP_SIZE_M=gm,
        num_warps=4,
        num_stages=stages,
    )


def make_call(module, a, b, sa, sb, cfg):
    m, k = a.shape
    n = b.shape[0]
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    kernel = module._w8a8_block_fp8_matmul
    if cfg["BLOCK_SIZE_K"] > 32:
        kernel = module._w8a8_block_fp8_matmul_k_groups
    grid = (triton.cdiv(m, cfg["BLOCK_SIZE_M"]) * triton.cdiv(n, cfg["BLOCK_SIZE_N"]),)

    def call():
        kernel[grid](
            a,
            b,
            out,
            sa,
            sb,
            m,
            n,
            k,
            32,
            32,
            a.stride(0),
            a.stride(1),
            b.stride(1),
            b.stride(0),
            out.stride(0),
            out.stride(1),
            sa.stride(0),
            sa.stride(1),
            sb.stride(1),
            sb.stride(0),
            **cfg,
            needs_masking=k % cfg["BLOCK_SIZE_K"] != 0,
        )

    return call, out


def measure(call, scratch=None):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    if scratch is None:
        with torch.cuda.graph(graph):
            for _ in range(200):
                call()
        graph.replay()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
    else:
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            scratch.zero_()
            start.record()
            call()
            end.record()
    times = []
    for _ in range(5):
        if scratch is None:
            start.record()
        graph.replay()
        if scratch is None:
            end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (200 if scratch is None else 1))
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-module")
    args = parser.parse_args()
    baseline = fp8
    if args.baseline_module:
        spec = importlib.util.spec_from_file_location(
            "fp8_baseline", args.baseline_module
        )
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
    torch.manual_seed(307)
    torch.backends.cuda.matmul.allow_tf32 = False
    scratch = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                triton=triton.__version__,
                cuda=torch.version.cuda,
            )
        ),
        flush=True,
    )
    for n, k, warm, scrubbed in [
        (1152, 5120, config(32, 16, 4), config(32, 16, 5)),
        (3072, 1280, config(32, 64, 4), config(32, 32, 5, 32)),
    ]:
        m = 16
        a = (torch.randint(-4, 5, (m, k), device="cuda").float() / 4).to(
            torch.float8_e4m3fn
        )
        b = (torch.randint(-4, 5, (n, k), device="cuda").float() / 4).to(
            torch.float8_e4m3fn
        )
        sa = 2.0 ** torch.randint(-1, 2, (m, k // 32), device="cuda").float()
        sb = 2.0 ** torch.randint(-1, 2, (n // 32, k // 32), device="cuda").float()
        da = a.double() * sa.double().repeat_interleave(32, 1)
        db = b.double() * sb.double().repeat_interleave(32, 0).repeat_interleave(32, 1)
        reference = (da @ db.t()).to(torch.bfloat16)
        for mode, cfg in [("warm", warm), ("scrubbed", scrubbed)]:
            configs = {"baseline": cfg, "unrolled": config(128, 32, 3)}
            functions = {}
            for label, module in [("baseline", baseline), ("unrolled", fp8)]:
                call, out = make_call(module, a, b, sa, sb, configs[label])
                call()
                torch.testing.assert_close(out, reference, rtol=0, atol=0)
                functions[label] = call
            samples = {label: [] for label in functions}
            for iteration in range(10):
                order = (
                    list(functions) if iteration % 2 == 0 else list(reversed(functions))
                )
                for label in order:
                    samples[label].append(
                        measure(
                            functions[label], scratch if mode == "scrubbed" else None
                        )
                    )
            print(
                json.dumps(
                    dict(
                        shape=[m, n, k],
                        mode=mode,
                        configs=configs,
                        samples_us=samples,
                        median_us={
                            label: statistics.median(v) for label, v in samples.items()
                        },
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
