# SPDX-License-Identifier: Apache-2.0
"""Compare the actual MiniMaxM3MLP with/without fused SwiGLU quantization.

Run from the SGLang root with PYTHONPATH=python SGLANG_JIT_DEEPGEMM_PRECOMPILE=0.
TP>1 measures one rank's GEMMs; no all-reduce or routed/shared overlap is timed.
"""

import argparse
import gc
import importlib.metadata
import json
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models.minimax_m3 import (
    _M3_FUSED_SWIGLU_MAX_TOKENS,
    MiniMaxM3MLP,
)
from sglang.test.layer_ut_utils import init_single_process_dist


def make_layer(hidden, intermediate, tp, backend):
    fp8_utils.FP8_GEMM_RUNNER_BACKEND = backend
    torch.manual_seed(20260915)
    config = SimpleNamespace(
        hidden_size=hidden, hidden_act="swigluoai", swiglu_alpha=1.702, swiglu_limit=7.0
    )
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", use_mxfp8=True
    )
    layer = MiniMaxM3MLP(
        config,
        quant_config=quant_config,
        intermediate_size=intermediate,
        tp_rank=0,
        tp_size=tp,
        reduce_results=False,
    ).cuda()
    for proj in [layer.gate_up_proj, layer.down_proj]:
        n, k = proj.weight.shape
        raw = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / k**0.5
        q, s = sglang_per_token_group_quant_fp8(
            raw, 32, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True
        )
        packed = s.T.contiguous().view(torch.uint8).reshape(k // 128, n, 4)
        raw_scales = packed.permute(1, 0, 2).reshape(n, k // 32)
        proj.weight.data.copy_(q)
        proj.weight_scale_inv.data.copy_(raw_scales)
        proj.quant_method.process_weights_after_loading(proj)
    return layer


def error(a, b):
    d = a.float() - b.float()
    if not a.numel():
        return {"max_abs": 0.0, "relative_rms": 0.0, "different_elements": 0}
    return {
        "max_abs": d.abs().max().item(),
        "relative_rms": (
            (
                d.square().mean().sqrt()
                / a.float().square().mean().sqrt().clamp_min(1e-20)
            ).item()
        ),
        "different_elements": (d != 0).sum(dtype=torch.int64).item(),
    }


def emit(path, row):
    print(json.dumps(row), flush=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def setup(args):
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.manual_seed(20260913)
    torch._dynamo.config.recompile_limit = 64
    emit(
        args.output,
        {
            "kind": "environment",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("triton", "flashinfer-python", "sgl-deep-gemm")
            },
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "input_weights": "synthetic; public MiniMax-M3-MXFP8 dimensions",
            "rounds": args.rounds,
        },
    )


def stat(values):
    ordered = sorted(values)
    return {
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
        "p90_us": ordered[int((len(ordered) - 1) * 0.9)],
        "samples_us": values,
    }


def measure(functions, rounds, inner):
    graphs, outputs = {}, {}
    for name, fn in functions.items():
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(inner):
                outputs[name] = fn()
        graphs[name] = graph
        deadline = time.perf_counter() + 0.2
        while time.perf_counter() < deadline:
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = {key: [] for key in functions}
    rng = random.Random(84)
    for _ in range(rounds):
        names = list(functions)
        rng.shuffle(names)
        for name in names:
            start.record()
            for _ in range(10):
                graphs[name].replay()
            end.record()
            torch.cuda.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / (10 * inner))
    eager = {}
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
    for name, fn in functions.items():
        for a, b in zip(starts, ends):
            a.record()
            fn()
            b.record()
        torch.cuda.synchronize()
        eager[name] = stat([a.elapsed_time(b) * 1000 for a, b in zip(starts, ends)])
    return {"graph_warm": {k: stat(v) for k, v in samples.items()}, "eager": eager}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 30, 240, 400, 1024, 8192]
    )
    parser.add_argument("--tp", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--rounds", type=int, default=11)
    args = parser.parse_args()
    setup(args)
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    publish(ServerArgs(model_path="dummy"), role="test")
    init_single_process_dist(master_port=29683)
    try:
        for model, h, intermediate in [
            ("m3-shared", 6144, 3072),
            ("m3-dense", 6144, 12288),
        ]:
            for tp in args.tp:
                layer = make_layer(
                    h, intermediate, tp, fp8_utils.Fp8GemmRunnerBackend.DEEP_GEMM
                )
                auto_layer = make_layer(
                    h, intermediate, tp, fp8_utils.Fp8GemmRunnerBackend.AUTO
                )
                assert layer._fuse_swiglu_mxfp8, (
                    "Expected default DeepGEMM fast path; check environment"
                )
                for m in args.rows:
                    x = torch.randn(m, h, dtype=torch.bfloat16, device="cuda")

                    def baseline():
                        layer._fuse_swiglu_mxfp8 = False
                        return layer(x)

                    def candidate():
                        layer._fuse_swiglu_mxfp8 = True
                        return layer(x)

                    def default_backend():
                        return auto_layer(x)

                    expected, actual = baseline(), candidate()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    emit(
                        args.output,
                        {
                            "model": model,
                            "m": m,
                            "hidden": h,
                            "intermediate_local": intermediate // tp,
                            "tp_local_shape": tp,
                            "fusion_active": 0 < m <= _M3_FUSED_SWIGLU_MAX_TOKENS,
                            "component": "MiniMaxM3MLP (no collective)",
                            "correctness": error(expected, actual),
                            "auto_backend": str(
                                auto_layer.down_proj.quant_method.mxfp8_dense_backend
                            ),
                            "auto_vs_deepgemm_error": error(
                                expected, default_backend()
                            ),
                            **measure(
                                {
                                    "baseline": baseline,
                                    "candidate": candidate,
                                    "auto": default_backend,
                                },
                                args.rounds,
                                16 if m < 1024 else 4,
                            ),
                        },
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
    finally:
        from sglang.srt.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
