"""Full synthetic shared MLP benchmark through the fused producer and consumer.

From the repository root, select an idle SM89 GPU and run:
    SGLANG_ENABLE_SILU_STATIC_FP8_FUSION=1 python \
        benchmark/kernels/bench_silu_fp8_fusion_mlp.py \
        --model-path /path/to/Qwen3-8B-FP8 --output /path/to/new-results

The checkpoint bootstraps the normal Engine runtime; loading and the bootstrap
request are outside module timing. Measures both projections and the fused or
unfused activation/quantization with synthetic weights, 9 alternating pairs and
20 Graph replays per measurement. Untimed traces record the actual GEMMs.
This is not a full MoE checkpoint accuracy test or an HTTP latency benchmark.
"""

import gc
import json
import statistics
import traceback
from pathlib import Path

import torch


def capture(fn, x):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            fn(x)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y = fn(x)
    return graph, y


def measure(graphs, iterations=20):
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for i in range(iterations):
        graphs[i % len(graphs)].replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def make_hook(config):
    done = False
    OUT = Path(config["output"])

    def hook(module, args, output):
        nonlocal done
        if done:
            return
        done = True
        from sglang.kernels.ops.quantization.silu_and_mul_static_fp8 import (
            silu_and_mul_static_fp8,
        )
        from sglang.srt.layers.quantization.fp8_utils import static_quant_fp8
        from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

        OUT.mkdir(parents=True, exist_ok=False)
        result = dict(
            status="running",
            synthetic_weights=True,
            tp=1,
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
            scope="gate_up + activation/quant + down; actual delivered Qwen2MoeMLP.forward and ModelOpt.apply",
            baseline="real Qwen2MoeMLP.forward; reduce_results=False",
            cases=[],
        )

        def save():
            (OUT / "results.json").write_text(json.dumps(result, indent=2))

        original_dtype = torch.get_default_dtype()
        try:
            with (
                torch.inference_mode(),
                torch.random.fork_rng(devices=[0]),
                torch.device("cuda"),
            ):
                torch.manual_seed(9271)
                for dtype in (torch.bfloat16, torch.float16):
                    torch.set_default_dtype(dtype)
                    for width in (256, 512, 1024):
                        mlp = Qwen2MoeMLP(
                            4096,
                            width,
                            "silu",
                            module.down_proj.quant_method.quant_config,
                            reduce_results=False,
                            prefix="validation.shared_expert",
                            tp_rank=0,
                            tp_size=1,
                            allow_silu_fp8_quant=True,
                        )
                        for linear in (mlp.gate_up_proj, mlp.down_proj):
                            assert (
                                type(linear.quant_method).__name__
                                == "ModelOptFp8LinearMethod"
                            )
                            linear.weight.copy_(
                                torch.randn(linear.weight.shape, dtype=dtype).to(
                                    torch.float8_e4m3fn
                                )
                            )
                            linear.weight_scale.fill_(0.01)
                            linear.input_scale.fill_(0.02)
                            linear.quant_method.process_weights_after_loading(linear)
                        down = mlp.down_proj
                        method = down.quant_method
                        assert not method.use_marlin and not method.use_sm120_fp8
                        assert not down.use_flashinfer_bmm and down.bias is None
                        assert not mlp._enable_silu_fp4_quant_fusion

                        fusion = mlp._silu_fp8_fusion
                        assert fusion is not None

                        def baseline(x, mlp=mlp, fusion=fusion):
                            mlp._silu_fp8_fusion = None
                            try:
                                return mlp(x)
                            finally:
                                mlp._silu_fp8_fusion = fusion

                        candidate = mlp

                        for m in (1, 16, 128, 512, 2048, 8192):
                            xs = [torch.randn(m, 4096, dtype=dtype) for _ in range(4)]
                            gate, _ = mlp.gate_up_proj(xs[0])
                            q, rows = silu_and_mul_static_fp8(gate, down.input_scale)
                            qr, rr = static_quant_fp8(
                                mlp.act_fn(gate), down.input_scale, repeat_scale=True
                            )
                            assert torch.equal(
                                q.view(torch.uint8), qr.view(torch.uint8)
                            )
                            assert torch.equal(rows, rr)
                            ref = baseline(xs[0])
                            assert torch.equal(ref, candidate(xs[0])) and bool(
                                ref.isfinite().all()
                            )
                            base_pairs = [capture(baseline, x) for x in xs]
                            fused_pairs = [capture(candidate, x) for x in xs]
                            for factor in (0.75, -0.5):
                                xs[0].mul_(factor)
                                down.input_scale.mul_(1.25)
                                base_pairs[0][0].replay()
                                fused_pairs[0][0].replay()
                                torch.cuda.synchronize()
                                assert torch.equal(base_pairs[0][1], fused_pairs[0][1])
                                assert torch.equal(mlp(xs[0]), fused_pairs[0][1])
                            down.input_scale.fill_(0.02)
                            timing = {}
                            for mode, count in (
                                ("hot", 1),
                                ("rotating_inputs_outputs", 4),
                            ):
                                ga = [p[0] for p in base_pairs[:count]]
                                gb = [p[0] for p in fused_pairs[:count]]
                                pairs = []
                                for i in range(9):
                                    order = [("base", ga), ("fused", gb)]
                                    if i % 2:
                                        order.reverse()
                                    values = {
                                        name: measure(graphs) for name, graphs in order
                                    }
                                    pairs.append(
                                        dict(
                                            baseline_us=values["base"],
                                            fused_us=values["fused"],
                                            reduction_pct=100
                                            * (1 - values["fused"] / values["base"]),
                                        )
                                    )
                                timing[mode] = pairs
                            row = dict(
                                dtype=str(dtype),
                                M=m,
                                K=width,
                                H=4096,
                                numeric="byte_and_output_exact",
                                changed_input_scale_replays=2,
                                timing=timing,
                            )
                            result["cases"].append(row)
                            save()
                            print(
                                "CASE",
                                str(dtype),
                                width,
                                m,
                                {
                                    mode: round(
                                        statistics.median(
                                            p["reduction_pct"] for p in pairs
                                        ),
                                        3,
                                    )
                                    for mode, pairs in timing.items()
                                },
                                flush=True,
                            )
                            if dtype == torch.bfloat16 and m == 2048:
                                for name, fn in (
                                    ("baseline", baseline),
                                    ("fused", candidate),
                                ):
                                    torch.cuda.synchronize()
                                    with torch.profiler.profile(
                                        activities=[
                                            torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA,
                                        ]
                                    ) as prof:
                                        fn(xs[0])
                                        torch.cuda.synchronize()
                                    prof.export_chrome_trace(
                                        str(OUT / f"K{width}-{name}-trace.json")
                                    )
                            del (
                                base_pairs,
                                fused_pairs,
                                ga,
                                gb,
                                xs,
                                gate,
                                q,
                                rows,
                                qr,
                                rr,
                                ref,
                            )
                            gc.collect()
                        del mlp, down, method
                        gc.collect()
                result["status"] = "passed"
        except BaseException:
            result["status"] = "failed"
            result["error"] = traceback.format_exc()
            raise
        finally:
            torch.set_default_dtype(original_dtype)
            save()

    return hook


def main():
    import argparse

    import sglang
    from sglang.srt.environ import envs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local Qwen3-8B ModelOpt FP8 checkpoint for runtime initialization",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New result directory; existing directories are rejected",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    if not envs.SGLANG_ENABLE_SILU_STATIC_FP8_FUSION.get():
        parser.error("Set SGLANG_ENABLE_SILU_STATIC_FP8_FUSION=1")

    engine = None
    try:
        engine = sglang.Engine(
            model_path=args.model_path,
            tp_size=1,
            dtype="bfloat16",
            kv_cache_dtype="bfloat16",
            random_seed=9271,
            context_length=4096,
            max_running_requests=4,
            max_total_tokens=8192,
            mem_fraction_static=0.5,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            log_level="info",
            forward_hooks=[
                dict(
                    name="shared-full-module",
                    target_modules=["model.layers.0.mlp"],
                    hook_factory="bench_silu_fp8_fusion_mlp:make_hook",
                    config=dict(output=str(args.output.resolve())),
                )
            ],
        )
        engine.generate(
            "The capital of France is", dict(temperature=0, max_new_tokens=8)
        )
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
