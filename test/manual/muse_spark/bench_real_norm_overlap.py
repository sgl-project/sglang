"""Real BF16 backend A/B; base host method and PR method share weights/inputs.

Not a serving benchmark. Imports the registered real-component fixture.
"""

import argparse
import ast
import importlib.util
import json
import statistics
from pathlib import Path

import torch

from sglang.srt.models import muse_spark


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-source", required=True)
    p.add_argument("--backend", choices=["triton", "trtllm"], required=True)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 129])
    p.add_argument("--hidden", type=int, default=8192)
    p.add_argument("--expert-hidden", type=int, default=4096)
    p.add_argument("--intermediate", type=int, default=4096)
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--graph", action="store_true")
    args = p.parse_args()
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "registered/kernels/ops/moe/test_muse_spark_norm_overlap.py"
    )
    spec = importlib.util.spec_from_file_location("real_fixture", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    tree = ast.parse(Path(args.baseline_source).read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MuseSpark1xMoE"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_forward_experts"
    )
    namespace = dict(vars(muse_spark))
    exec(
        compile(
            ast.Module(
                body=[*ast.parse("from __future__ import annotations").body, method],
                type_ignores=[],
            ),
            args.baseline_source,
            "exec",
        ),
        namespace,
    )
    baseline = namespace["_forward_experts"]
    report = {
        "config": vars(args),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "rows": [],
    }
    with torch.inference_mode(), fixture.single_rank():
        torch.manual_seed(args.seed)
        layer = fixture.make_moe(
            args.backend,
            hidden=args.hidden,
            expert_hidden=args.expert_hidden,
            intermediate=args.intermediate,
            experts=args.experts,
        )
        for tokens in args.tokens:
            x = torch.randn(tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
            runs = [lambda: baseline(layer, x), lambda: layer._forward_experts(x)]
            for _ in range(5):
                a, b = [run() for run in runs]
            torch.cuda.synchronize()
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            graphs, outputs = [], []
            if args.graph:
                for run in runs:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = run()
                    graphs.append(graph)
                    outputs.append(output)
                for _ in range(5):
                    x.normal_()
                    expected = fixture.serialized(layer, x)
                    for graph in graphs:
                        graph.replay()
                    torch.cuda.synchronize()
                    for out in outputs:
                        torch.testing.assert_close(out, expected, rtol=0, atol=0)
                runs = [graph.replay for graph in graphs]
            raw = []
            for pair in range(args.pairs):
                measurements = [None, None]
                for idx in [0, 1] if pair % 2 == 0 else [1, 0]:
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(args.iterations):
                        runs[idx]()
                    end.record()
                    end.synchronize()
                    measurements[idx] = start.elapsed_time(end) * 1000 / args.iterations
                raw.append(measurements)
            report["rows"].append(
                {
                    "tokens": tokens,
                    "baseline_candidate_us": raw,
                    "median_baseline_us": statistics.median(x[0] for x in raw),
                    "median_candidate_us": statistics.median(x[1] for x in raw),
                }
            )
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
