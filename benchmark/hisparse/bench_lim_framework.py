"""Run native LIM+CopySFA mixed-row accuracy and optional subpath timings.

The fixture covers K/C/page boundaries, graph replay, ragged prefill, rollback,
radix-prefix reuse, request padding and an independent TP padding suffix.
Historical reference/legacy comparisons require their recorded source revision.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu


def main(default_widths="1,2,3,4,5,8,14", default_timing=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", default=default_widths)
    parser.add_argument("--capacity", type=int, help="C; defaults to at least Q*2048")
    parser.add_argument("--graph", choices=("on", "off", "both"), default="both")
    parser.add_argument(
        "--timing", action=argparse.BooleanOptionalAction, default=default_timing
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "test/manual/npu/test_lim_short_dynamic.py"
    )
    spec = importlib.util.spec_from_file_location("native_lim_fixture", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for graph in (False, True) if args.graph == "both" else (args.graph == "on",):
        for width in map(int, args.widths.split(",")):
            capacity = args.capacity or max(8192, (width * 2048 + 4095) // 4096 * 4096)
            if not 1 <= width <= 14 or capacity < width * 2048:
                parser.error("native query width must be 1..14 and C must cover Q*2048")
            record = {"width": width, "capacity": capacity, "graph": graph}
            fixture.test_short_sparse_graph_lifecycle(
                width, capacity, graph, record.__setitem__, measure=args.timing
            )
            record["accuracy"] = "pass"
            records.append(record)
            result = dict(
                scope="fixed-projection manager subpath; excludes model weights, scheduler and sampling",
                fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                cases=records,
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
