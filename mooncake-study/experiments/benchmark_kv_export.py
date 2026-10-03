"""Complete selected-layer KV export timing, including async bounds validation."""

import argparse
import json
import statistics
import time
from pathlib import Path

import msgspec
import torch
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.protocol import LayerGeometry, validate_kv_spec
from sglang.test.training_capture_utils import Registrar, make_kv_spec


def measure(fn, iterations):
    torch.cuda.synchronize()
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin.record()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return {
        "wall_us": (time.perf_counter() - start) * 1e6 / iterations,
        "stream_us": begin.elapsed_time(end) * 1000 / iterations,
    }


def run_case(args, *, dtype, index_dtype, rows, destination):
    kv = msgspec.structs.replace(
        make_kv_spec(),
        dtype=str(dtype).removeprefix("torch."),
        codec=f"dense_{'bf16' if dtype == torch.bfloat16 else 'fp16'}_post_rope_v1",
        selected_layer_ids=[0, 14, 27],
        layers=[
            LayerGeometry(
                layer_id=i, num_kv_heads=8, key_head_dim=128, value_head_dim=128
            )
            for i in (0, 14, 27)
        ],
    )
    validate_kv_spec(kv)
    sources = {
        f"target_{kind}.{layer.layer_id}": torch.randn(
            4096, 8, 128, dtype=dtype, device="cuda"
        )
        for layer in kv.layers
        for kind in ("k", "v")
    }
    exporter = SelectedLayerKVExporter(kv, sources)
    pool = HostBufferPool(
        kv=kv,
        max_tokens=rows + 7,
        slots=1,
        max_bytes=16 << 20,
        registrar=Registrar(),
        device=torch.device("cuda"),
        kv_d2h_batch_tokens=rows + 7,
        kv_export_backend="hicache",
        max_device_bytes=16 << 20,
    )
    slot = pool.acquire()
    graphs = {}
    try:
        slot.kv_exporter = exporter.bind(slot)
        tensors = slot.tensors if destination == "host" else slot.device_tensors
        indices = torch.randint(0, 4096, (rows,), dtype=index_dtype, device="cuda")
        functions = {
            "torch": lambda: exporter.export(indices, tensors, 3, rows + 3),
            "hicache": lambda: slot.kv_exporter.export(indices, tensors, 3, rows + 3),
        }
        # Fresh source values for each backend make the source-reuse check real.
        for fn in functions.values():
            for source in sources.values():
                source.normal_()
            expected = {
                name: source.index_select(0, indices).cpu()
                for name, source in sources.items()
            }
            for name in sources:
                tensors[name].fill_(-19)
            torch.cuda._sleep(200000)
            fn()
            for source in sources.values():
                source.fill_(31)
            completion = torch.cuda.Event()
            completion.record()
            completion.synchronize()
            for name, value in expected.items():
                actual = tensors[name].cpu()
                torch.testing.assert_close(actual[3 : rows + 3], value, rtol=0, atol=0)
                if not (
                    (actual[:3] == -19).all() and (actual[rows + 3 :] == -19).all()
                ):
                    raise AssertionError("export modified rows outside its destination")

        for fn in functions.values():
            for _ in range(10):
                fn()
        torch.cuda.synchronize()
        for name, fn in functions.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.graph_batch):
                    fn()
            graphs[name] = graph
        timings = {name: {"eager": [], "graph": []} for name in functions}
        for repeat in range(args.repeats):
            order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
            for name in order:
                timings[name]["eager"].append(measure(functions[name], args.iterations))
                timing = measure(graphs[name].replay, args.iterations)
                timings[name]["graph"].append(
                    {key: value / args.graph_batch for key, value in timing.items()}
                )
        return {
            "dtype": str(dtype),
            "index_dtype": str(index_dtype),
            "rows": rows,
            "destination": destination,
            "exact_source_reuse_and_guard_rows": True,
            "device_allocated_bytes": pool.stats()["device_allocated_bytes"],
            "timings": timings,
            "median_us": {
                name: {
                    mode: {
                        key: statistics.median(sample[key] for sample in samples)
                        for key in ("wall_us", "stream_us")
                    }
                    for mode, samples in modes.items()
                }
                for name, modes in timings.items()
            },
        }
    finally:
        torch.cuda.synchronize()
        graphs.clear()
        pool.release(slot, transfer_complete=True)
        pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 128])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--graph-batch", type=int, default=10)
    args = parser.parse_args()
    if min(*args.rows, args.iterations, args.repeats, args.graph_batch) < 1:
        parser.error("Positive measurement dimensions are required")
    if max(args.rows) > 1024:
        parser.error("This fixture supports up to 1024 rows")
    if args.output.exists():
        parser.error("Use a fresh output path")
    torch.set_num_threads(1)
    torch.manual_seed(9)
    report = {
        "status": "running",
        "source_revision": args.source_revision,
        "config": vars(args) | {"output": str(args.output)},
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "scope": "Complete export calls; eager includes Python/validation and synchronization, graph batches device work. Not a serving SLO.",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for dtype in (torch.bfloat16, torch.float16):
            for index_dtype in (torch.int32, torch.int64):
                for rows in args.rows:
                    for destination in ("host", "device"):
                        result = run_case(
                            args,
                            dtype=dtype,
                            index_dtype=index_dtype,
                            rows=rows,
                            destination=destination,
                        )
                        report["cases"].append(result)
                        print(json.dumps(result), flush=True)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
