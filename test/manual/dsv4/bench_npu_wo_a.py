"""Compare BF16 wo_a einsum and F.linear on one Ascend NPU.

python test/manual/dsv4/bench_npu_wo_a.py --batch-sizes 1 8

Simulates one DSV4 Flash TP8 rank with one local output group. Both variants
share the same inputs and 43 distinct layer weights by default. One iteration
replays all --layers projections; ms_per_iteration reports their total time,
and us_per_layer reports the average. Model loading and serving are not timed.
F.linear is called directly, as in the model, so no optimization flags are needed.
Optional --profile-dir exports a trace for checking weight transposes.
"""

import argparse
import json
import math
import statistics

import torch
import torch.nn.functional as F


def _einsum(o, weight):
    return torch.einsum("tgd,grd->tgr", o, weight.unsqueeze(0))


def _capture(fn, inputs, weights):
    def run():
        return [fn(o, w) for o, w in zip(inputs, weights)]

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            run()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        outputs = run()
    return graph, outputs


def _time_graph(graph, iterations):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument(
        "--layers",
        type=int,
        default=43,
        help="Distinct wo_a projections per graph replay.",
    )
    parser.add_argument(
        "--dim", type=int, default=4096, help="Input feature width of the local group."
    )
    parser.add_argument(
        "--o-lora-rank",
        type=int,
        default=1024,
        help="Output projection width, not a distributed rank ID.",
    )
    parser.add_argument(
        "--input-stride-factor",
        type=int,
        default=8,
        help="Token stride / D; 8 models the 64-head output sliced to 8 TP-local heads.",
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Graph replays per timing round."
    )
    parser.add_argument(
        "--rounds", type=int, default=5, help="Timing rounds; results use the median."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--profile-dir")
    args = parser.parse_args()
    if (
        min(
            *args.batch_sizes,
            args.layers,
            args.dim,
            args.o_lora_rank,
            args.input_stride_factor,
            args.iterations,
            args.rounds,
        )
        < 1
    ):
        parser.error("dimensions and iteration counts must be positive")

    import torch_npu

    torch.npu.set_device(args.device)
    torch.manual_seed(42)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "device": torch.npu.get_device_name(),
                "args": vars(args),
            }
        ),
        flush=True,
    )
    weights = [
        torch.randn(
            args.o_lora_rank, args.dim, device=args.device, dtype=torch.bfloat16
        )
        / math.sqrt(args.dim)
        for _ in range(args.layers)
    ]
    variants = {
        "einsum_original": _einsum,
        "linear_original": F.linear,
    }
    for tokens in args.batch_sizes:
        inputs = [
            torch.randn(
                tokens,
                args.input_stride_factor,
                args.dim,
                device=args.device,
                dtype=torch.bfloat16,
            )[:, :1, :]
            for _ in weights
        ]
        graphs = {}
        outputs = {}
        for name, fn in variants.items():
            graphs[name], outputs[name] = _capture(fn, inputs, weights)
            graphs[name].replay()
        torch.npu.synchronize()
        for actual, reference in zip(
            outputs["linear_original"], outputs["einsum_original"]
        ):
            torch.testing.assert_close(actual, reference, rtol=0.016, atol=0.016)

        # A captured graph must see an in-place online weight update even when
        # the update bypasses model.post_load_weights(), as direct updates do.
        pointer = weights[0].data_ptr()
        saved = weights[0].clone()
        before_update = outputs["einsum_original"][0].clone()
        weights[0].neg_()
        for graph in graphs.values():
            graph.replay()
        torch.npu.synchronize()
        assert not torch.equal(outputs["einsum_original"][0], before_update)
        # No bias: negating W must negate the result. This also catches a stale
        # weight in the baseline graph rather than only comparing two graphs.
        expected_after_update = -before_update
        for tensors in outputs.values():
            torch.testing.assert_close(
                tensors[0], expected_after_update, rtol=0.016, atol=0.016
            )
        assert weights[0].data_ptr() == pointer
        weights[0].copy_(saved)
        torch.npu.synchronize()

        samples = {name: [] for name in graphs}
        names = list(graphs)
        for round_id in range(args.rounds):
            # Rotate measurement order to reduce a fixed-order clock bias.
            offset = round_id % len(names)
            for name in names[offset:] + names[:offset]:
                samples[name].append(_time_graph(graphs[name], args.iterations))
        baseline = statistics.median(samples["einsum_original"])
        for name, times in samples.items():
            elapsed = statistics.median(times)
            print(
                json.dumps(
                    {
                        "batch_size": tokens,
                        "variant": name,
                        "layers": args.layers,
                        "input_stride": list(inputs[0].stride()),
                        "ms_per_iteration": elapsed,
                        "us_per_layer": elapsed * 1000 / args.layers,
                        "speedup": baseline / elapsed,
                        "samples_ms": times,
                    }
                ),
                flush=True,
            )

        if args.profile_dir:
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                record_shapes=True,
                with_stack=False,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    f"{args.profile_dir}/bs{tokens}"
                ),
            ) as profiler:
                for name, graph in graphs.items():
                    with torch.autograd.profiler.record_function(name):
                        graph.replay()
                        torch.npu.synchronize()
                    profiler.step()


if __name__ == "__main__":
    main()
