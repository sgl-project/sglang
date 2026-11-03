"""Compare staging implementations with identical graph replay inputs.

Run with --module pointing to either residency Python file in an initialized
NPU environment. This does not benchmark model throughput or validate accuracy.
"""

import argparse
import importlib.util
import json
import time

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import create_shm_tensor, free_shm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=12288)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--short-length", type=int, default=1024)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("staging_under_test", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device_id = torch.npu.current_device()
    device = torch.device("npu", device_id)
    b, c, q, d = args.batch, args.capacity, args.width, 576
    context = c + 128
    host, _, pointer = create_shm_tensor(
        shape=(1, (b + 1) * context, 1, d),
        dtype=torch.bfloat16,
        device_id=device_id,
        name="resident_staging_benchmark",
    )
    host.zero_()
    cache = torch.zeros((b + 1, c, 1, d), dtype=torch.bfloat16, device=device)
    mapping = torch.arange((b + 1) * context, dtype=torch.int32, device=device).view(
        b + 1, context
    )
    requests = torch.arange(1, b + 1, dtype=torch.int32, device=device)
    lengths = torch.empty(b, dtype=torch.int32, device=device)
    first = torch.empty_like(lengths)
    token_requests = requests.repeat_interleave(q)
    positions = torch.empty(b * q, dtype=torch.int32, device=device)
    valid = torch.ones(b * q, dtype=torch.bool, device=device)
    current = torch.ones((b * q, 1, d), dtype=torch.bfloat16, device=device)
    sources = torch.arange(b * q, dtype=torch.int64, device=device)
    old_begin = torch.full((b + 1,), -1, dtype=torch.int32, device=device)
    old_end = old_begin.clone()
    extents = tuple(old_begin.clone() for _ in range(3))
    history = (
        torch.empty(b * c, dtype=torch.int64, device=device),
        torch.empty(b * c, dtype=torch.int64, device=device),
        torch.empty(b * c, dtype=torch.bool, device=device),
    )
    workspace = (
        torch.empty(b * q, dtype=torch.int64, device=device),
        torch.empty(b * q, dtype=torch.bool, device=device),
    )

    def run():
        module.stage_resident_kv(
            current=current,
            source_indices=sources,
            host=host,
            host_address=pointer,
            cache=cache,
            req_to_token=mapping,
            requests=requests,
            lengths=lengths,
            first_positions=first,
            token_requests=token_requests,
            token_positions=positions,
            token_valid=valid,
            old_begin=old_begin,
            old_end=old_end,
            next_extents=extents,
            history_workspace=history,
            current_workspace=workspace,
            capacity=c,
            context_len=context,
        )

    try:
        for label, length in (("long", c), ("short_warm", args.short_length)):
            lengths.fill_(length)
            first.fill_(length - q)
            positions.copy_(
                torch.arange(
                    length - q, length, dtype=torch.int32, device=device
                ).repeat(b)
            )
            old_begin.fill_(-1)
            old_end.fill_(-1)
            run()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                run()
            for _ in range(10):
                graph.replay()
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(args.iterations):
                graph.replay()
            torch.npu.synchronize()
            elapsed = time.perf_counter() - start
            print(
                json.dumps(
                    {
                        "case": label,
                        "module": args.module,
                        "batch": b,
                        "capacity": c,
                        "width": q,
                        "length": length,
                        "iterations": args.iterations,
                        "replay_mean_us": elapsed * 1e6 / args.iterations,
                    }
                )
            )
    finally:
        torch.npu.synchronize()
        free_shm(device_id)


if __name__ == "__main__":
    main()
