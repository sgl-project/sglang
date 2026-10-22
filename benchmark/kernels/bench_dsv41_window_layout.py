"""Measure eager request-window metadata construction on synthetic inputs.

Run this file with PYTHONPATH pointing to each checkout's python directory.
Compilation and warmup are outside the measured interval.
"""

import argparse
import json
import time

import torch

from sglang.srt.mem_cache.dsv41_request_window import window_layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    for rows, groups in ((4096, 1), (32768, 8)):
        req = torch.arange(groups, device="cuda").repeat_interleave(rows // groups)
        pos = torch.arange(8192, 8192 + rows // groups, device="cuda").repeat(groups)
        floor = torch.full_like(pos, 8192)

        def run():
            return window_layout(
                req, pos, zero_row=65536, floor=floor, num_groups=groups
            )

        for _ in range(20):
            run()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            for _ in range(args.iterations):
                run()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000 / args.iterations)
        print(
            json.dumps({"rows": rows, "groups": groups, "wall_ms": samples}), flush=True
        )


if __name__ == "__main__":
    main()
