#!/usr/bin/env python3
"""Interruptible CUDA load for an otherwise idle, allocated experiment GPU."""

import argparse
import os
import signal
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--duty-cycle", type=float, default=0.60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--duration", type=float, default=0)
    args = parser.parse_args()
    if not 0 < args.duty_cycle <= 1 or min(args.matrix_size, args.batch_size) < 1:
        parser.error("positive sizes and a duty cycle in (0, 1] are required")
    if args.duration < 0:
        parser.error("duration must be nonnegative")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("The resident worker requires exactly one visible GPU")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    a = torch.randn(
        args.matrix_size, args.matrix_size, device="cuda", dtype=torch.bfloat16
    )
    b = torch.randn_like(a) / args.matrix_size**0.5
    out = torch.empty_like(a)
    started = last_report = time.monotonic()
    iterations = 0
    print(
        f"idle CUDA load pid={os.getpid()} device={torch.cuda.get_device_name(0)} "
        f"allocated_bytes={torch.cuda.memory_allocated()} duty_cycle={args.duty_cycle}",
        flush=True,
    )
    with torch.inference_mode():
        while not stop.is_set():
            tick = time.monotonic()
            if args.duration and tick - started >= args.duration:
                break
            for _ in range(args.batch_size):
                torch.mm(a, b, out=out)
            torch.cuda.synchronize()
            iterations += args.batch_size
            busy = time.monotonic() - tick
            stop.wait(busy * (1 - args.duty_cycle) / args.duty_cycle)
            if time.monotonic() - last_report >= 60:
                print(f"idle matmul iterations={iterations}", flush=True)
                last_report = time.monotonic()
        torch.cuda.synchronize()
    print(f"idle load stopped; iterations={iterations}", flush=True)


if __name__ == "__main__":
    main()
