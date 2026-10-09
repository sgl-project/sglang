"""Concurrency probe: is the custom_ops package's host state racy?

The single-threaded bare repro (npu_hc_pre_boundary_repro.py) never faults,
but serving interleaves SEVERAL ops from the same custom_ops .so
(npu_hc_pre / npu_hc_post / inplace_partial_rotary_mul / ...) on the same
and adjacent streams, with allocator churn between them. If the package
shares a tiling/workspace/scratch arena without full synchronization, one
op's launch can cross-write another's tiling -> garbage MTE addresses while
every plog scalar stays identical (the observed signature).

This probe hammers npu_hc_pre with the crash shapes from multiple threads
while sibling ops from the same .so run concurrently. A fault here pins the
bug inside the op package -- in-house, not CANN. A clean run does NOT fully
exonerate it (serving interleaving may differ), it only lowers its rank.

Run on the device:
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 test/manual/npu_hc_pre_concurrent_repro.py --seconds 120 \
    --so /usr/local/python3.12.13/lib/python3.12/site-packages/custom_ops/\
custom_ops_lib.cpython-312-x86_64-linux-gnu.so
"""

import argparse
import os
import random
import threading
import time

import torch
import torch_npu  # noqa: F401

T_CRASH = 2048
HIDDEN = 16384
HC_MULT = 4
HC_CHANNELS = 24


def ensure_custom_ops(so_arg):
    if hasattr(torch.ops.custom, "npu_hc_pre"):
        return
    for path in (so_arg, os.environ.get("CUSTOM_OPS_SO_PATH", "")):
        if path and os.path.isfile(path):
            torch.ops.load_library(path)
            if hasattr(torch.ops.custom, "npu_hc_pre"):
                print(f"loaded custom ops from {path}", flush=True)
                return
    raise SystemExit("custom ops not registered; pass --so <custom_ops_lib .so>")


def hc_pre_weights():
    return (
        torch.randn(HC_CHANNELS, HIDDEN, dtype=torch.float32, device="npu"),
        torch.ones(3, dtype=torch.float32, device="npu"),
        torch.zeros(HC_CHANNELS, dtype=torch.float32, device="npu"),
    )


def hammer_hc_pre(stop, counter, lock):
    hc_fn, hc_scale, hc_base = hc_pre_weights()
    while not stop.is_set():
        x = torch.randn(
            T_CRASH, HC_MULT, HIDDEN // HC_MULT, dtype=torch.bfloat16, device="npu"
        )
        y, post, comb = torch.ops.custom.npu_hc_pre(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult=HC_MULT,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        # A5 batched hc_post consumes hc_pre's outputs, as in serving.
        try:
            torch.ops.custom.npu_hc_post(
                y.unsqueeze(0), y.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
            )
        except Exception as exc:  # signature drift: report once, keep running
            with lock:
                counter["hc_post_err"] = counter.get("hc_post_err", 0) + 1
                if counter["hc_post_err"] == 1:
                    print(f"hc_post skipped: {exc}", flush=True)
        with lock:
            counter["hc_pre"] = counter.get("hc_pre", 0) + 1
            n = counter["hc_pre"]
        if n % 500 == 0:
            print(f"[hc_pre] {n} calls, y=0x{y.data_ptr():x}", flush=True)
        del x, y, post, comb


def hammer_rotary(stop, counter, lock):
    rope_dim = 128
    while not stop.is_set():
        q = torch.randn(T_CRASH, 8, 128, dtype=torch.bfloat16, device="npu")
        cos4 = torch.randn(T_CRASH, 1, 1, rope_dim, dtype=torch.float32, device="npu")
        sin4 = torch.randn(T_CRASH, 1, 1, rope_dim, dtype=torch.float32, device="npu")
        try:
            torch.ops.custom.inplace_partial_rotary_mul(
                q.unsqueeze(1),
                cos4,
                sin4,
                rotary_mode="interleave",
                partial_slice=[0, rope_dim],
            )
            with lock:
                counter["rotary"] = counter.get("rotary", 0) + 1
        except Exception as exc:
            with lock:
                counter["rotary_err"] = counter.get("rotary_err", 0) + 1
                if counter["rotary_err"] == 1:
                    print(f"rotary skipped: {exc}", flush=True)
            return
        del q, cos4, sin4


def churn_allocator(stop, counter, lock):
    while not stop.is_set():
        pad = torch.empty(
            random.choice([1, 2, 4, 8, 16, 32, 64]) * 1024 * 1024,
            dtype=torch.uint8,
            device="npu",
        )
        time.sleep(0.001)
        del pad
        with lock:
            counter["churn"] = counter.get("churn", 0) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument(
        "--threads", type=int, default=4, help="concurrent hc_pre hammer threads"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--so", default="")
    args = parser.parse_args()

    ensure_custom_ops(args.so)
    random.seed(args.seed)
    torch.npu.set_device(0)
    torch.manual_seed(args.seed)
    print(f"op schema: {torch.ops.custom.npu_hc_pre.default._schema}", flush=True)

    stop = threading.Event()
    counter, lock = {}, threading.Lock()
    threads = []
    for i in range(args.threads):
        threads.append(
            threading.Thread(
                target=hammer_hc_pre, args=(stop, counter, lock), daemon=True
            )
        )
    threads.append(
        threading.Thread(target=hammer_rotary, args=(stop, counter, lock), daemon=True)
    )
    threads.append(
        threading.Thread(
            target=churn_allocator, args=(stop, counter, lock), daemon=True
        )
    )
    for t in threads:
        t.start()
    try:
        time.sleep(args.seconds)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=30)
    print(f"done, no fault. counters: {counter}")


if __name__ == "__main__":
    main()
