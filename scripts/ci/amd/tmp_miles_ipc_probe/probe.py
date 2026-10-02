"""TEMPORARY: probe the ROCm 10 / gfx942 corruption of the first CUDA-IPC copy into resumed
torch_memory_saver memory, and which receiver-side changes avoid it.

A sender shares a flattened bf16 bucket over CUDA IPC; the receiver keeps its weights in a TMS
region, pauses/resumes them, then copies the bucket in (a colocated weight sync in miniature).
MODE picks the receiver variant. Run under LD_PRELOAD of the TMS hook with 2 visible GPUs.
"""

import os
import time

import torch
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import reduce_tensor
from torch_memory_saver import torch_memory_saver

SHAPES = [(9728, 896), (896, 4864), (1152, 896), (896, 896)] * 24 + [(151936, 896)]
ITERS = 3
MODE = os.environ.get("MODE", "base")
# Modes whose sender ships one 4 MiB dummy tensor ahead of the buckets.
DUMMY_MODES = ("pre_ipc", "post_resume_dummy")


def make_srcs(it):
    g = torch.Generator(device="cuda").manual_seed(it)
    return [torch.randn(s, dtype=torch.bfloat16, device="cuda", generator=g) for s in SHAPES]


def report(tag, weights, srcs):
    bad, off = [], 0
    for i, (w, s) in enumerate(zip(weights, srcs)):
        ne = (w != s).flatten()
        n = int(ne.sum())
        if n:
            idx = ne.nonzero()
            lo, hi = int(idx[0]), int(idx[-1])
            dst = w.data_ptr() + lo * 2
            bad.append(
                f"#{i} {n}/{w.numel()} bad, flat bytes [{(off + lo) * 2:#x}, {(off + hi + 1) * 2:#x}),"
                f" dst {dst:#x} (dst % 2MiB = {dst % (2 << 20):#x})"
            )
        off += w.numel()
    print(f"[{MODE}] {tag}: {len(bad)}/{len(weights)} tensors corrupted", flush=True)
    for line in bad[:6]:
        print(f"    {line}", flush=True)


def copy_in(weights, flat):
    off = 0
    for w in weights:
        w.copy_(flat[off : off + w.numel()].view(w.shape))
        off += w.numel()
    torch.cuda.synchronize()


def receiver(q, done):
    torch.cuda.set_device(0)
    print(f"[{MODE}] receiver sees {torch.cuda.device_count()} GPUs", flush=True)
    with torch_memory_saver.region(tag="weights"):
        weights = [torch.empty(s, dtype=torch.bfloat16, device="cuda") for s in SHAPES]

    if MODE == "pre_ipc":
        rebuild, args = q.get()
        rebuild(*args).clone()
        torch.cuda.synchronize()
        done.put(None)
    elif MODE == "self_ipc":
        # Open an IPC handle of this process's own allocation, so no peer process is needed.
        t = torch.ones(1 << 20, device="cuda")
        rebuild, args = reduce_tensor(t)
        try:
            x = rebuild(*args)
            print(f"[{MODE}] self IPC open ok, sum={float(x.sum())}", flush=True)
        except Exception as e:
            print(f"[{MODE}] self IPC open failed: {e!r}", flush=True)

    for it in range(ITERS):
        torch_memory_saver.pause("weights")
        torch_memory_saver.resume("weights")
        if MODE == "touch_before":
            for w in weights:
                w.zero_()
            torch.cuda.synchronize()
        if MODE == "post_resume_dummy" and it == 0:
            rebuild, args = q.get()
            rebuild(*args).clone()
            torch.cuda.synchronize()
            done.put(None)

        rebuild, args = q.get()
        flat = rebuild(*args)
        if MODE == "sync_after_open":
            torch.cuda.synchronize()
            time.sleep(1)
        if MODE == "scratch_first":
            flat.clone()
            torch.cuda.synchronize()
        copy_in(weights, flat)
        # Only after the copy: GPU work between the IPC open and the copy can hide the bug.
        srcs = make_srcs(it)
        report(f"iter {it}", weights, srcs)
        if MODE == "copy_twice" and it == 0:
            copy_in(weights, flat)
            report(f"iter {it} second copy", weights, srcs)
        del flat
        done.put(None)


def main():
    assert torch.cuda.device_count() >= 2, "the bug needs >= 2 visible GPUs"
    ctx = mp.get_context("spawn")
    q, done = ctx.Queue(), ctx.Queue()
    env = os.environ.copy()
    if MODE == "recv_1gpu":
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
    p = ctx.Process(target=receiver, args=(q, done))
    p.start()
    os.environ.clear()
    os.environ.update(env)

    torch.cuda.set_device(0)
    if MODE in DUMMY_MODES:
        dummy = torch.ones(1 << 20, device="cuda")
        q.put(reduce_tensor(dummy))
        done.get()
    for it in range(ITERS):
        flat = torch.cat([s.flatten() for s in make_srcs(it)])
        torch.cuda.synchronize()
        q.put(reduce_tensor(flat))
        done.get()
        del flat
    p.join(timeout=120)
    assert p.exitcode == 0, f"receiver exit code {p.exitcode}"


if __name__ == "__main__":
    main()
