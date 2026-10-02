"""ROCm 10 / gfx942: the first CUDA-IPC copy into resumed torch_memory_saver memory is corrupted.

A sender shares a flattened bf16 bucket over CUDA IPC; the receiver keeps its weights in a TMS region,
pauses/resumes them, then copies the bucket in (colocated weight sync in miniature).
Fails when the process sees >= 2 GPUs and opens its first IPC handle after the resume;
PRE_IPC=1 (open one IPC handle beforehand) makes it pass. Passes on the ROCm 7.0 image.

    LD_PRELOAD=<torch_memory_saver_hook_mode_preload .so> python repro.py      # needs 2 visible GPUs
"""

import os

import torch
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import reduce_tensor
from torch_memory_saver import torch_memory_saver

SHAPES = [(9728, 896), (896, 4864), (1152, 896), (896, 896)] * 24 + [(151936, 896)]
ITERS = 4


def make_srcs(it):
    g = torch.Generator(device="cuda").manual_seed(it)
    return [torch.randn(s, dtype=torch.bfloat16, device="cuda", generator=g) for s in SHAPES]


def receiver(q, done):
    torch.cuda.set_device(0)
    with torch_memory_saver.region(tag="weights"):
        weights = [torch.empty(s, dtype=torch.bfloat16, device="cuda") for s in SHAPES]
    if os.environ.get("PRE_IPC") == "1":
        rebuild, args = q.get()
        rebuild(*args).clone()
        done.put(None)
    for it in range(ITERS):
        torch_memory_saver.pause("weights")
        torch_memory_saver.resume("weights")
        rebuild, args = q.get()
        flat, off = rebuild(*args), 0
        for w in weights:
            w.copy_(flat[off : off + w.numel()].view(w.shape))
            off += w.numel()
        torch.cuda.synchronize()
        bad = sum(not torch.equal(w, s) for w, s in zip(weights, make_srcs(it)))
        print(f"iter {it}: {bad}/{len(weights)} tensors corrupted", flush=True)
        del flat
        done.put(None)


def main():
    assert torch.cuda.device_count() >= 2, "the bug needs >= 2 visible GPUs"
    ctx = mp.get_context("spawn")
    q, done = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=receiver, args=(q, done))
    p.start()
    torch.cuda.set_device(0)
    if os.environ.get("PRE_IPC") == "1":
        dummy = torch.ones(1 << 20, device="cuda")
        q.put(reduce_tensor(dummy))
        done.get()
    for it in range(ITERS):
        flat = torch.cat([s.flatten() for s in make_srcs(it)])
        torch.cuda.synchronize()
        q.put(reduce_tensor(flat))
        done.get()
        del flat
    p.join()


if __name__ == "__main__":
    main()

