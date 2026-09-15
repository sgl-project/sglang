"""torchrun --nproc_per_node 4 ar_bench.py : microbench all-reduce variants at verify-step message sizes (TP4, bf16)."""
import os, time, torch, torch.distributed as dist
os.environ.setdefault("SGLANG_USE_AITER", "1")
rank = int(os.environ["RANK"]); ws = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=ws)
dev = torch.device("cuda", rank)
group = dist.group.WORLD
cpu_group = dist.new_group(backend="gloo")
sizes = {"bs1x4 (4x6144)": 4*6144, "bs2x4 (8x6144)": 8*6144, "bs4x4 (16x6144)": 16*6144, "bs8x4 (32x6144)": 32*6144, "bs24x4 (96x6144)": 96*6144}
def bench(fn, x, iters=200):
    for _ in range(20): fn(x)
    torch.cuda.synchronize(); dist.barrier()
    t = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    t.record()
    for _ in range(iters): fn(x)
    e.record(); torch.cuda.synchronize()
    return t.elapsed_time(e) / iters * 1000  # us
def check(fn, x, ref):
    y = fn(x)
    return (y.float() - ref.float()).abs().max().item(), ref.float().abs().max().item()
variants = {}
variants["NCCL"] = lambda x: (dist.all_reduce(x.clone()), )[0]
def nccl_out(x):
    y = x.clone(); dist.all_reduce(y); return y
variants["NCCL"] = nccl_out
# aiter custom AR (SGLang default on ROCm)
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce as AiterCAR
car = AiterCAR(group=cpu_group, device=dev)
variants["aiter CAR (auto: 2-stage >=160KB)"] = lambda x: car.all_reduce(x, use_new=True)

variants["aiter CAR old kernel"] = lambda x: car.all_reduce(x, use_new=False)
_w = None
def fused_norm(x):
    global _w
    n = x.numel() // 6144
    if _w is None or _w[0].shape[0] != n:
        _w = (torch.zeros(n, 6144, device=dev, dtype=torch.bfloat16), torch.ones(6144, device=dev, dtype=torch.bfloat16))
    r = car.fused_allreduce_rmsnorm(x.view(n, 6144), _w[0], _w[1], 1e-6)
    return r[0] if isinstance(r, (tuple, list)) else r
def fused_ar_rms(use_1stage):
    def f(x):
        global _w
        n = x.numel() // 6144
        if _w is None or _w[0].shape[0] != n:
            _w = (torch.zeros(n, 6144, device=dev, dtype=torch.bfloat16), torch.ones(6144, device=dev, dtype=torch.bfloat16))
        r = car.custom_fused_ar_rms(x.view(n, 6144), _w[0], _w[1], 1e-6, use_1stage=use_1stage)
        return r[0] if isinstance(r, (tuple, list)) else r
    return f
variants["aiter fused AR+RMSNorm 2-stage"] = fused_ar_rms(False)
variants["aiter fused AR+RMSNorm 1-stage"] = fused_ar_rms(True)
# sglang CAR: default (1-stage <512KB else 2-stage) and deterministic 1-stage
from sglang.srt.distributed.device_communicators.custom_all_reduce import CustomAllreduce as SglCAR


os.environ["SGLANG_USE_1STAGE_ALLREDUCE"] = "1"
try:
    from sglang.srt import environ as _e
    sgl1 = SglCAR(group=cpu_group, device=dev)
    sgl1.use_amd_deterministic_impl = True
    variants["sglang CAR 1-stage (deterministic)"] = lambda x: sgl1._all_reduce_impl(x, registered=False)
except Exception as ex:
    if rank == 0: print("1stage variant unavailable:", ex)
# quick reduce at each quant level
from sglang.srt.distributed.device_communicators.quick_all_reduce import QuickAllReduce
qrs = {}
for lvl in []:
    os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = lvl
    try:
        qrs[lvl] = QuickAllReduce(group=cpu_group, device=dev)
        variants[f"quick-reduce {lvl}"] = (lambda q: (lambda x: q.quick_all_reduce(x)))(qrs[lvl])
    except Exception as ex:
        if rank == 0: print("QR", lvl, "unavailable:", ex)
torch.manual_seed(rank)
for name, n in sizes.items():
    x = (torch.randn(n, device=dev, dtype=torch.bfloat16) * 3)
    ref = x.clone(); dist.all_reduce(ref)
    if rank == 0: print(f"\n== {name}: {n*2/1024:.0f} KB bf16")
    for vn, fn in variants.items():
        try:
            err, mx = (float('nan'), 0.0) if 'fused' in vn else check(fn, x, ref)
            us = bench(fn, x)
            if rank == 0: print(f"  {vn:44s} {us:8.1f} us   max|err| {err:.3g} (ref max {mx:.1f})")
        except Exception as ex:
            if rank == 0: print(f"  {vn:44s} FAILED: {str(ex)[:120]}")
        dist.barrier()
dist.destroy_process_group()
