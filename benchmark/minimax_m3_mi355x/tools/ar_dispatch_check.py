"""torchrun --nproc_per_node 4: exercise GroupCoordinator.all_reduce with the small-message 1-stage route on/off."""
import os, torch, torch.distributed as dist
from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel, get_tp_group
rank=int(os.environ["RANK"]); ws=int(os.environ["WORLD_SIZE"]); torch.cuda.set_device(rank)
init_distributed_environment(world_size=ws, rank=rank, local_rank=rank, distributed_init_method="env://", backend="nccl")
initialize_model_parallel(tensor_model_parallel_size=ws)
g=get_tp_group(); dev=torch.device("cuda",rank)
print0=lambda *a: print(*a) if rank==0 else None
print0("ca_comm:", type(g.ca_comm).__name__, "small_ca_comm:", type(g.small_ca_comm).__name__ if g.small_ca_comm else None, "max_bytes:", g._small_ca_max_bytes)
def bench(fn, x, iters=200):
    for _ in range(20): fn(x)
    torch.cuda.synchronize(); dist.barrier()
    t=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); t.record()
    for _ in range(iters): fn(x)
    e.record(); torch.cuda.synchronize(); return t.elapsed_time(e)/iters*1000
torch.manual_seed(rank)
for rows in (4, 8, 16, 32, 96, 512):
    x=torch.randn(rows,6144,device=dev,dtype=torch.bfloat16)*3
    ref=x.clone(); dist.all_reduce(ref)
    method=g._resolve_outplace_all_reduce_method(x)
    out=g.all_reduce(x.clone())
    err=(out.float()-ref.float()).abs().max().item()
    us=bench(lambda t: g.all_reduce(t), x)
    print0(f"rows={rows:4d} {rows*6144*2/1024:7.0f} KB method={method:9s} {us:6.1f} us  max|err|={err:.3g}")
dist.barrier()
