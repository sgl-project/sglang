"""8-GPU correctness coverage for FlyDSL per-call recv_cap."""

import os

import mori.shmem as ms
import torch
import torch.distributed as dist

from sglang.kernels.third_party.flydsl_a2a import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)


def _make_inputs(rank, world_size, cur, hidden, epr, topk, device):
    torch.manual_seed(1000 + rank + cur)
    inp = torch.randn(cur, hidden, dtype=torch.bfloat16, device=device)
    weights = torch.rand(cur, topk, dtype=torch.float32, device=device)
    indices = torch.empty(cur, topk, dtype=torch.int32, device=device)
    for token in range(cur):
        for slot in range(topk):
            pe = (rank + token + slot) % world_size
            indices[token, slot] = pe * epr + ((token * topk + slot) % epr)
    return inp, weights, indices


def _run_once(op, inp, weights, indices, cur, recv_cap):
    ret = op.dispatch(inp, weights, None, indices, recv_cap=recv_cap)
    torch.cuda.synchronize()
    total_recv = int(ret[4].item())
    assert ret[0].shape[0] == recv_cap
    assert total_recv <= recv_cap
    combined, _ = op.combine(ret[0], None, ret[3], cur_tok=cur, recv_cap=recv_cap)
    torch.cuda.synchronize()
    return total_recv, combined[:cur].clone()


def _worker(rank, world_size, port):
    os.environ.update(
        LOCAL_RANK=str(rank),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch._C._distributed_c10d._register_process_group(
        "flydsl_dynamic_recv_cap", dist.group.WORLD
    )
    ms.shmem_torch_process_group_init("flydsl_dynamic_recv_cap")

    device = torch.device("cuda", rank)
    hidden, epr, topk, physical_cap = 7168, 48, 6, world_size * 4096
    op = FlyDSLDispatchCombineIntraNodeOp(
        FlyDSLDispatchCombineConfig(
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden,
            max_num_inp_token_per_rank=4096,
            num_experts_per_rank=epr,
            num_experts_per_token=topk,
            data_type=torch.bfloat16,
            max_token_type_size=torch.bfloat16.itemsize,
        )
    )
    ms.shmem_barrier_all()

    for cur, dynamic_cap in ((32, 256), (5 if rank == 0 else 0, 32)):
        inp, weights, indices = _make_inputs(
            rank, world_size, cur, hidden, epr, topk, device
        )
        full_count, full_out = _run_once(op, inp, weights, indices, cur, physical_cap)
        dyn_count, dyn_out = _run_once(op, inp, weights, indices, cur, dynamic_cap)
        assert dyn_count == full_count
        if cur:
            torch.testing.assert_close(dyn_out, full_out, rtol=0, atol=0)

    ms.shmem_barrier_all()
    try:
        ms.shmem_finalize()
    except Exception:
        pass
    dist.destroy_process_group()


def test_flydsl_dynamic_recv_cap():
    torch.multiprocessing.spawn(_worker, args=(8, 29831), nprocs=8, join=True)


if __name__ == "__main__":
    test_flydsl_dynamic_recv_cap()
