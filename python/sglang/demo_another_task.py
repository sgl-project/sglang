import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from setproctitle import setproctitle


def worker(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    setproctitle(f"demo_another_task::worker::{rank}")
    print(f"[GPU {rank}] started")

    # TODO use more memory
    big_tensors = [
        torch.empty(1024 * 1024 * 1024, dtype=torch.int8, device=device)
        for i in range(60)
    ]
    print(f"[GPU {rank}] allocated big tensors {[x.shape for x in big_tensors]=}")

    # num_iterations = 1000000000
    num_iterations = 3

    for iteration in range(num_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        num_repeat = 10000
        for _ in range(num_repeat):
            a = torch.randn(1024, 1024, device=device)
            b = torch.randn(1024, 1024, device=device)
            c = torch.matmul(a, b)

            x = torch.randn(1048576, device=device)
            y = torch.randn(1048576, device=device)
            z = x + y

            t = torch.randn(1024, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time = elapsed_time_ms / num_repeat

        print(f"[GPU {rank}] Iteration {iteration}: Avg time = {avg_time:.3f} ms")

    print(f"[GPU {rank}, {time.time()}] synchronize & barrier")
    torch.cuda.synchronize()
    torch.distributed.barrier(device_ids=[rank])

    print(f"[GPU {rank}, {time.time()}] del start")
    del big_tensors, a, b, c, x, y, z, t

    print(f"[GPU {rank}, {time.time()}] {torch.cuda.mem_get_info()=}")
    print(f"[GPU {rank}, {time.time()}] empty_cache start")
    torch.cuda.empty_cache()

    print(f"[GPU {rank}, {time.time()}] synchronize start")
    torch.cuda.synchronize()
    print(f"[GPU {rank}, {time.time()}] synchronize end")

    print(f"[GPU {rank}, {time.time()}] {torch.cuda.mem_get_info()=}")

    print(f"[GPU {rank}, {time.time()}] barrier")
    torch.distributed.barrier(device_ids=[rank])


def main():
    world_size = 8

    mp.set_start_method('spawn')
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
