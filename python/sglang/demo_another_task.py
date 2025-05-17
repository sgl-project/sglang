import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    print(f"[GPU {rank}] started")

    iteration = 0
    while True:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        for _ in range(100):
            a = torch.randn(1024, 1024, device=device)
            b = torch.randn(1024, 1024, device=device)
            c = torch.matmul(a, b)

            x = torch.randn(1_000_000, device=device)
            y = torch.randn(1_000_000, device=device)
            z = x + y

            t = torch.randn(1024, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time = elapsed_time_ms / 100

        print(f"[GPU {rank}] Iteration {iteration}: Avg time = {avg_time:.3f} ms")
        iteration += 1


def main():
    world_size = 8
    if torch.cuda.device_count() < world_size:
        print("Need at least 8 GPUs")
        return

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
