import os
import threading
import time
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_memory_saver
import zmq
from setproctitle import setproctitle
from sglang.srt.utils import get_zmq_socket


def _create_recv_socket(rank: int):
    port = 50000 + rank
    context = zmq.Context(2)
    return get_zmq_socket(context, zmq.PULL, f"tcp://localhost:{port}", True)


def _try_recv(recv_socket):
    try:
        recv_req = recv_socket.recv_pyobj(zmq.NOBLOCK)
        print(f"{recv_req=}")
        return True
    except zmq.ZMQError:
        return False


def worker_background_thread(rank: int):
    memory_saver = torch_memory_saver.TorchMemorySaver(enable_use_mem_pool=False)
    recv_socket = _create_recv_socket(rank)
    print(f"worker_background_thread init")

    while True:
        if _try_recv(recv_socket):
            break
        else:
            time.sleep(0.001)

    print(f"[GPU {rank}, {time.time()}] pause start")
    memory_saver.pause()

    print(f"[GPU {rank}, {time.time()}] synchronize start")
    torch.cuda.synchronize()
    print(f"[GPU {rank}, {time.time()}] synchronize end")


def worker(args, rank, world_size):
    if args.stop_mode == 'background_thread_memory_saver':
        thread = threading.Thread(target=worker_background_thread, args=(rank,))
        thread.daemon = True
        thread.start()

    if args.stop_mode == 'torch_empty_cache':
        recv_socket = _create_recv_socket(rank)

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

    num_iterations = 1000000000
    # num_iterations = 3

    for iteration in range(num_iterations):
        if args.stop_mode == 'torch_empty_cache':
            if _try_recv(recv_socket):
                break

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        num_repeat = 10
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

    # print(f"[GPU {rank}, {time.time()}] synchronize & barrier")
    # torch.cuda.synchronize()
    # torch.distributed.barrier(device_ids=[rank])

    # if 1:
    #     print(f"[GPU {rank}, {time.time()}] pause start")
    #     memory_saver.pause()

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


def main(args):
    world_size = 8

    mp.set_start_method('spawn')
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(args, rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--stop-mode", type=str, choices=['background_thread_memory_saver', 'torch_empty_cache'])
    args = parser.parse_args()
    main(args)
