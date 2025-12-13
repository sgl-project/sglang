import os
import torch
import torch.distributed as dist
from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator

# Configuration
SIZES_TO_TEST = [
    (1024 * 1024, "4MB"),        # 4MB
    (1024 * 1024 * 16, "64MB"),  # 64MB
    (1024 * 1024 * 64, "256MB")  # 256MB
]

def init_distributed_env():
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["LOCAL_RANK"] = "0"
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def benchmark_op(func, args, iterations=20):
    # Warmup
    for _ in range(5):
        func(*args)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        func(*args)
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iterations

def run_all_reduce_test(coord_custom, coord_torch, device, num_elements, label):
    rank = dist.get_rank()
    
    # Setup data
    t_in = torch.ones(num_elements, device=device, dtype=torch.float32) * (rank + 1)
    
    # 1. Correctness Check
    # Important: Clone input because all_reduce might be in-place
    res_custom = coord_custom.all_reduce(t_in.clone())
    res_torch = coord_torch.all_reduce(t_in.clone())

    if not torch.allclose(res_custom, res_torch):
        raise RuntimeError(f"[All-Reduce] Mismatch for size {label}")

    # 2. Benchmark
    # We pass a fresh clone every time to simulate real usage
    def run_custom():
        coord_custom.all_reduce(t_in.clone())
        
    def run_torch():
        coord_torch.all_reduce(t_in.clone())

    time_custom = benchmark_op(run_custom, (), iterations=20)
    time_torch = benchmark_op(run_torch, (), iterations=20)

    if rank == 0:
        print(f"[All-Reduce | {label:<6}] Custom: {time_custom:.4f} ms | Torch: {time_torch:.4f} ms | Speedup: {time_torch/time_custom:.2f}x")

def run_all_gather_test(coord_custom, coord_torch, device, num_elements, label, separate):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    dim_size = 1024
    batch_size = num_elements // dim_size
    t_in = torch.ones((batch_size, dim_size), device=device, dtype=torch.float32) * (rank + 1)

    # 1. Correctness Check
    res_custom = coord_custom.all_gather(t_in, dim=0, separate_tensors=separate)
    res_torch = coord_torch.all_gather(t_in, dim=0, separate_tensors=separate)

    if separate:
        if not isinstance(res_custom, list) or not isinstance(res_torch, list):
             raise TypeError("[All-Gather Separate] Expected list return type")
        for i in range(world_size):
            if not torch.allclose(res_custom[i], res_torch[i]):
                raise RuntimeError(f"[All-Gather Separate] Mismatch at rank {i}")
    else:
        if not torch.allclose(res_custom, res_torch):
            raise RuntimeError(f"[All-Gather Fused] Mismatch")

    # 2. Benchmark
    time_custom = benchmark_op(coord_custom.all_gather, (t_in, 0, separate))
    time_torch = benchmark_op(coord_torch.all_gather, (t_in, 0, separate))
    
    mode_str = "Separate" if separate else "Fused   "
    if rank == 0:
        print(f"[All-Gather {mode_str} | {label:<6}] Custom: {time_custom:.4f} ms | Torch: {time_torch:.4f} ms | Speedup: {time_torch/time_custom:.2f}x")

def run_gather_test(coord_custom, coord_torch, device, num_elements, label):
    rank = dist.get_rank()
    dst_rank = 0

    dim_size = 1024
    batch_size = num_elements // dim_size
    t_in = torch.ones((batch_size, dim_size), device=device, dtype=torch.float32) * (rank + 1)

    # 1. Correctness Check
    res_custom = coord_custom.gather(t_in, dst=dst_rank, dim=0)
    res_torch = coord_torch.gather(t_in, dst=dst_rank, dim=0)

    if rank == dst_rank:
        if res_custom is None or res_torch is None:
             raise RuntimeError(f"[Gather] Returned None on destination rank")
        if not torch.allclose(res_custom, res_torch):
            raise RuntimeError(f"[Gather] Mismatch on rank {rank}")
    
    # 2. Benchmark
    time_custom = benchmark_op(coord_custom.gather, (t_in, dst_rank, 0))
    time_torch = benchmark_op(coord_torch.gather, (t_in, dst_rank, 0))

    if rank == 0:
         print(f"[Gather           | {label:<6}] Custom: {time_custom:.4f} ms | Torch: {time_torch:.4f} ms | Speedup: {time_torch/time_custom:.2f}x")

def main():
    local_rank = init_distributed_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Running communication tests on {world_size} GPUs.")
        print("-" * 100)

    all_ranks = list(range(world_size))
    
    # 1. Coordinator with Custom Ops enabled
    coord_custom = GroupCoordinator(
        group_ranks=[all_ranks],
        local_rank=local_rank,
        torch_distributed_backend="nccl",
        use_device_communicator=True,
        group_name="custom_group" # Give distinct name to avoid confusion
    )

    # 2. Coordinator with Custom Ops disabled (Pure PyTorch Fallback)
    coord_torch = GroupCoordinator(
        group_ranks=[all_ranks],
        local_rank=local_rank,
        torch_distributed_backend="nccl",
        use_device_communicator=False,
        group_name="torch_group"
    )

    for num_elements, label in SIZES_TO_TEST:
        run_all_reduce_test(coord_custom, coord_torch, device, num_elements, label)
        run_all_gather_test(coord_custom, coord_torch, device, num_elements, label, separate=False)
        run_all_gather_test(coord_custom, coord_torch, device, num_elements, label, separate=True)
        run_gather_test(coord_custom, coord_torch, device, num_elements, label)
        
        if rank == 0:
            print("-" * 100)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()