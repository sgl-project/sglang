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

def run_all_gather_test(coord_custom, coord_torch, device, num_elements, label, separate, dim_to_test=0):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    dim_size = 1024
    batch_size = num_elements // dim_size
    t_in = torch.ones((batch_size, dim_size), device=device, dtype=torch.float32) * (rank + 1)

    # 1. Correctness Check
    res_custom = coord_custom.all_gather(t_in, dim=dim_to_test, separate_tensors=separate)
    res_torch = coord_torch.all_gather(t_in, dim=dim_to_test, separate_tensors=separate)

    if separate:
        if not isinstance(res_custom, list) or not isinstance(res_torch, list):
             raise TypeError(f"[All-Gather Separate dim={dim_to_test}] Expected list return type")
        for i in range(world_size):
            if not torch.allclose(res_custom[i], res_torch[i]):
                # 打印详细的调试信息
                print(f"\n[DEBUG] Mismatch details for dim={dim_to_test}, rank {i}:")
                print(f"  Input shape: {t_in.shape}")
                print(f"  Custom result[{i}] shape: {res_custom[i].shape}")
                print(f"  Torch result[{i}] shape: {res_torch[i].shape}")
                print(f"  Custom result[{i}] mean: {res_custom[i].mean():.6f}")
                print(f"  Torch result[{i}] mean: {res_torch[i].mean():.6f}")
                print(f"  Max absolute diff: {torch.abs(res_custom[i] - res_torch[i]).max():.6e}")
                print(f"  Custom result[{i}] sample: {res_custom[i].flatten()[:10]}")
                print(f"  Torch result[{i}] sample: {res_torch[i].flatten()[:10]}")
                raise RuntimeError(f"[All-Gather Separate dim={dim_to_test}] Mismatch at rank {i}")
    else:
        if not torch.allclose(res_custom, res_torch):
            raise RuntimeError(f"[All-Gather Fused dim={dim_to_test}] Mismatch")

    # 2. Benchmark
    time_custom = benchmark_op(coord_custom.all_gather, (t_in, dim_to_test, separate))
    time_torch = benchmark_op(coord_torch.all_gather, (t_in, dim_to_test, separate))
    
    mode_str = "Separate" if separate else "Fused   "
    if rank == 0:
        print(f"[All-Gather {mode_str} dim={dim_to_test} | {label:<6}] Custom: {time_custom:.4f} ms | Torch: {time_torch:.4f} ms | Speedup: {time_torch/time_custom:.2f}x")

def run_gather_test(coord_custom, coord_torch, device, num_elements, label, dim_to_test=0):
    rank = dist.get_rank()
    dst_rank = 0

    dim_size = 1024
    batch_size = num_elements // dim_size
    t_in = torch.ones((batch_size, dim_size), device=device, dtype=torch.float32) * (rank + 1)

    # 1. Correctness Check
    res_custom = coord_custom.gather(t_in, dst=dst_rank, dim=dim_to_test)
    res_torch = coord_torch.gather(t_in, dst=dst_rank, dim=dim_to_test)

    if rank == dst_rank:
        if res_custom is None or res_torch is None:
             raise RuntimeError(f"[Gather dim={dim_to_test}] Returned None on destination rank")
        if not torch.allclose(res_custom, res_torch):
            raise RuntimeError(f"[Gather dim={dim_to_test}] Mismatch on rank {rank}")
    
    # 2. Benchmark
    time_custom = benchmark_op(coord_custom.gather, (t_in, dst_rank, dim_to_test))
    time_torch = benchmark_op(coord_torch.gather, (t_in, dst_rank, dim_to_test))

    if rank == 0:
         print(f"[Gather dim={dim_to_test}      | {label:<6}] Custom: {time_custom:.4f} ms | Torch: {time_torch:.4f} ms | Speedup: {time_torch/time_custom:.2f}x")

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

    # Test different dimensions
    dims_to_test = [0, 1]
    
    for num_elements, label in SIZES_TO_TEST:
        # All-Reduce doesn't have a dim parameter, so test it once
        run_all_reduce_test(coord_custom, coord_torch, device, num_elements, label)
        
        # Test all_gather and gather with different dimensions
        for dim in dims_to_test:
            run_all_gather_test(coord_custom, coord_torch, device, num_elements, label, separate=False, dim_to_test=dim)
            run_all_gather_test(coord_custom, coord_torch, device, num_elements, label, separate=True, dim_to_test=dim)
            run_gather_test(coord_custom, coord_torch, device, num_elements, label, dim_to_test=dim)
        
        if rank == 0:
            print("-" * 120)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()