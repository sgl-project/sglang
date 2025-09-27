import ctypes
import multiprocessing as mp
import random
import socket
import unittest
from typing import Any, List, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.managers.schedule_batch import global_server_args_dict
import sgl_kernel


def _run_allreduce_rmsnorm_worker(world_size, rank, distributed_init_port, test_configs):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
        device_id=rank,
    )
    group = dist.group.WORLD

    try:
        # Initialize global server args
        global_server_args_dict.update({
            "chunked_prefill_size": 2048,
            "enable_flashinfer_allreduce_fusion": True
        })

        for config in test_configs:
            batch_size, hidden_size, dtype, variance_epsilon = config
            print(f"Rank {rank}: Testing batch_size={batch_size}, hidden_size={hidden_size}, dtype={dtype}")

            # Create symmetric memory buffers
            CHUNK_SIZE = batch_size + 512
            staging_buffer = symm_mem.empty(
                (CHUNK_SIZE, hidden_size),
                device=device,
                dtype=dtype
            )

            try:
                symm_mem_hdl = symm_mem.rendezvous(staging_buffer, group)
            except Exception as e:
                print(f"Rank {rank}: Symmetric memory rendezvous failed: {e}")
                raise

            residual_buffer = torch.empty_like(staging_buffer)
            MAX_CTAS = 16

            # Use same seed to ensure data consistency across ranks
            torch.manual_seed(42)
            base_input = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
            base_residual = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
            base_weight = torch.randn(hidden_size, dtype=dtype, device=device)

            # Set up data according to distributed training scenario
            input_tensor = base_input + rank * 0.1  # Different input per rank
            residual_tensor = base_residual  # Same residual across all ranks
            weight = base_weight  # Same weight across all ranks

            # Save original residual for comparison
            original_residual = residual_tensor.clone()

            # Prepare fusion operator data
            staging_buffer[:batch_size].copy_(input_tensor)
            residual_buffer[:batch_size].copy_(residual_tensor)

            # =============== Fusion operator computation ===============
            tokens_per_rank = (batch_size + world_size - 1) // world_size
            start_idx = rank * tokens_per_rank
            end_idx = min((rank + 1) * tokens_per_rank, batch_size)
            offset = start_idx * hidden_size * input_tensor.element_size()

            try:
                # Call fusion operator
                sgl_kernel.fused_rs_ln_ag_cta(
                    staging_buffer[start_idx:end_idx],
                    residual_buffer[start_idx:end_idx], 
                    weight,
                    symm_mem_hdl.multicast_ptr + offset,
                    symm_mem_hdl.signal_pad_ptrs_dev,
                    rank,
                    world_size,
                    MAX_CTAS,
                    variance_epsilon
                )
            except Exception as e:
                print(f"Rank {rank}: Fusion operator failed: {e}")
                raise

            # Wait for all ranks to complete kernel execution
            torch.cuda.synchronize()
            dist.barrier(group=group)

            fused_output = staging_buffer[:batch_size].clone()
            fused_residual = residual_buffer[:batch_size].clone()

            # =============== Reference computation ===============
            # Collect input data from all ranks
            all_inputs = [torch.zeros_like(input_tensor) for _ in range(world_size)]
            dist.all_gather(all_inputs, input_tensor, group=group)

            # AllReduce all inputs
            total_input = torch.stack(all_inputs).sum(dim=0)  # [batch_size, hidden_size]

            # ReduceScatter + LayerNorm + AllGather semantics:
            # 1. Each rank processes a portion of tokens in the batch
            tokens_per_rank = (batch_size + world_size - 1) // world_size
            start_idx = rank * tokens_per_rank
            end_idx = min((rank + 1) * tokens_per_rank, batch_size)

            # 2. Each rank uses allreduced input and its own residual to compute this portion
            my_combined = total_input[start_idx:end_idx] + original_residual[start_idx:end_idx]

            # 3. LayerNorm on my slice
            orig_dtype = my_combined.dtype
            x = my_combined.float()
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            normalized = x * torch.rsqrt(variance + variance_epsilon)
            my_output_slice = (normalized * weight.float()).to(orig_dtype)
            my_residual_slice = my_combined

            # 4. AllGather results from all ranks
            all_output_slices = [torch.zeros(tokens_per_rank, hidden_size, dtype=dtype, device=device) 
                               for _ in range(world_size)]
            all_residual_slices = [torch.zeros(tokens_per_rank, hidden_size, dtype=dtype, device=device) 
                                 for _ in range(world_size)]

            # Handle case where last rank may have different slice size
            my_actual_slice_size = end_idx - start_idx
            padded_output_slice = torch.zeros(tokens_per_rank, hidden_size, dtype=dtype, device=device)
            padded_residual_slice = torch.zeros(tokens_per_rank, hidden_size, dtype=dtype, device=device)
            padded_output_slice[:my_actual_slice_size] = my_output_slice
            padded_residual_slice[:my_actual_slice_size] = my_residual_slice

            dist.all_gather(all_output_slices, padded_output_slice, group=group)
            dist.all_gather(all_residual_slices, padded_residual_slice, group=group)

            # Reconstruct complete output and residual
            ref_output_parts = []
            ref_residual_parts = []
            for i in range(world_size):
                start_i = i * tokens_per_rank
                end_i = min((i + 1) * tokens_per_rank, batch_size)
                actual_size_i = end_i - start_i
                ref_output_parts.append(all_output_slices[i][:actual_size_i])
                ref_residual_parts.append(all_residual_slices[i][:actual_size_i])

            ref_output = torch.cat(ref_output_parts, dim=0)
            ref_residual = torch.cat(ref_residual_parts, dim=0)

            # =============== Compare results ===============
            try:
                # Test output (should be same across all ranks)
                torch.testing.assert_close(
                    fused_output, ref_output, 
                    rtol=1e-2, atol=1e-3,
                    msg=f"Rank {rank}: Output mismatch for config {config}"
                )

                # Only test residual slice that each rank is responsible for
                my_fused_residual_slice = fused_residual[start_idx:end_idx]
                my_ref_residual_slice = ref_residual[start_idx:end_idx]

                # Handle empty slice case
                if my_fused_residual_slice.numel() > 0 and my_ref_residual_slice.numel() > 0:
                    torch.testing.assert_close(
                        my_fused_residual_slice, my_ref_residual_slice,
                        rtol=1e-2, atol=1e-3,
                        msg=f"Rank {rank}: My residual slice mismatch for config {config}"
                    )

                print(f"Rank {rank}: Test passed for config {config}")

            except Exception as e:
                print(f"Rank {rank}: Test failed for config {config}: {e}")

                # Only report differences within slice
                if start_idx < end_idx:  # Ensure slice is not empty
                    my_fused_output_slice = fused_output[start_idx:end_idx]
                    my_ref_output_slice = ref_output[start_idx:end_idx]
                    slice_output_diff = (my_fused_output_slice - my_ref_output_slice).abs().max().item()
                    print(f"Rank {rank}: My output slice max diff: {slice_output_diff}")

                    my_fused_residual_slice = fused_residual[start_idx:end_idx]  
                    my_ref_residual_slice = ref_residual[start_idx:end_idx]
                    slice_residual_diff = (my_fused_residual_slice - my_ref_residual_slice).abs().max().item()
                    print(f"Rank {rank}: My residual slice max diff: {slice_residual_diff}")

                raise

        print(f"Rank {rank}: All tests completed successfully")

    except Exception as e:
        print(f"Rank {rank}: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"Rank {rank}: Cleanup failed: {e}")


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (world_size, i, distributed_init_port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert (
            procs[i].exitcode == 0
        ), f"Process {i} failed with exit code {procs[i].exitcode}"


class TestAllReduceRMSNorm(unittest.TestCase):
    # Test configurations: (batch_size, hidden_size, dtype, variance_epsilon)
    test_configs = [
        (32, 512, torch.bfloat16, 1e-6),
        (64, 1024, torch.bfloat16, 1e-5),
        (128, 2048, torch.bfloat16, 1e-6),
    ]

    world_sizes = [2, 4]  # Test different GPU counts

    def test_allreduce_rmsnorm_correctness(self):
        for world_size in self.world_sizes:
            available_gpus = torch.cuda.device_count()
            if world_size > available_gpus:
                print(
                    f"Skipping world_size={world_size}, requires {world_size} GPUs, found {available_gpus}"
                )
                continue

            print(f"Running allreduce_rmsnorm test for world_size={world_size}")
            multi_process_parallel(
                world_size, 
                _run_allreduce_rmsnorm_worker, 
                target_args=(self.test_configs,)
            )
            print(f"AllReduce RMSNorm fusion test with world_size={world_size}: PASSED")

    def test_different_batch_sizes(self):
        """Test different batch sizes, including cases where batch_size is not divisible by world_size"""
        world_size = 2
        available_gpus = torch.cuda.device_count()
        if world_size > available_gpus:
            print(f"Skipping batch size test, requires {world_size} GPUs, found {available_gpus}")
            return

        # Test batch_sizes that are not divisible by world_size
        special_configs = [
            (33, 512, torch.bfloat16, 1e-6),   # 33 % 2 != 0
            (127, 1024, torch.bfloat16, 1e-6), # 127 % 2 != 0
            (255, 2048, torch.bfloat16, 1e-6), # 255 % 2 != 0
        ]

        print("Running allreduce_rmsnorm test for uneven batch sizes")
        multi_process_parallel(
            world_size,
            _run_allreduce_rmsnorm_worker,
            target_args=(special_configs,)
        )
        print("AllReduce RMSNorm fusion test with uneven batch sizes: PASSED")

    def test_small_batch_sizes(self):
        """Test small batch sizes"""
        world_size = 2
        available_gpus = torch.cuda.device_count()
        if world_size > available_gpus:
            print(f"Skipping small batch test, requires {world_size} GPUs, found {available_gpus}")
            return

        small_configs = [
            (1, 512, torch.bfloat16, 1e-6),    # Very small batch
            (2, 1024, torch.bfloat16, 1e-6),   # 1 token per rank
            (4, 2048, torch.bfloat16, 1e-6),   # 2 tokens per rank
        ]

        print("Running allreduce_rmsnorm test for small batch sizes")
        multi_process_parallel(
            world_size,
            _run_allreduce_rmsnorm_worker,
            target_args=(small_configs,)
        )
        print("AllReduce RMSNorm fusion test with small batch sizes: PASSED")


if __name__ == "__main__":
    unittest.main()