import gc
import multiprocessing as mp
import os
import time
import unittest

import torch

import sglang as sgl
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.utils import MultiprocessingSerializer
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    """Preprocess tensor for update weights - handles DTensor conversion.
    For test purposes, we just return the tensor as-is
    In FSDP, we can call tensor.full_tensor() to gather the partial tensors into a full tensor
    """
    return tensor


def worker_process(
    rank, world_size, tp_size, port, tensor_queue, barrier, batch_update=False
):
    try:
        torch.cuda.set_device(rank)
        new_tensor = torch.full((16384, 2048), 1.5, device=f"cuda:{rank}")

        param_names = [f"model.layers.{i}.mlp.up_proj.weight" for i in range(6, 16)]

        if rank == 0:
            # Process 0: Create engine and collect serialized tensors
            engine = sgl.Engine(
                model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                tp_size=tp_size,
                mem_fraction_static=0.2,
                disable_cuda_graph=True,
            )

            # Check initial param values
            _check_param(
                engine, param_names[0], [0.0087, -0.0214, -0.0004, 0.0039, 0.0110]
            )

            memory_before = torch.cuda.memory_allocated()

            if batch_update:
                # Batch update: collect all tensors first, then update once
                print("=== Testing BATCH UPDATE (all 10 layers at once) ===")
                all_named_tensors = []
                total_gather_time = 0

                for param_name in param_names:
                    gather_start = time.perf_counter()
                    gathered_tensors = []

                    # Add local tensor
                    local_serialized = MultiprocessingSerializer.serialize(
                        _preprocess_tensor_for_update_weights(new_tensor)
                    )
                    gathered_tensors.append(local_serialized)

                    # Collect from other ranks via queue
                    for _ in range(1, world_size):
                        remote_serialized = tensor_queue.get()
                        gathered_tensors.append(remote_serialized)

                    all_named_tensors.append(
                        (param_name, LocalSerializedTensor(values=gathered_tensors))
                    )
                    total_gather_time += time.perf_counter() - gather_start

                # Single batch update
                update_start = time.perf_counter()
                engine.update_weights_from_tensor(
                    named_tensors=all_named_tensors, flush_cache=True
                )
                update_time = time.perf_counter() - update_start

                print(
                    f"Batch update - Gather time: {total_gather_time:.03f}s, Update time: {update_time:.03f}s, Total: {total_gather_time + update_time:.03f}s"
                )

            else:
                # Sequential update: update one by one
                print("=== Testing SEQUENTIAL UPDATE (one by one) ===")
                total_update_time = 0
                total_gather_time = 0

                for param_name in param_names:
                    gather_start = time.perf_counter()
                    gathered_tensors = []

                    # Add local tensor
                    local_serialized = MultiprocessingSerializer.serialize(
                        _preprocess_tensor_for_update_weights(new_tensor)
                    )
                    gathered_tensors.append(local_serialized)

                    # Collect from other ranks via queue
                    for _ in range(1, world_size):
                        remote_serialized = tensor_queue.get()
                        gathered_tensors.append(remote_serialized)

                    gather_time = time.perf_counter() - gather_start
                    total_gather_time += gather_time

                    # Update weights using gathered serialized tensors
                    update_start = time.perf_counter()
                    engine.update_weights_from_tensor(
                        named_tensors=[
                            (param_name, LocalSerializedTensor(values=gathered_tensors))
                        ],
                        flush_cache=(param_name == param_names[-1]),
                    )
                    update_time = time.perf_counter() - update_start
                    total_update_time += update_time

                    print(
                        f"{param_name} - Gather: {gather_time:.03f}s, Update: {update_time:.03f}s"
                    )

                print(
                    f"Sequential update - Total gather time: {total_gather_time:.03f}s, Total update time: {total_update_time:.03f}s, Total: {total_gather_time + total_update_time:.03f}s"
                )

            # Check updated param values
            for param_name in param_names[:3]:
                _check_param(engine, param_name, [1.5] * 5)

        else:
            # Non-rank-0 processes: Send serialized tensors via queue
            for param_name in param_names:
                serialized_tensor = MultiprocessingSerializer.serialize(
                    _preprocess_tensor_for_update_weights(new_tensor)
                )
                tensor_queue.put(serialized_tensor)

        # Critical: Wait for all processes to complete before exiting
        # This ensures tensor IPC handles remain valid until SGLang finishes
        print(f"Rank {rank} waiting at barrier...")
        barrier.wait()
        print(f"Rank {rank} passed barrier")

        if rank == 0:
            engine.shutdown()

            # Memory cleanup and check
            del new_tensor
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            memory_after = torch.cuda.memory_allocated()
            assert (
                memory_after <= memory_before + 1024
            ), f"Memory leak detected: {memory_after - memory_before} bytes"

    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise


def test_update_weights_from_tensor_spmd(tp_size, world_size, batch_update=False):
    assert torch.cuda.device_count() >= max(
        tp_size, world_size
    ), f"At least {max(tp_size, world_size)} GPUs are required"
    torch.cuda.empty_cache()

    # Create shared resources
    tensor_queue = mp.Queue()
    barrier = mp.Barrier(world_size)

    # Start worker processes
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=worker_process,
            args=(rank, world_size, tp_size, 0, tensor_queue, barrier, batch_update),
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()
        assert p.exitcode == 0, f"Process failed with exit code {p.exitcode}"


class TestUpdateWeightsFromTensor(CustomTestCase):
    def test_update_weights_from_tensor_spmd_sequential(self):
        """
        Test SPMD version with sequential updates (one by one).
        """
        TP_SIZE = 2
        test_update_weights_from_tensor_spmd(
            TP_SIZE, world_size=TP_SIZE, batch_update=False
        )

    def test_update_weights_from_tensor_spmd_batch(self):
        """
        Test SPMD version with batch update (all layers at once).
        """
        TP_SIZE = 2
        test_update_weights_from_tensor_spmd(
            TP_SIZE, world_size=TP_SIZE, batch_update=True
        )


def _check_param(engine, param_name, expect_values):
    actual_values = torch.tensor(engine.get_weights_by_name(param_name))[0, :5]
    assert torch.allclose(
        actual_values, torch.tensor(expect_values), atol=0.002
    ), f"{actual_values=}"


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    unittest.main()
