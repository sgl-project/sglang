import copy
import multiprocessing
import os
import time
import traceback
import unittest
import uuid
from multiprocessing import Process
from pathlib import Path
from unittest.mock import patch

import sgl_kernel  # noqa: F401
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.distributed.utils import reduce_scatter_single
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, find_available_port

register_cpu_ci(est_time=43, suite="extra-test-cpu-intel")


def run_distributed_test(
    rank, world_size, master_port, output_writer, fn, init_legacy_shm=True
):
    try:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_SIZE"] = str(world_size)

        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        if init_legacy_shm:
            torch.ops.sgl_kernel.initialize(world_size, rank)
        fn(rank, world_size)
        execution_ok = True
    except Exception as e:
        print(f"subprocess[{rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    output_writer.send(execution_ok)
    output_writer.close()

    if dist.is_initialized():
        dist.destroy_process_group()


def run_delayed_shm_test(
    rank, master_port, group_name, delayed, proceed, output_writer
):
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = master_port
        if rank == 1:
            delayed.set()
            if not proceed.wait(30):
                raise TimeoutError("rank 1 was not released")
        handle = torch.ops.sgl_kernel.shm_group_initialize(group_name, 2, rank)
        path = Path(
            f"/dev/shm/deepspeed_allreduce_buffer_{os.getuid()}_localhost_"
            f"{master_port}_{group_name}_{rank}"
        )
        if path.exists():
            raise AssertionError(f"shared memory still named: {path}")
        tensor = torch.tensor([float(rank + 1)])
        torch.ops.sgl_kernel.shm_allreduce(tensor, dist.ReduceOp.SUM, handle)
        torch.testing.assert_close(tensor, torch.tensor([3.0]))
        output_writer.send(None)
    except Exception:
        output_writer.send(traceback.format_exc())
    finally:
        output_writer.close()


def all_reduce_fn(rank, world_size):
    op = dist.ReduceOp.SUM
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        tensor = torch.randn(2, 10, dtype=dtype)
        tensor_shm = copy.deepcopy(tensor)

        dist.all_reduce(tensor, op=op)
        torch.ops.sgl_kernel.shm_allreduce(tensor_shm, op)

        torch.testing.assert_close(tensor, tensor_shm)


def all_gather_fn(rank, world_size):
    dim = -1

    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        tensor = torch.randn(2, 10, dtype=dtype)

        if dim < 0:
            # Convert negative dim to positive.
            dim += tensor.dim()

        input_size = tensor.size()
        output_size = (input_size[0] * world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=tensor.dtype, device=tensor.device
        )
        dist.all_gather_into_tensor(output_tensor, tensor)
        output_tensor = output_tensor.reshape((world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
        )

        output_shm = torch.ops.sgl_kernel.shm_allgather(tensor, dim)

        torch.testing.assert_close(output_tensor, output_shm)


def all_gather_into_tensor_fn(rank, world_size):
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        tensor = torch.randn(2, 10, dtype=dtype)

        input_size = tensor.size()
        output_size = (input_size[0] * world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=tensor.dtype, device=tensor.device
        )
        output_shm = torch.empty(output_size, dtype=tensor.dtype, device=tensor.device)
        dist.all_gather_into_tensor(output_tensor, tensor)

        torch.ops.sgl_kernel.shm_allgather_into_tensor(output_shm, tensor)

        torch.testing.assert_close(output_tensor, output_shm)


def reduce_scatter_tensor_fn(rank, world_size):
    op = dist.ReduceOp.SUM

    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        # (2560, 4096) makes the input exceed the 32 MiB shm buffer, ending in a partial chunk.
        for N, D in [(4, 10), (2560, 4096)]:
            input_size = (world_size * N, D)
            tensor = torch.randn(input_size, dtype=dtype)

            output_size = (N, D)
            output_tensor = torch.empty(output_size, dtype=dtype)
            output_shm = torch.empty_like(output_tensor)

            reduce_scatter_single(output_tensor, tensor, op=op)

            torch.ops.sgl_kernel.shm_reduce_scatter_tensor(output_shm, tensor, op)

            torch.testing.assert_close(output_tensor, output_shm)


def init_group(rank):
    group_ranks_list = [[0, 2], [1, 3]]

    groups = [dist.new_group(ranks=ranks, backend="gloo") for ranks in group_ranks_list]

    group_idx = rank % 2
    group = groups[group_idx]
    group_ranks = group_ranks_list[group_idx]

    group_size = dist.get_world_size(group)
    group_rank = dist.get_rank(group)

    group_name = f"test_group_{os.environ['MASTER_PORT']}_" + "_".join(
        map(str, group_ranks)
    )

    handle = torch.ops.sgl_kernel.shm_group_initialize(
        group_name, group_size, group_rank
    )

    return group, handle


def group_all_reduce_fn(rank, world_size):
    group, handle = init_group(rank)

    op = dist.ReduceOp.SUM

    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        for size in [(2, 10), (600000,)]:
            tensor = torch.randn(size, dtype=dtype)
            tensor_shm = copy.deepcopy(tensor)

            dist.all_reduce(tensor, op=op, group=group)

            torch.ops.sgl_kernel.shm_allreduce(tensor_shm, op, handle)

            torch.testing.assert_close(tensor, tensor_shm)


def group_all_gather_fn(rank, world_size):
    group, handle = init_group(rank)

    group_size = dist.get_world_size(group)

    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        tensor = torch.randn(2, 10, dtype=dtype)

        input_size = tensor.size()
        output_size = (input_size[0] * group_size,) + input_size[1:]

        output_tensor = torch.empty(
            output_size,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        output_shm = torch.empty(
            output_size,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        dist.all_gather_into_tensor(output_tensor, tensor, group=group)

        torch.ops.sgl_kernel.shm_allgather_into_tensor(output_shm, tensor, handle)

        torch.testing.assert_close(output_tensor, output_shm)


def group_all_to_all_fn(rank, world_size):
    group, handle = init_group(rank)

    group_size = dist.get_world_size(group)

    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        tensor = torch.randn(group_size * 2, 10, dtype=dtype)
        output_tensor = torch.empty_like(tensor)
        output_shm = torch.empty_like(tensor)

        dist.all_to_all_single(output_tensor, tensor, group=group)

        torch.ops.sgl_kernel.shm_alltoall(output_shm, tensor, handle)

        torch.testing.assert_close(output_tensor, output_shm)


class TestComm(CustomTestCase):
    def test_group_shared_memory_unlinked(self):
        """An initialized group must not leave named shared memory behind."""
        group_name = f"test_{uuid.uuid4().hex}"
        master_port = str(find_available_port(23456))
        path = Path(
            f"/dev/shm/deepspeed_allreduce_buffer_{os.getuid()}_localhost_"
            f"{master_port}_{group_name}_0"
        )
        try:
            with patch.dict(
                os.environ, MASTER_ADDR="localhost", MASTER_PORT=master_port
            ):
                handle = torch.ops.sgl_kernel.shm_group_initialize(group_name, 1, 0)
            self.assertFalse(path.exists())
            tensor = torch.ones(1)
            torch.ops.sgl_kernel.shm_allreduce(tensor, dist.ReduceOp.SUM, handle)
            torch.testing.assert_close(tensor, torch.ones(1))
        finally:
            path.unlink(missing_ok=True)

    def test_group_shared_memory_existing_name(self):
        group_name = f"test_{uuid.uuid4().hex}"
        master_port = str(find_available_port(23456))
        path = Path(
            f"/dev/shm/deepspeed_allreduce_buffer_{os.getuid()}_localhost_"
            f"{master_port}_{group_name}_0"
        )
        try:
            with path.open("xb") as existing:
                existing.write(b"existing")
            with patch.dict(
                os.environ, MASTER_ADDR="localhost", MASTER_PORT=master_port
            ):
                with self.assertRaisesRegex(RuntimeError, "shm_open failed"):
                    torch.ops.sgl_kernel.shm_group_initialize(group_name, 1, 0)
            self.assertEqual(path.read_bytes(), b"existing")
        finally:
            path.unlink(missing_ok=True)

    def test_group_shared_memory_delayed_rank(self):
        """A slow rank must map each segment before its name is unlinked."""
        mp.set_start_method("spawn", force=True)
        group_name = f"test_{uuid.uuid4().hex}"
        master_port = str(find_available_port(23456))
        paths = [
            Path(
                f"/dev/shm/deepspeed_allreduce_buffer_{os.getuid()}_localhost_"
                f"{master_port}_{group_name}_{rank}"
            )
            for rank in range(2)
        ]
        delayed = mp.Event()
        proceed = mp.Event()
        output_reader, output_writer = mp.Pipe(duplex=False)
        processes = [
            Process(
                target=run_delayed_shm_test,
                args=(rank, master_port, group_name, delayed, proceed, output_writer),
            )
            for rank in range(2)
        ]
        try:
            for process in processes:
                process.start()
            self.assertTrue(delayed.wait(30), "rank 1 did not reach the delay")
            deadline = time.monotonic() + 30
            while not paths[0].exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(paths[0].exists(), "rank 0 did not create its segment")
            proceed.set()
            for _ in processes:
                self.assertTrue(output_reader.poll(60), "rank did not complete")
                self.assertIsNone(output_reader.recv())
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
        finally:
            proceed.set()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(10)
            output_reader.close()
            output_writer.close()
            for path in paths:
                path.unlink(missing_ok=True)

    def _spawn_and_check(self, fn, world_size=2, init_legacy_shm=True):
        mp.set_start_method("spawn", force=True)
        master_port = find_available_port(23456)

        processes = []
        output_reader, output_writer = multiprocessing.Pipe(duplex=False)

        for rank in range(world_size):
            p = Process(
                target=run_distributed_test,
                kwargs=dict(
                    rank=rank,
                    world_size=world_size,
                    master_port=master_port,
                    output_writer=output_writer,
                    fn=fn,
                    init_legacy_shm=init_legacy_shm,
                ),
            )
            p.start()
            processes.append(p)

        for _ in range(world_size):
            self.assertTrue(output_reader.recv(), "Subprocess fail. Check logs above.")

        for p in processes:
            p.join()

    def test_all_reduce(self):
        self._spawn_and_check(all_reduce_fn, init_legacy_shm=True)

    def test_all_gather(self):
        self._spawn_and_check(all_gather_fn, init_legacy_shm=True)

    def test_all_gather_into_tensor(self):
        self._spawn_and_check(all_gather_into_tensor_fn, init_legacy_shm=True)

    def test_reduce_scatter_tensor(self):
        self._spawn_and_check(reduce_scatter_tensor_fn, init_legacy_shm=True)

    def test_group_all_reduce(self):
        self._spawn_and_check(group_all_reduce_fn, world_size=4, init_legacy_shm=False)

    def test_group_all_gather(self):
        self._spawn_and_check(group_all_gather_fn, world_size=4, init_legacy_shm=False)

    def test_group_all_to_all(self):
        self._spawn_and_check(group_all_to_all_fn, world_size=4, init_legacy_shm=False)


if __name__ == "__main__":
    unittest.main()
