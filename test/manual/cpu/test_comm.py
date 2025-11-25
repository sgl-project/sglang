import copy
import multiprocessing
import os
import traceback
import unittest
from multiprocessing import Process

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.test_utils import CustomTestCase, find_available_port


def run_distributed_test(rank, world_size, master_port, output_writer, fn):
    try:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_SIZE"] = str(world_size)

        dist.init_process_group("gloo", rank=rank, world_size=world_size)
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


class TestComm(CustomTestCase):
    def _spawn_and_check(self, fn, world_size=2):
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
                ),
            )
            p.start()
            processes.append(p)

        for _ in range(world_size):
            self.assertTrue(output_reader.recv(), "Subprocess fail. Check logs above.")

        for p in processes:
            p.join()

    def test_all_reduce(self):
        self._spawn_and_check(all_reduce_fn)

    def test_all_gather(self):
        self._spawn_and_check(all_gather_fn)


if __name__ == "__main__":
    unittest.main()
