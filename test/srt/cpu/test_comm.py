import copy
import itertools
import multiprocessing
import os
import random
import traceback
import unittest
from multiprocessing import Process

import sgl_kernel
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import precision

from sglang.test.test_utils import CustomTestCase, find_available_port


def all_reduce_test(rank, world_size, master_port, output_writer):
    try:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_SIZE"] = str(world_size)

        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.ops.sgl_kernel.initialize(world_size, rank)

        op = dist.ReduceOp.SUM

        for dtype in [torch.float32, torch.bfloat16, torch.bfloat16]:
            tensor = torch.randn(2, 10, dtype=dtype)
            tensor_shm = copy.deepcopy(tensor)

            dist.all_reduce(tensor, op=op)

            torch.ops.sgl_kernel.shm_allreduce(tensor_shm, op)
            torch.testing.assert_close(tensor, tensor_shm)

        execution_ok = True
    except Exception as e:
        print(f"subprocess[{rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    output_writer.send(execution_ok)
    output_writer.close()

    if dist.is_initialized():
        dist.destroy_process_group()


class TestComm(CustomTestCase):
    def test_all_reduce(self):
        # TODO: test different input dtypes
        mp.set_start_method("spawn", force=True)

        world_size = 2
        master_port = find_available_port(23456)
        processes = []
        output_reader, output_writer = multiprocessing.Pipe(duplex=False)

        for rank in range(world_size):
            p = Process(
                target=all_reduce_test,
                kwargs=dict(
                    rank=rank,
                    world_size=world_size,
                    master_port=master_port,
                    output_writer=output_writer,
                ),
            )
            p.start()
            processes.append(p)

        for _ in range(world_size):
            self.assertTrue(
                output_reader.recv(), f"Subprocess fail. Check the logs above."
            )

        for p in processes:
            p.join()

    # def test_all_gather(self):


if __name__ == "__main__":
    unittest.main()
