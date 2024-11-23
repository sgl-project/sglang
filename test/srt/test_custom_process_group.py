import os
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

from sglang.srt.utils import init_custom_process_group, kill_child_process
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

mp.set_start_method("spawn", force=True)


class TestParameterUpdateDistributed(unittest.TestCase):
    @classmethod
    def init_process(
        cls, rank, world_size, base_url, model_name, tensor_value, server_pid
    ):
        try:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            torch.cuda.set_device(rank)

            group_name = "test_group_for_custom_process_group"
            group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:29500",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )

            if rank == 0:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype="bfloat16"
                ).to(f"cuda:{rank}")
                print("HF model loaded on rank 0")
            elif rank == 1:
                process = popen_launch_server(
                    model_name,
                    base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=("--base-gpu-id", "1"),
                )
                server_pid.value = process.pid
                print("SGLang server launched on rank 1")
                time.sleep(5)

            tensor = torch.ones(1).cuda() * (rank + 2)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
            print(f"Rank {rank} sees sum: {tensor.item()}")

            if rank == 0:
                tensor_value.value = tensor.item()

            dist.barrier(group=group)

        finally:
            if group is not None:
                dist.destroy_process_group(group)

    @classmethod
    def setUpClass(cls):
        cls.world_size = 2
        cls.model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.tensor_value = mp.Value("d", 0.0)
        cls.server_pid = mp.Value("i", 0)

        mp.spawn(
            cls.init_process,
            args=(
                cls.world_size,
                cls.base_url,
                cls.model_name,
                cls.tensor_value,
                cls.server_pid,
            ),
            nprocs=cls.world_size,
            join=True,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.server_pid.value != 0:
            kill_child_process(cls.server_pid.value, include_self=True)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def test_custom_process_group(self):
        self.assertEqual(self.__class__.tensor_value.value, 5)


if __name__ == "__main__":
    unittest.main()
