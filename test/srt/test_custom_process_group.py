import os
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

import sglang as sgl
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
    def init_process(cls, rank, world_size, base_url, model_name, tensor_value):
        try:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "30000"
            torch.cuda.set_device(rank)
            engine = None

            group_name = "test_group_for_custom_process_group"
            group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:30000",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            print(f"Initialized custom process group on rank {rank}")
            print(f"rank: {rank}, before barrier")
            dist.barrier(group=group)
            print(f"rank: {rank}, after barrier")

            if rank == 0:
                # hf_model = AutoModelForCausalLM.from_pretrained(
                #     model_name, torch_dtype="bfloat16"
                # ).to(f"cuda:{rank}")
                print("HF model loaded on rank 0")
            elif rank == 1:
                # engine = sgl.Runtime(
                #     model_path=model_name, random_seed=42, base_gpu_id=rank
                # )
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
            if engine is not None:
                engine.shutdown()

    @classmethod
    def setUpClass(cls):
        cls.world_size = 2
        cls.model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.tensor_value = mp.Value("d", 0.0)

        mp.spawn(
            cls.init_process,
            args=(
                cls.world_size,
                cls.base_url,
                cls.model_name,
                cls.tensor_value,
            ),
            nprocs=cls.world_size,
            join=True,
        )

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def test_custom_process_group(self):
        self.assertEqual(self.__class__.tensor_value.value, 5)


if __name__ == "__main__":
    unittest.main()
