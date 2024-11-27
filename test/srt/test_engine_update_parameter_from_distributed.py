import gc
import os
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

import sglang as sgl
from sglang.srt.utils import init_custom_process_group
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
)

mp.set_start_method("spawn", force=True)


class TestParameterUpdateGroup(unittest.TestCase):
    @classmethod
    def init_process(cls, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "65500"

        if rank == 0:
            os.environ["NCCL_CUMEM_ENABLE"] = "0"
            os.environ["NCCL_NVLS_ENABLE"] = "0"
            hf_model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST, torch_dtype="bfloat16"
            ).to("cuda:0")
            group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:65500",
                world_size=world_size,
                rank=rank,
                group_name="test_parameter_update_group",
            )
            del hf_model
            gc.collect()
            torch.cuda.empty_cache()

        elif rank == 1:
            engine = sgl.Engine(
                model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                random_seed=42,
                base_gpu_id=rank,
            )
            engine.init_parameter_update_group(
                master_address="localhost",
                master_port="65500",
                rank_offset=1,
                world_size=world_size,
                group_name="test_parameter_update_group",
                backend="nccl",
            )
            engine.shutdown()

    @classmethod
    def setUpClass(cls):
        cls.world_size = 2

        mp.spawn(
            cls.init_process,
            args=(cls.world_size,),
            nprocs=cls.world_size,
            join=True,
        )

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def test_init_parameter_update_group(self):
        print(
            "Successfully initialized parameter update group between huggingface and SGLang server."
        )


if __name__ == "__main__":
    unittest.main()
