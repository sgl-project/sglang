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
    def init_process(cls, rank, world_size, base_url, model_name):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "65500"
        torch.cuda.set_device(rank)

        if rank == 0:
            # Rank 0: 加载HF模型
            os.environ["NCCL_CUMEM_ENABLE"] = "0"
            os.environ["NCCL_NVLS_ENABLE"] = "0"
            hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16").to("cuda:0")
            group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:65500",
                world_size=world_size,
                rank=rank,
                group_name="test_parameter_update_group",
            )
            print(f"rank: {rank}, before barrier")
            dist.barrier(group=group)
            print(f"rank: {rank}, after barrier")
            param_name = "model.layers.0.self_attn.q_proj.weight"
            param = hf_model.get_parameter(param_name)
            shape = list(param.shape)
            print(f"rank: {rank}, param_name: {param_name}, shape: {shape}")
            dtype = str(param.dtype).split(".")[-1]
            print(f"[Rank 0] Parameter shape: {shape}, dtype: {dtype}")
            torch.distributed.broadcast(
                param, src=0, group=group
            )
            del hf_model
            gc.collect()
            torch.cuda.empty_cache()

        elif rank == 1:
            engine = sgl.Engine(model_path=model_name, random_seed=42, base_gpu_id=rank)
            engine.init_parameter_update_group(
                master_address="localhost",
                master_port="65500",
                rank_offset=1,
                world_size=world_size,
                group_name="test_parameter_update_group",
                backend="nccl",
            )
            param_name = "model.layers.0.self_attn.q_proj.weight"
            shape = [2048, 2048]
            dtype = "bfloat16"
            engine.update_parameter_from_distributed(
                name=param_name,
                dtype=dtype,
                shape=shape,
                empty_cache=True,
            )
            engine.shutdown()

    @classmethod
    def setUpClass(cls):
        cls.world_size = 2
        cls.model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        mp.spawn(
            cls.init_process,
            args=(cls.world_size, cls.base_url, cls.model_name),
            nprocs=cls.world_size,
            join=True,
        )

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        time.sleep(1)

    def test_init_parameter_update_group(self):
        print(
            "Successfully initialized parameter update group between huggingface and SGLang server."
        )


if __name__ == "__main__":
    unittest.main()


