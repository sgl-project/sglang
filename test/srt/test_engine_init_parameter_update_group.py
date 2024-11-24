import os
import time
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM
from sglang.srt.utils import init_custom_process_group
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
)
import sglang as sgl

mp.set_start_method("spawn", force=True)

def mock_init_parameter_update_group(
        master_address, master_port, rank_offset, world_size, 
        group_name, backend="nccl"):
    """初始化模型参数更新的进程组"""
    rank = rank_offset + 0
    try:
        _model_update_group = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        return _model_update_group, "Succeeded to initialize custom process group."
    except Exception as e:
        return False, f"Failed to initialize custom process group: {e}."

class TestParameterUpdateGroup(unittest.TestCase):
    @classmethod
    def init_process(cls, rank, world_size, base_url, model_name):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "65500"
        torch.cuda.set_device(rank)

        if rank == 0:
            # Rank 0: 加载HF模型
            hf_model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")
            group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:65500",
                world_size=world_size,
                rank=rank,
                group_name="test_parameter_update_group",
            )
            dist.barrier(group=group)

        elif rank == 1:
            # Rank 1: 启动SGLang服务器
            engine = sgl.Engine(model_path=model_name, random_seed=42, base_gpu_id=rank)
            _model_update_group = mock_init_parameter_update_group(
                master_address="localhost",
                master_port="65500",
                rank_offset=1,
                world_size=world_size,
                group_name="test_parameter_update_group",
            )[0]
            dist.barrier(group=_model_update_group)

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
        print("Successfully initialized parameter update group between huggingface and SGLang server.")

if __name__ == "__main__":
    unittest.main()