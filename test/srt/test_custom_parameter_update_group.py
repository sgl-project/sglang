import os
import time
import unittest
import requests
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

class TestParameterUpdateGroup(unittest.TestCase):
    @classmethod
    def init_process(cls, rank, world_size, base_url, model_name, server_pid):
        # 设置分布式环境
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        torch.cuda.set_device(rank)
        
        if rank == 0:
            print(f"[Rank 0] Starting initialization")
            hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(
                f"cuda:{rank}"
            )
            print(f"[Rank 0] HF model loaded")
            
            group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:29500",
                world_size=world_size,
                rank=rank,
                group_name="test_parameter_update_group",
            )
            print(f"[Rank 0] Process group initialized")

        elif rank == 1:
            print(f"[Rank 1] Starting server launch")
            server_env = os.environ.copy()
            server_env["CUDA_VISIBLE_DEVICES"] = str(rank)
            server_env["RANK"] = str(rank)
            server_env["BACKEND"] = "nccl"
            process = popen_launch_server(
                model_name,
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                env=server_env,
            )
            server_pid = process.pid
            print(f"[Rank 1] Server launched with pid {process.pid}")
            
            response = requests.post(
                    f"{base_url}/init_parameter_update_group",
                    json={
                        "master_address": "localhost",
                        "master_port": "29500",
                        "rank_offset": 1,
                        "world_size": world_size,
                        "group_name": "test_parameter_update_group",
                        "backend": "nccl"
                    },
                timeout=30
            )

    @classmethod
    def setUpClass(cls):
        cls.world_size = 2
        cls.model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.server_pid = None

        mp.spawn(
            cls.init_process,
            args=(cls.world_size, cls.base_url, cls.model_name, cls.server_pid),
            nprocs=cls.world_size,
            join=True
    )

    @classmethod
    def tearDownClass(cls):
        if cls.server_pid != 0:
            kill_child_process(cls.server_pid, include_self=True)
        for p in cls.processes:
            p.join()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def test_init_parameter_update_group(self):
        pass

if __name__ == "__main__":
    unittest.main()