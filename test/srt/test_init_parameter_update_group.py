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

mp.set_start_method("spawn", force=True)


class TestParameterUpdateGroup(unittest.TestCase):
    @classmethod
    def init_process(cls, rank, world_size, base_url, model_name, server_pid):
        try:
            # 设置分布式环境
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
            torch.cuda.set_device(0)  # 使用本地GPU ID 0，因为每个进程只能看到一个GPU

            print(
                f"[Rank {rank}] Using GPU: {torch.cuda.current_device()} "
                f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
            )

            if rank == 0:
                print(f"[Rank 0] Starting initialization")
                hf_model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")
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
                server_env["RANK"] = str(rank)
                server_env["BACKEND"] = "nccl"
                process = popen_launch_server(
                    model_name,
                    base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    env=server_env,
                )
                server_pid.value = process.pid
                print(f"[Rank 1] Server launched with pid {process.pid}")

                response = requests.post(
                    f"{base_url}/init_parameter_update_group",
                    json={
                        "master_address": "localhost",
                        "master_port": "29500",
                        "rank_offset": 1,
                        "world_size": world_size,
                        "group_name": "test_parameter_update_group",
                        "backend": "nccl",
                    },
                    timeout=30,
                )
                print(
                    f"[Rank 1] Parameter update group initialized with response: {response.status_code}"
                )

            print(f"[Rank {rank}] Process initialization completed")

        except Exception as e:
            print(f"[Rank {rank}] Error occurred: {str(e)}")
            raise

    @classmethod
    def setUpClass(cls):
        cls.world_size = 2
        cls.model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.server_pid = mp.Value("i", 0)

        print("Starting multiprocessing spawn")
        mp.spawn(
            cls.init_process,
            args=(
                cls.world_size,
                cls.base_url,
                cls.model_name,
                cls.server_pid,
            ),
            nprocs=cls.world_size,
            join=True,
        )
        print("Multiprocessing spawn completed")

    @classmethod
    def tearDownClass(cls):
        print("Starting teardown")
        # 先清理分布式进程组
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            print("Process group destroyed")

        # 然后清理服务器进程
        if cls.server_pid.value != 0:
            print(f"Cleaning up server process {cls.server_pid.value}")
            kill_child_process(cls.server_pid.value, include_self=True)
            print("Server process cleaned up")

        time.sleep(1)  # 给进程一些清理的时间

    def test_init_parameter_update_group(self):
        print(
            "Successfully initialized parameter update group between huggingface and SGLang server."
        )


if __name__ == "__main__":
    unittest.main()
