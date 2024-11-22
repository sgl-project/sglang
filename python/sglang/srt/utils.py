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
    def init_process(
        cls,
        rank,
        world_size,
        base_url,
        model_name,
        success_flag,
        server_pid,
        ready_event,
    ):
        try:
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

                # 通知 Rank 1 可以发送请求
                ready_event.set()

                # 等待所有进程完成
                dist.barrier(group=group)
                success_flag.value = 1
                print(f"[Rank 0] Passed barrier and set success_flag")

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
                server_pid.value = process.pid
                print(f"[Rank 1] Server launched with pid {process.pid}")

                # 等待 Rank 0 初始化完成
                ready_event.wait()

                try:
                    print(f"[Rank 1] Sending init request")
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
                    if response.status_code == 200:
                        print(f"[Rank 1] Init request sent successfully")
                    else:
                        print(
                            f"[Rank 1] Init request failed with status {response.status_code}"
                        )
                except Exception as e:
                    print(f"[Rank 1] Error sending request: {e}")

                # 等待 Rank 0 通过 barrier
                dist.barrier(group=dist.group.WORLD)
                success_flag.value = 1
                print(f"[Rank 1] Passed barrier and set success_flag")
        finally:
            if rank == 0 and torch.distributed.is_initialized():
                print(f"[Rank 0] Destroying process group")
                dist.destroy_process_group()

    @classmethod
    def setUpClass(cls):
        cls.world_size = 2
        cls.model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        # 用于跨进程共享状态
        cls.success_flag = mp.Value("i", 0)
        cls.server_pid = mp.Value("i", 0)
        cls.ready_event = mp.Event()

        # 启动多进程
        cls.processes = []
        for rank in range(cls.world_size):
            p = mp.Process(
                target=cls.init_process,
                args=(
                    rank,
                    cls.world_size,
                    cls.base_url,
                    cls.model_name,
                    cls.success_flag,
                    cls.server_pid,
                    cls.ready_event,
                ),
            )
            p.start()
            cls.processes.append(p)

        # 等待进程完成
        for p in cls.processes:
            p.join()

    @classmethod
    def tearDownClass(cls):
        if cls.server_pid.value != 0:
            kill_child_process(cls.server_pid.value, include_self=True)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def test_init_parameter_update_group(self):
        # 验证参数更新组是否成功初始化
        self.assertEqual(self.success_flag.value, 1)


if __name__ == "__main__":
    unittest.main()
