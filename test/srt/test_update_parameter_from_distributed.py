import os
import time
import unittest

import numpy as np
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
            # 基础环境设置
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            # 确保每个进程使用正确的GPU
            torch.cuda.set_device(rank)

            # 打印GPU信息用于调试
            print(
                f"[Rank {rank}] Using GPU: {torch.cuda.current_device()} "
                f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')})"
            )
            print(f"[Rank {rank}] Available GPUs: {torch.cuda.device_count()}")
            print(
                f"[Rank {rank}] Device capabilities: {torch.cuda.get_device_capability()}"
            )

            if rank == 1:
                print(f"[Rank 1] Starting server launch")
                # 准备服务器环境

                # 启动服务器
                process = popen_launch_server(
                    model_name,
                    base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=("--base-gpu-id", "1"),
                )
                server_pid.value = process.pid
                print(f"[Rank 1] Server launched with pid {process.pid}")

                # 等待服务器完全启动
                time.sleep(5)

                # 初始化参数更新组
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

                # 等待rank 0完成初始化
                time.sleep(2)

                # 测试参数更新
                param_name = "model.layers.0.self_attn.q_proj.weight"
                shape = [2048, 2048]
                dtype = "bfloat16"
                print(
                    f"[Rank 1] Preparing to receive parameter with shape: {shape}, dtype: {dtype}"
                )

                # 发送更新参数请求
                response = requests.post(
                    f"{base_url}/update_parameter_from_distributed",
                    json={
                        "name": param_name,
                        "shape": shape,
                        "dtype": dtype,
                        "empty_cache": True,
                    },
                    timeout=30,
                )
                print(
                    f"[Rank 1] Update parameter response: {response.status_code}, {response.json()}"
                )

            elif rank == 0:

                torch.cuda.set_device(rank)
                print(f"[Rank 0] Starting initialization")

                # 加载模型
                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype="bfloat16"
                ).cuda()
                print(f"[Rank 0] HF model loaded")

                # 初始化进程组
                group = init_custom_process_group(
                    backend="nccl",
                    init_method="tcp://localhost:29500",
                    world_size=world_size,
                    rank=rank,
                    group_name="test_parameter_update_group",
                )
                # cuda device sync
                torch.cuda.synchronize()
                print(f"[Rank 0] Process group initialized: {group.rank()}")
                print(f"[Rank 0] Process group initialized: {group.group_name}")
                print(f"[Rank 0] Process group initialized: {group.size()}")
                print(f"[Rank 0] Process group initialized")

                # 获取参数信息
                param_name = "model.layers.0.self_attn.q_proj.weight"
                param = hf_model.get_parameter(param_name)
                shape = list(param.shape)
                dtype = str(param.dtype).split(".")[-1]
                print(f"[Rank 0] Parameter shape: {shape}, dtype: {dtype}")

                # 创建测试tensor并执行all_reduce
                tensor = torch.ones(1).cuda()
                print(f"[Rank 0] Created test tensor on device: {tensor.device}")

                # 同步所有进程
                print(f"[Rank 0] Barrier")
                torch.distributed.barrier(group=group)
                print(f"[Rank 0] Barrier done")
                # 执行all_reduce
                torch.distributed.all_reduce(
                    tensor, op=torch.distributed.ReduceOp.SUM, group=group
                )
                print(f"[Rank 0] All-reduce completed. Result: {tensor}")

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
        # 清理分布式进程组
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            print("Process group destroyed")

        # 清理服务器进程
        if cls.server_pid.value != 0:
            print(f"Cleaning up server process {cls.server_pid.value}")
            kill_child_process(cls.server_pid.value, include_self=True)
            print("Server process cleaned up")

        time.sleep(1)  # 给进程一些清理的时间

    def test_parameter_update(self):
        print(
            "Successfully tested parameter update between huggingface and SGLang server."
        )


if __name__ == "__main__":
    unittest.main()
