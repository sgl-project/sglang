import asyncio
import json
import os
import time
import unittest
from multiprocessing import process
from types import SimpleNamespace

import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

import sglang as sgl
from sglang.bench_offline_throughput import BenchArgs, throughput_test
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import init_custom_process_group, kill_child_process
from sglang.test.few_shot_gsm8k_engine import run_eval
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

mp.set_start_method("spawn", force=True)


def mock_init_parameter_update_group(
    master_address,
    master_port,
    rank_offset,
    world_size,
    group_name,
    backend="nccl",
):
    rank = rank_offset + 0

    _model_update_group = init_custom_process_group(
        backend=backend,
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=rank,
        group_name=group_name,
    )

    return _model_update_group


class TestParameterUpdateGroup(unittest.TestCase):
    @classmethod
    def init_process(cls, rank, world_size):
        try:
            # 设置分布式环境
            torch.cuda.set_device(rank)

            print(
                f"[Rank {rank}] Using GPU: {torch.cuda.current_device()} "
                f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
            )

            if rank == 0:
                os.environ["NCCL_CUMEM_ENABLE"] = "0"
                os.environ["NCCL_NVLS_ENABLE"] = "0"
                print(f"[Rank 0] Starting initialization")
                group = init_custom_process_group(
                    backend="nccl",
                    init_method="tcp://localhost:29500",
                    world_size=world_size,
                    rank=rank,
                    group_name="test_parameter_update_group",
                )
                print(f"[Rank 0] Process group initialized")
                print(f"[Rank 0] before barrier")
                dist.barrier(group=group)
                print(f"[Rank 0] after barrier")

            elif rank == 1:
                print(f"[Rank 1] Starting server launch")
                time.sleep(2)
                _model_update_group = mock_init_parameter_update_group(
                    master_address="localhost",
                    master_port="29500",
                    rank_offset=1,
                    world_size=world_size,
                    group_name="test_parameter_update_group",
                    backend="nccl",
                )
                print(f"[Rank 1] before barrier")
                dist.barrier(group=_model_update_group)
                print(f"[Rank 1] after barrier")

            print(f"[Rank {rank}] Process initialization completed")

        except Exception as e:
            print(f"[Rank {rank}] Error occurred: {str(e)}")
            raise

    @classmethod
    def setUpClass(cls):
        cls.world_size = 2

        print("Starting multiprocessing spawn")
        mp.spawn(
            cls.init_process,
            args=(cls.world_size,),
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

        time.sleep(1)  # 给进程一些清理的时间

    def test_init_parameter_update_group(self):
        print(
            "Successfully initialized parameter update group between huggingface and SGLang server."
        )


if __name__ == "__main__":
    unittest.main()
