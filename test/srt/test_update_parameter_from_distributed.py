import gc
import os
import time
import unittest

import numpy as np
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
    def init_process(cls, rank, world_size, base_url, model_name, param_queue):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "65500"
        torch.cuda.set_device(rank)
        parameter_name = "model.layers.1.self_attn.q_proj.weight"
        truncate_size = 100

        if rank == 0:
            os.environ["NCCL_CUMEM_ENABLE"] = "0"
            os.environ["NCCL_NVLS_ENABLE"] = "0"
            # 移除所有 print 语句
            cls.hf_instruct_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="bfloat16"
            ).to("cuda:0")
            cls.hf_base_model = AutoModelForCausalLM.from_pretrained(
                model_name.replace("-instruct", ""), torch_dtype="bfloat16"
            ).to("cuda:0")
            cls.hf_instruct_param = (
                cls.hf_instruct_model.get_parameter(parameter_name)[:truncate_size]
                .cpu()
                .detach()
                .float()
                .numpy()
                .tolist()
            )
            cls.hf_base_param = (
                cls.hf_base_model.get_parameter(parameter_name)[:truncate_size]
                .cpu()
                .detach()
                .float()
                .numpy()
                .tolist()
            )
            param_queue.put(("hf_instruct_param", cls.hf_instruct_param))
            param_queue.put(("hf_base_param", cls.hf_base_param))
            torch.cuda.synchronize()
            cls.group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:65500",
                world_size=world_size,
                rank=rank,
                group_name="test_parameter_update_group",
            )
            print(cls.hf_instruct_model.get_parameter(parameter_name).shape)
            torch.cuda.synchronize()
            print(f"rank: {rank}, try to barrier")
            torch.cuda.synchronize()
            torch.distributed.barrier(group=cls.group)
            print(f"rank: {rank}, try to broadcast hf_instruct_param")
            torch.cuda.synchronize()
            torch.distributed.broadcast(
                cls.hf_base_model.get_parameter(parameter_name), src=0, group=cls.group
            )
            print(f"rank: {rank}, try to del hf_instruct_model")
            del cls.hf_instruct_model
            print(f"rank: {rank}, try to del hf_base_model")
            del cls.hf_base_model
            print(f"rank: {rank}, try to gc")
            gc.collect()
            print(f"rank: {rank}, try to empty cache")
            torch.cuda.empty_cache()

        elif rank == 1:
            cls.engine = sgl.Engine(
                model_path=model_name, random_seed=42, base_gpu_id=rank
            )
            print(f"rank: {rank}, before init_parameter_update_group")
            print(f"rank: {rank}, before get_weights_by_parameter_name")
            cls.engine_instruct_param = cls.engine.get_weights_by_parameter_name(
                parameter_name, truncate_size
            )
            torch.cuda.synchronize()
            print(f"rank: {rank}, before put engine_instruct_param")
            param_queue.put(("engine_instruct_param", cls.engine_instruct_param))
            print(f"rank: {rank}, after put engine_instruct_param")
            torch.cuda.synchronize()
            cls.engine.init_parameter_update_group(
                master_address="localhost",
                master_port="65500",
                rank_offset=1,
                world_size=world_size,
                group_name="test_parameter_update_group",
                backend="nccl",
            )
            torch.cuda.synchronize()
            print(f"rank: {rank}, before update_parameter_from_distributed")
            torch.cuda.synchronize()
            print(f"rank: {rank}, try to update_parameter_from_distributed")
            cls.engine.update_parameter_from_distributed(
                parameter_name,
                dtype="bfloat16",
                shape=torch.Size([2048, 2048]),
                empty_cache=True,
            )
            print(f"rank: {rank}, after update_parameter_from_distributed")
            torch.cuda.synchronize()
            print(f"rank: {rank}, before get engine_base_param")
            cls.engine_base_param = cls.engine.get_weights_by_parameter_name(
                parameter_name, truncate_size
            )
            print(f"rank: {rank}, after get engine_base_param")
            torch.cuda.synchronize()
            print(f"rank: {rank}, before put engine_base_param")
            param_queue.put(("engine_base_param", cls.engine_base_param))
            print(f"rank: {rank}, after put engine_base_param")
            cls.engine.shutdown()

    @classmethod
    def setUpClass(cls):
        cls.world_size = 2
        cls.model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def test_init_parameter_update_group(cls):
        param_queue = mp.Queue()
        results = {}

        # 启动子进程
        context = mp.spawn(
            cls.init_process,
            args=(cls.world_size, cls.base_url, cls.model_name, param_queue),
            nprocs=cls.world_size,
            join=False,
        )

        # 先尝试获取队列数据
        timeout = 60
        start_time = time.time()

        while len(results) < 4 and time.time() - start_time < timeout:
            try:
                key, value = param_queue.get(timeout=5)  # 增加超时时间
                print(f"Got parameter: {key}")  # 添加日志
                results[key] = value
            except Exception as e:
                print(f"Queue get error: {e}")
                if all(
                    not p.is_alive() for p in context.processes
                ):  # 修正：检查所有子进程
                    print("Child processes have terminated")
                    break

        # 等待所有子进程结束
        context.join()

        if len(results) != 4:
            raise RuntimeError(f"Expected 4 parameters but got {len(results)}")

        hf_instruct_param = results["hf_instruct_param"]
        hf_base_param = results["hf_base_param"]
        engine_instruct_param = results["engine_instruct_param"]
        engine_base_param = results["engine_base_param"]
        assert np.allclose(np.array(hf_instruct_param), np.array(engine_instruct_param))
        assert np.allclose(np.array(hf_base_param), np.array(engine_base_param))
        assert not np.allclose(np.array(hf_instruct_param), np.array(engine_base_param))
        assert not np.allclose(np.array(hf_base_param), np.array(engine_instruct_param))


if __name__ == "__main__":
    unittest.main()
