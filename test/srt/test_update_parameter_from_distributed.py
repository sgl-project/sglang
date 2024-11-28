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
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

mp.set_start_method("spawn", force=True)


class TestParameterUpdateGroup(unittest.TestCase):

    @classmethod
    def init_process(
        cls,
        rank,
        world_size,
        param_queue,
        parameter_name,
        truncate_size,
        state_dict_key_to_shape,
    ):
        torch.cuda.set_device(rank)
        if rank == 0:
            os.environ["NCCL_CUMEM_ENABLE"] = "0"
            os.environ["NCCL_NVLS_ENABLE"] = "0"
            cls.hf_instruct_model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST, torch_dtype="bfloat16"
            ).to("cuda:0")
            base_model_name = DEFAULT_SMALL_MODEL_NAME_FOR_TEST.replace("-Instruct", "")
            cls.hf_base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype="bfloat16"
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
            assert not np.allclose(
                np.array(cls.hf_instruct_param), np.array(cls.hf_base_param)
            )
            param_queue.put(("hf_instruct_param", cls.hf_instruct_param))
            param_queue.put(("hf_base_param", cls.hf_base_param))
            cls.group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:65500",
                world_size=world_size,
                rank=rank,
                group_name="test_parameter_update_group",
            )
            torch.distributed.broadcast(
                cls.hf_base_model.get_parameter(parameter_name), src=0, group=cls.group
            )
            del cls.hf_instruct_model
            del cls.hf_base_model
            gc.collect()
            torch.cuda.empty_cache()

        elif rank == 1:
            cls.engine = sgl.Engine(
                model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                random_seed=42,
                base_gpu_id=rank,
            )
            cls.engine_instruct_param = cls.engine.get_weights_by_parameter_name(
                parameter_name, truncate_size
            )
            torch.cuda.synchronize()
            param_queue.put(("engine_instruct_param", cls.engine_instruct_param))
            torch.cuda.synchronize()
            print("init parameter update group")
            cls.engine.init_parameter_update_group(
                master_address="localhost",
                master_port="65500",
                rank_offset=1,
                world_size=world_size,
                group_name="test_parameter_update_group",
                backend="nccl",
            )
            torch.cuda.synchronize()
            print("init parameter update group done")
            torch.cuda.synchronize()
            print(f"update parameter from distributed {parameter_name} bfloat16")
            torch.cuda.synchronize()
            print(f"{state_dict_key_to_shape[parameter_name]}")
            cls.engine.update_parameter_from_distributed(
                parameter_name,
                dtype="bfloat16",
                shape=state_dict_key_to_shape[parameter_name],
                empty_cache=True,
            )
            torch.cuda.synchronize()
            print("update parameter from distributed done")
            cls.engine_base_param = cls.engine.get_weights_by_parameter_name(
                parameter_name, truncate_size
            )
            param_queue.put(("engine_base_param", cls.engine_base_param))
            cls.engine.shutdown()

    @classmethod
    def setUpClass(cls):
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST, torch_dtype="bfloat16"
        ).to("cuda:0")
        state_dict = model.state_dict()
        state_dict_keys = list(state_dict.keys())
        cls.state_dict_key_to_shape = {
            key: state_dict[key].shape for key in state_dict_keys
        }
        cls.world_size = 2
        del model
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def test_init_parameter_update_group(cls):
        parameter_name = list(cls.state_dict_key_to_shape.keys())[0]
        print(f"parameter_name: {parameter_name}")
        truncate_size = 100
        param_queue = mp.Queue()
        results = {}

        # 启动子进程
        context = mp.spawn(
            cls.init_process,
            # pass in parameter rather than get it from cls
            args=(
                cls.world_size,
                param_queue,
                parameter_name,
                truncate_size,
                cls.state_dict_key_to_shape,
            ),
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

        hf_instruct_param = np.array(results["hf_instruct_param"])
        hf_base_param = np.array(results["hf_base_param"])
        engine_instruct_param = np.array(results["engine_instruct_param"])
        engine_base_param = np.array(results["engine_base_param"])
        print(f"hf_instruct_param: {hf_instruct_param.shape}")
        print(f"engine_instruct_param: {engine_instruct_param.shape}")
        print(f"hf_base_param: {hf_base_param.shape}")
        print(f"engine_base_param: {engine_base_param.shape}")
        assert np.allclose(hf_instruct_param, engine_instruct_param)
        assert np.allclose(hf_base_param, engine_base_param)
        assert not np.allclose(hf_instruct_param, hf_base_param)
        assert not np.allclose(hf_instruct_param, engine_base_param)
        assert not np.allclose(hf_base_param, engine_instruct_param)


if __name__ == "__main__":
    unittest.main()
