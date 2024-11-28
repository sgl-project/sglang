import gc
import os
import time
import unittest

import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

import sglang as sgl
from sglang.srt.utils import init_custom_process_group
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
)

mp.set_start_method("spawn", force=True)


class TestParameterUpdateGroup(unittest.TestCase):

    @classmethod
    def init_process(
        cls,
        rank,
        world_size,
        param_queue,
        truncate_size,
        state_dict_key_to_shape,
        tp_size,
        model_name,
    ):
        torch.cuda.set_device(rank)
        parameters = [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.2.self_attn.k_proj.weight",
            "model.layers.3.self_attn.v_proj.weight",
            "model.layers.4.self_attn.o_proj.weight",
            "model.layers.5.mlp.gate_proj.weight",
            "model.layers.6.mlp.up_proj.weight",
            "model.layers.7.mlp.down_proj.weight",
            "model.layers.8.post_attention_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        if rank == 0:
            os.environ["NCCL_CUMEM_ENABLE"] = "0"
            os.environ["NCCL_NVLS_ENABLE"] = "0"
            cls.hf_instruct_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="bfloat16"
            ).to("cuda:0")
            base_model_name = model_name.replace("-Instruct", "")
            cls.hf_base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype="bfloat16"
            ).to("cuda:0")
            cls.hf_instruct_params = []
            cls.hf_base_params = []
            print(f"get parameter in hf instruct model and base model")
            for parameter_name in parameters:
                cls.hf_instruct_params.append(
                    cls.hf_instruct_model.get_parameter(parameter_name)[:truncate_size]
                    .cpu()
                    .detach()
                    .float()
                    .numpy()
                    .tolist()
                )
                cls.hf_base_params.append(
                    cls.hf_base_model.get_parameter(parameter_name)[:truncate_size]
                    .cpu()
                    .detach()
                    .float()
                    .numpy()
                    .tolist()
                )

            param_queue.put(("hf_instruct_params", cls.hf_instruct_params))
            param_queue.put(("hf_base_params", cls.hf_base_params))
            print(f"rank {rank} init custom process group")
            cls.group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:65500",
                world_size=world_size,
                rank=rank,
                group_name="test_parameter_update_group",
            )
            start_time = time.time()
            print(f"rank {rank} broadcast parameter")
            for parameter_name in state_dict_key_to_shape.keys():
                torch.distributed.broadcast(
                    cls.hf_base_model.get_parameter(parameter_name),
                    src=0,
                    group=cls.group,
                )
            end_time = time.time()
            print(f"rank {rank} broadcast parameter time: {end_time - start_time}")
            del cls.hf_instruct_model
            del cls.hf_base_model
            gc.collect()
            torch.cuda.empty_cache()

        elif rank == 1:
            cls.engine = sgl.Engine(
                model_path=model_name,
                random_seed=42,
                base_gpu_id=rank,
                tp_size=tp_size,
            )

            cls.engine_instruct_params = []
            print(f"rank {rank} get parameter in engine instruct model")
            for parameter_name in parameters:
                cls.engine_instruct_params.append(
                    cls.engine.get_weights_by_parameter_name(
                        parameter_name, truncate_size
                    )
                )
            param_queue.put(("engine_instruct_params", cls.engine_instruct_params))
            print(f"rank {rank} init parameter update group")
            cls.engine.init_parameter_update_group(
                master_address="localhost",
                master_port="65500",
                rank_offset=1,
                world_size=world_size,
                group_name="test_parameter_update_group",
                backend="nccl",
            )
            start_time = time.time()
            print(f"rank {rank} update parameter from distributed")
            for parameter_name in state_dict_key_to_shape.keys():
                cls.engine.update_parameter_from_distributed(
                    parameter_name,
                    dtype="bfloat16",
                    shape=state_dict_key_to_shape[parameter_name],
                    empty_cache=True,
                )
            end_time = time.time()
            print(
                f"rank {rank} update parameter from distributed time: {end_time - start_time}"
            )

            cls.engine_base_params = []
            print(f"rank {rank} get parameter in engine base model")
            for parameter_name in parameters:
                cls.engine_base_params.append(
                    cls.engine.get_weights_by_parameter_name(
                        parameter_name, truncate_size
                    )
                )
            param_queue.put(("engine_base_params", cls.engine_base_params))
            print(f"rank {rank} shutdown engine")
            cls.engine.shutdown()

    @classmethod
    def setUpClass(cls):
        assert torch.cuda.device_count() >= 2, "At least 2 GPUs are required"
        cls.test_suits = [1]
        cls.model_names = [
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_MODEL_NAME_FOR_TEST,
        ]

        if torch.cuda.device_count() >= 4:
            cls.test_suits.append(2)

        # 初始化每个模型的 state_dict_key_to_shape
        cls.model_state_dict_shapes = {}
        for model_name in cls.model_names:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="bfloat16"
            ).to("cuda:0")
            state_dict = model.state_dict()
            state_dict_keys = list(state_dict.keys())
            cls.model_state_dict_shapes[model_name] = {
                key: state_dict[key].shape for key in state_dict_keys
            }
            del model
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

    @classmethod
    def test_init_parameter_update_group(cls):
        truncate_size = 10

        for model_name in cls.model_names:
            print(f"Testing model: {model_name}")
            state_dict_key_to_shape = cls.model_state_dict_shapes[model_name]

            for tp_size in cls.test_suits:
                print(f"test tp_size: {tp_size}")
                param_queue = mp.Queue()
                results = {}

                context = mp.spawn(
                    cls.init_process,
                    args=(
                        1 + tp_size,
                        param_queue,
                        truncate_size,
                        state_dict_key_to_shape,
                        tp_size,
                        model_name,
                    ),
                    nprocs=2,
                    join=False,
                )

                while len(results) < 4:
                    try:
                        key, value = param_queue.get(timeout=5)
                        results[key] = value
                    except Exception as e:
                        if all(not p.is_alive() for p in context.processes):
                            break

                context.join()

                if len(results) != 4:
                    raise RuntimeError(f"Expected 4 parameters but got {len(results)}")

                hf_instruct_params = results["hf_instruct_params"]
                hf_base_params = results["hf_base_params"]
                engine_instruct_params = results["engine_instruct_params"]
                engine_base_params = results["engine_base_params"]

                for i in range(len(hf_instruct_params)):
                    assert np.allclose(
                        np.array(hf_instruct_params[i]),
                        np.array(engine_instruct_params[i]),
                    )
                    assert np.allclose(
                        np.array(hf_base_params[i]), np.array(engine_base_params[i])
                    )
                    assert not np.allclose(
                        np.array(hf_instruct_params[i]), np.array(hf_base_params[i])
                    )

                del context
                param_queue.close()
                param_queue.join_thread()
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(2)


if __name__ == "__main__":
    unittest.main()
