import gc
import os
import time
import unittest

import numpy as np
import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

import sglang as sgl
from sglang.srt.utils import init_custom_process_group
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)
from sglang.utils import terminate_process

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
        use_engine,
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
            print(f"rank {rank} world_size: {world_size} tp_size: {tp_size}")
            cls.group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:65500",
                world_size=world_size,
                rank=rank,
                group_name="test_parameter_update_group",
            )
            dist.barrier(group=cls.group, device_ids=[rank])
            torch.cuda.synchronize()
            time_begin_broadcast = time.time()
            for parameter_name in state_dict_key_to_shape.keys():
                torch.distributed.broadcast(
                    cls.hf_base_model.get_parameter(parameter_name),
                    src=0,
                    group=cls.group,
                )
            torch.cuda.synchronize()
            time_end_broadcast = time.time()
            broadcast_time = time_end_broadcast - time_begin_broadcast
            print(f"rank {rank} broadcast parameter time: {broadcast_time:.3f}s")
            param_queue.put(("broadcast_time", broadcast_time))

            del cls.hf_instruct_model
            del cls.hf_base_model
            gc.collect()
            torch.cuda.empty_cache()
        elif rank in [1, 2]:
            torch.cuda.set_device(rank)
            torch.cuda.synchronize()
            base_gpu_id = 1 if rank == 1 else 1 + tp_size
            if use_engine:
                engine = sgl.Engine(
                    model_path=model_name,
                    random_seed=42,
                    base_gpu_id=base_gpu_id,
                    tp_size=tp_size,
                )
            else:
                if rank == 1:
                    url = DEFAULT_URL_FOR_TEST
                else:
                    url = DEFAULT_URL_FOR_TEST.replace("2157", "2159")
                process = popen_launch_server(
                    model_name,
                    url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=(
                        "--base-gpu-id",
                        str(base_gpu_id),
                        "--tp-size",
                        str(tp_size),
                    ),
                )
            torch.cuda.synchronize()
            if not use_engine:
                print(f"rank {rank} url: {url}")
            instruct_params = []
            for parameter_name in parameters:
                instruct_params.append(
                    engine.get_weights_by_name(parameter_name, truncate_size)
                    if use_engine
                    else requests.get(
                        f"{url}/get_weights_by_name",
                        json={"name": parameter_name, "truncate_size": truncate_size},
                    ).json()
                )

            param_queue.put((f"sgl_dp_{rank}_instruct_params", instruct_params))
            if use_engine:
                engine.init_weight_update_group(
                    master_address="localhost",
                    master_port="65500",
                    rank_offset=base_gpu_id,
                    world_size=world_size,
                    group_name="test_parameter_update_group",
                    backend="nccl",
                )
            else:
                requests.post(
                    f"{url}/init_weight_update_group",
                    json={
                        "master_address": "localhost",
                        "master_port": "65500",
                        "rank_offset": base_gpu_id,
                        "world_size": world_size,
                        "group_name": "test_parameter_update_group",
                        "backend": "nccl",
                    },
                )
            torch.cuda.synchronize()
            time_begin_update = time.time()
            for parameter_name in state_dict_key_to_shape.keys():
                if use_engine:
                    engine.update_parameter_from_distributed(
                        parameter_name,
                        dtype=torch.bfloat16,
                        shape=state_dict_key_to_shape[parameter_name],
                        empty_cache=True,
                    )
                else:
                    requests.post(
                        f"{url}/update_parameter_from_distributed",
                        json={
                            "name": parameter_name,
                            "dtype": "bfloat16",
                            "shape": state_dict_key_to_shape[parameter_name],
                            "empty_cache": True,
                        },
                    )
            torch.cuda.synchronize()
            time_end_update = time.time()
            update_time = time_end_update - time_begin_update
            print(
                f"fully update model_name {model_name} rank {rank} parameter from distributed time: {update_time:.3f}s"
            )
            param_queue.put((f"update_sgl_dp_{rank}_time", update_time))
            base_params = []
            for parameter_name in parameters:
                if use_engine:
                    base_params.append(
                        engine.get_weights_by_name(parameter_name, truncate_size)
                    )
                else:
                    base_params.append(
                        requests.get(
                            f"{url}/get_weights_by_name",
                            json={
                                "name": parameter_name,
                                "truncate_size": truncate_size,
                            },
                        ).json()
                    )
            param_queue.put((f"sgl_dp_{rank}_base_params", base_params))
            if use_engine:
                engine.shutdown()
            else:
                terminate_process(process)

    @classmethod
    def setUpClass(cls):
        cls.model_names = [
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_MODEL_NAME_FOR_TEST,
        ]
        assert torch.cuda.device_count() >= 2, "At least 2 GPUs are required"
        # test_suits : tp, dp
        cls.test_suits = [(1, 1)]

        if torch.cuda.device_count() >= 4:
            cls.test_suits.extend([(2, 1), (1, 2)])

        if torch.cuda.device_count() >= 5:
            cls.test_suits.append((2, 2))

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

    @classmethod
    def test_init_weight_update_group(cls):
        truncate_size = 10

        for model_name in cls.model_names:
            state_dict_key_to_shape = cls.model_state_dict_shapes[model_name]

            for tp_size, dp_size in cls.test_suits:
                for use_engine in [False, True]:
                    print(
                        f"Testing model: {model_name} tp_size: {tp_size}, dp_size: {dp_size} use_engine: {use_engine}"
                    )
                    param_queue = mp.Queue()
                    results = {}

                    context = mp.spawn(
                        cls.init_process,
                        args=(
                            1 + tp_size * dp_size,
                            param_queue,
                            truncate_size,
                            state_dict_key_to_shape,
                            tp_size,
                            model_name,
                            use_engine,
                        ),
                        nprocs=1 + dp_size,
                        join=False,
                    )

                    while len(results) < 3 * (1 + dp_size):
                        try:
                            key, value = param_queue.get(timeout=5)
                            results[key] = value
                        except Exception as e:
                            if all(not p.is_alive() for p in context.processes):
                                break

                    context.join()

                    if len(results) != 3 * (1 + dp_size):
                        raise RuntimeError(
                            f"Expected {3 * (1 + dp_size)} parameters but got {len(results)}"
                        )

                    hf_instruct_params = results.get("hf_instruct_params")
                    hf_base_params = results.get("hf_base_params")
                    broadcast_time = results.get("broadcast_time")
                    sgl_dp_one_instruct_params = results.get("sgl_dp_1_instruct_params")
                    sgl_dp_one_base_params = results.get("sgl_dp_1_base_params")
                    update_sgl_dp_one_time = results.get("update_sgl_dp_1_time")
                    sgl_dp_two_instruct_params = results.get(
                        "sgl_dp_2_instruct_params", None
                    )
                    sgl_dp_two_base_params = results.get("sgl_dp_2_base_params", None)
                    update_sgl_dp_two_time = results.get("update_sgl_dp_2_time", None)
                    if dp_size == 2:
                        assert sgl_dp_two_instruct_params is not None
                        assert sgl_dp_two_base_params is not None
                        assert update_sgl_dp_two_time is not None

                    for i in range(len(hf_instruct_params)):
                        assert np.allclose(
                            np.array(hf_instruct_params[i]),
                            np.array(sgl_dp_one_instruct_params[i]),
                        ), f"sgl_dp_one_instruct_params rank {i} is not close"

                        assert np.allclose(
                            np.array(hf_base_params[i]),
                            np.array(sgl_dp_one_base_params[i]),
                        ), f"sgl_dp_one_base_params rank {i} is not close"

                        assert not np.allclose(
                            np.array(hf_instruct_params[i]), np.array(hf_base_params[i])
                        ), f"hf_instruct_params rank {i} is not close"

                        if sgl_dp_two_base_params is not None:
                            assert np.allclose(
                                np.array(hf_base_params[i]),
                                np.array(sgl_dp_two_base_params[i]),
                            ), f"sgl_dp_two_base_params rank {i} is not close"

                        if sgl_dp_two_instruct_params is not None:
                            assert np.allclose(
                                np.array(hf_instruct_params[i]),
                                np.array(sgl_dp_two_instruct_params[i]),
                            ), f"sgl_dp_two_instruct_params rank {i} is not close"

                    # Time limit for broadcast and update on CI is 3 / 6
                    # On local H100, it's 1 / 2
                    time_limit = (
                        3 if model_name == DEFAULT_SMALL_MODEL_NAME_FOR_TEST else 6
                    )

                    assert (
                        broadcast_time < time_limit
                    ), f"broadcast_time exceeds time limit {time_limit}s"

                    assert (
                        update_sgl_dp_one_time < time_limit
                    ), f"update_sgl_dp_one_time exceeds time limit {time_limit}s"

                    if sgl_dp_two_instruct_params is not None:
                        assert (
                            update_sgl_dp_two_time < time_limit
                        ), f"update_sgl_dp_two_time exceeds time limit {time_limit}s"

                    del context
                    param_queue.close()
                    param_queue.join_thread()
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
