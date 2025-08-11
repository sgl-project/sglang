"""Test loading weights from remote instance.

This test suite simulates loading weights from a remote instance.
Rank 0 repsresents the seed instance, while ranks 1 respresent the
new instance that needs to loading weights from the seed instance.

"""

import gc
import os
import random
import unittest

import numpy as np
import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sglang as sgl
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)
from sglang.utils import terminate_process

mp.set_start_method("spawn", force=True)


def verify_params_close(params1, params2, error_msg):
    """Verify if two parameter arrays are close enough."""
    try:
        assert np.allclose(np.array(params1), np.array(params2)), error_msg
    except Exception as e:
        print(f"Parameters not close for {error_msg}")
        print("Params1:", np.array(params1))
        print("Params2:", np.array(params2))
        raise e

def init_process(
    rank,
    param_queue,
    truncate_size,
    tp_size,
    model_name,
    backend,
    checking_parameters,
    remote_instance_model_path,
    seed_instance_url,
    dst_instance_id,
    event_seed_ready,
    event_dst_ready,
):
    torch.cuda.set_device(rank)

    if rank == 0:
        init_process_seed(
            rank,
            param_queue,
            truncate_size,
            model_name,
            checking_parameters,
            tp_size,
            event_seed_ready,
            event_dst_ready,
        )
    elif rank in [1, 2]:
        init_process_dst(
            rank,
            param_queue,
            truncate_size,
            model_name,
            remote_instance_model_path,
            seed_instance_url,
            dst_instance_id,
            checking_parameters,
            backend,
            tp_size,
            event_seed_ready,
            event_dst_ready,
        )


def init_process_seed(
    rank,
    param_queue,
    truncate_size,
    model_name,
    checking_parameters,
    tp_size,
    event_seed_ready,
    event_dst_ready,
):
    # These two environment variables are very important
    # to avoid unexpected behaviors of CUDA and NCCL.
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    # Load model and get parameters
    torch.cuda.set_device(rank)
    torch.cuda.synchronize()

    url = DEFAULT_URL_FOR_TEST
    process = popen_launch_server(
        model_name,
        url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=(
            "--base-gpu-id",
            str(rank),
            "--tp-size",
            str(tp_size),
        ),
    )
    torch.cuda.synchronize()

    seed_params = []
    # Get the weights of seed instance for correctness check.
    for parameter_name in checking_parameters:
        seed_params.append(
            requests.get(
                f"{url}/get_weights_by_name",
                json={
                    "name": parameter_name,
                    "truncate_size": truncate_size,
                },
            ).json()
        )
    param_queue.put((f"seed_params", seed_params))

    event_seed_ready.set()
    event_dst_ready.wait()
    terminate_process(process)

def init_process_dst(
    rank,
    param_queue,
    truncate_size,
    model_name,
    remote_instance_model_path,
    seed_instance_url,
    dst_instance_id,
    checking_parameters,
    backend,
    tp_size,
    event_seed_ready,
    event_dst_ready,
):
    torch.cuda.set_device(rank)
    torch.cuda.synchronize()
    base_gpu_id = 1 if rank == 1 else 1 + tp_size

    event_seed_ready.wait()

    if backend == "Engine":
        print(f"[sgl] rank {rank} init engine")
        engine = sgl.Engine(
            model_path=remote_instance_model_path,
            base_gpu_id=base_gpu_id,
            tp_size=tp_size,
            cuda_graph_max_bs=2,
            tokenizer_path=model_name,
            model_config_path=model_name,
            seed_instance_url=seed_instance_url,
            dst_instance_id=dst_instance_id,
            load_format="remote_instance",
        )
    else:
        host, _, port = DEFAULT_URL_FOR_TEST.rpartition(":")
        url = ":".join([host, str(int(port) + 10000 + rank)])

        print(f"[sgl] rank {rank} init server on url: {url}")
        process = popen_launch_server(
            remote_instance_model_path,
            url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--base-gpu-id",
                str(base_gpu_id),
                "--tp-size",
                str(tp_size),
                "--cuda-graph-max-bs",
                2,
                "--tokenizer-path",
                model_name,
                "--model-config-path",
                model_name,
                "--seed-instance-url",
                seed_instance_url,
                "--dst-instance-id",
                dst_instance_id,
                "--load-format",
                "remote_instance",
            ),
        )
    torch.cuda.synchronize()

    event_dst_ready.set()

    # Get weights of destination instance loaded from remote instance.
    dst_params = []
    for parameter_name in checking_parameters:
        dst_params.append(
            engine.get_weights_by_name(parameter_name, truncate_size)
            if backend == "Engine"
            else requests.get(
                f"{url}/get_weights_by_name",
                json={"name": parameter_name, "truncate_size": truncate_size},
            ).json()
        )

    param_queue.put((f"sgl_dp_{rank}_dst_params", dst_params))

    # Shutdown the engine or terminate the server process.
    if backend == "Engine":
        engine.shutdown()
    else:
        terminate_process(process)


def test_load_weights_from_remote_instance(
    tp_size,
    dp_size,
    model_name,
    backend,
    truncate_size,
    checking_parameters,
    remote_instance_model_path,
    seed_instance_url,
    dst_instance_id,
):
    print(
        f"Testing model: {model_name} tp_size: {tp_size}, dp_size: {dp_size} backend: {backend}"
    )
    param_queue = mp.Queue()
    results = {}
    event_seed_ready = mp.Event()
    event_dst_ready = mp.Event()

    context = mp.spawn(
        init_process,
        args=(
            param_queue,
            truncate_size,
            tp_size,
            model_name,
            backend,
            checking_parameters,
            remote_instance_model_path,
            seed_instance_url,
            dst_instance_id,
            event_seed_ready,
            event_dst_ready,
        ),
        nprocs=1 + dp_size,
        join=False,
    )

    while len(results) < (1 + dp_size):
        try:
            key, value = param_queue.get(timeout=5)
            results[key] = value
        except Exception as e:
            if all(not p.is_alive() for p in context.processes):
                break

    context.join()

    if len(results) != (1 + dp_size):
        raise RuntimeError(
            f"Expected {(1 + dp_size)} parameters but got {len(results)}"
        )

    params = {
        "seed": results.get("seed_params"),
        "sgl_dp_1_dest": results.get("sgl_dp_1_dst_params"),
    }

    if dp_size == 2:
        dp2_params = {
            "sgl_dp_2_dest": results.get("sgl_dp_2_dst_params"),
        }
        assert all(v is not None for v in dp2_params.values())
        params.update(dp2_params)

    # Check the correctness of weights loaded from remote instance
    # by verifying the weights of seed instance and destination instance.
    for i in range(len(params["seed"])):
        verify_params_close(
            params["seed"][i],
            params["sgl_dp_1_dest"][i],
            f"sgl_dp_1_dst_params rank {i}",
        )

        if dp_size == 2:
            verify_params_close(
                params["seed"][i],
                params["sgl_dp_2_dest"][i],
                f"sgl_dp_2_dst_params rank {i}",
            )

    # Delete the context and close the parameter queue.
    del context
    param_queue.close()
    param_queue.join_thread()
    gc.collect()
    torch.cuda.empty_cache()

class TestLoadWeightsFromRemoteInstance(CustomTestCase):

    def test_load_weights_from_remote_instance(self):

        assert torch.cuda.device_count() >= 2, "At least 2 GPUs are required"
        # test_suits : tp, dp, model_name, backend, dst_instance_id
        if is_in_ci():
            mode = random.choice(["Engine", "Server"])
            test_suits = [
                (1, 1, DEFAULT_SMALL_MODEL_NAME_FOR_TEST, mode, "ci_test"),
            ]
        else:
            test_suits = [
                (1, 1, DEFAULT_SMALL_MODEL_NAME_FOR_TEST, "Engine", "engine_test"),
                (1, 1, DEFAULT_SMALL_MODEL_NAME_FOR_TEST, "Sever", "server_test"),
            ]

        truncate_size = 10
        checking_parameters = [
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
        ]

        for tp_size, dp_size, model_name, backend, dst_instance_id in test_suits:
            test_load_weights_from_remote_instance(
                tp_size,
                dp_size,
                model_name,
                backend,
                truncate_size,
                checking_parameters,
                "instance://127.0.0.1:60000",
                DEFAULT_URL_FOR_TEST,
                dst_instance_id,
            )

if __name__ == "__main__":
    unittest.main()
