"""Test loading weights from remote instance
Backend: transfer_engine
"""

import gc
import os
import unittest

import numpy as np
import requests
import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.utils import terminate_process

# Force NPU multiprocessing start method
mp.set_start_method("spawn", force=True)

register_npu_ci(est_time=300, suite="nightly-2-npu-a3", nightly=True)


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
    checking_parameters,
    seed_instance_ip,
    seed_instance_service_port,
    seed_instance_group_base_port,
    event_seed_ready,
    event_dst_ready_list,
    remote_instance_loader_backend,
):

    torch.npu.set_device(rank)

    if rank == 0:
        init_process_seed(
            rank,
            param_queue,
            truncate_size,
            model_name,
            checking_parameters,
            tp_size,
            event_seed_ready,
            event_dst_ready_list,
        )
    elif rank in [1, 2]:
        init_process_dst(
            rank,
            param_queue,
            truncate_size,
            model_name,
            seed_instance_ip,
            seed_instance_service_port,
            seed_instance_group_base_port,
            checking_parameters,
            tp_size,
            event_seed_ready,
            event_dst_ready_list,
            remote_instance_loader_backend,
        )


def init_process_seed(
    rank,
    param_queue,
    truncate_size,
    model_name,
    checking_parameters,
    tp_size,
    event_seed_ready,
    event_dst_ready_list,
):
    # Key NPU environment variables
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    torch.npu.set_device(rank)
    torch.npu.synchronize()

    url = DEFAULT_URL_FOR_TEST
    # Launch seed instance server
    process = popen_launch_server(
        model_name,
        url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=(
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--base-gpu-id",
            str(rank),
            "--tp-size",
            str(tp_size),
        ),
    )
    torch.npu.synchronize()

    # Get weights from seed instance
    seed_params = []
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

    # Notify all destination instances that seed is ready
    event_seed_ready.set()
    # Wait for all destination instances to finish
    for i in range(len(event_dst_ready_list)):
        event_dst_ready_list[i].wait()
    terminate_process(process)


def init_process_dst(
    rank,
    param_queue,
    truncate_size,
    model_name,
    seed_instance_ip,
    seed_instance_service_port,
    seed_instance_group_base_port,
    checking_parameters,
    tp_size,
    event_seed_ready,
    event_dst_ready_list,
    remote_instance_loader_backend,
):
    """Initialize destination instance and load weights remotely from seed instance."""
    torch.npu.set_device(rank * tp_size)
    torch.npu.synchronize()
    base_gpu_id = rank * tp_size

    event_seed_ready.wait()
    for i in range(rank - 1):
        event_dst_ready_list[i].wait()

    ports = []
    for i in range(tp_size):
        ports.append(seed_instance_group_base_port + (rank - 1) * tp_size + i)

    host, _, port = DEFAULT_URL_FOR_TEST.rpartition(":")
    url = ":".join([host, str(int(port) + 10000 + rank)])

    print(f"[sgl] rank {rank} init server on url: {url}")
    process = popen_launch_server(
        model_name,
        url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=(
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--base-gpu-id",
            str(base_gpu_id),
            "--tp-size",
            str(tp_size),
            "--cuda-graph-max-bs",
            2,
            "--tokenizer-path",
            model_name,
            "--remote-instance-weight-loader-seed-instance-ip",
            seed_instance_ip,
            "--remote-instance-weight-loader-seed-instance-service-port",
            seed_instance_service_port,
            "--remote-instance-weight-loader-send-weights-group-ports",
            f"[{','.join(str(port) for port in ports)}]",
            "--load-format",
            "auto",
            "--remote-instance-weight-loader-backend",
            remote_instance_loader_backend,
            "--weight-loader-disable-mmap",
        ),
    )
    torch.npu.synchronize()

    # Mark current destination instance as ready
    event_dst_ready_list[rank - 1].set()

    # Get weights from destination instance after remote loading
    dst_params = []
    for parameter_name in checking_parameters:
        dst_params.append(
            requests.get(
                f"{url}/get_weights_by_name",
                json={"name": parameter_name, "truncate_size": truncate_size},
            ).json()
        )

    param_queue.put((f"sgl_dp_{rank}_dst_params", dst_params))

    terminate_process(process)


def test_load_weights_from_remote_instance(
    tp_size,
    dp_size,
    model_name,
    truncate_size,
    checking_parameters,
    seed_instance_ip,
    seed_instance_service_port,
    seed_instance_group_base_port,
    remote_instance_loader_backend,
):

    print(
        f"Testing model: {model_name} | tp_size: {tp_size} | dp_size: {dp_size} | backend: {remote_instance_loader_backend}"
    )
    param_queue = mp.Queue()
    results = {}
    event_seed_ready = mp.Event()
    event_dst_ready_list = [mp.Event() for _ in range(dp_size)]

    context = mp.spawn(
        init_process,
        args=(
            param_queue,
            truncate_size,
            tp_size,
            model_name,
            checking_parameters,
            seed_instance_ip,
            seed_instance_service_port,
            seed_instance_group_base_port,
            event_seed_ready,
            event_dst_ready_list,
            remote_instance_loader_backend,
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

    # Organize weight data
    params = {
        "seed": results.get("seed_params"),
        "sgl_dp_1_dest": results.get("sgl_dp_1_dst_params"),
    }

    # Core validation: weight consistency between seed and destinations
    for i in range(len(params["seed"])):
        verify_params_close(
            params["seed"][i],
            params["sgl_dp_1_dest"][i],
            f"sgl_dp_1_dst_params rank {i}",
        )

    del context
    param_queue.close()
    param_queue.join_thread()
    gc.collect()
    torch.npu.empty_cache()


class TestLoadWeightsFromRemoteInstance(CustomTestCase):
    """Testcase: Verify weight loading from remote instance using transfer_engine and NPU backend.

    [Test Category] Parameter
    [Test Target] --weight-loader-disable-mmap; --remote-instance-weight-loader-seed-instance-ip; --remote-instance-weight-loader-seed-instance-service-port; --remote-instance-weight-loader-send-weights-group-ports; --remote-instance-weight-loader-backend
    """

    def test_load_weights_from_remote_instance(self):
        """Test remote weight loading with transfer_engine and nccl backends on NPU."""

        # Test cases with different backends
        test_suits = [
            (1, 1, DEFAULT_SMALL_MODEL_NAME_FOR_TEST, "transfer_engine"),
        ]

        # Weight validation configuration
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

        # Run all test cases
        for tp_size, dp_size, model_name, remote_backend in test_suits:
            test_load_weights_from_remote_instance(
                tp_size,
                dp_size,
                model_name,
                truncate_size,
                checking_parameters,
                "127.0.0.1",
                DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1000,
                60010,
                remote_backend,
            )


if __name__ == "__main__":
    unittest.main()
