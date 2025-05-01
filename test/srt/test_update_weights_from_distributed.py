"""Test distributed weight updates.

This test suite simulates a distributed training environment to ensure
correct weight synchronization. On rank 0, the instruct model represents
pre-training weights, and the base model represents post-training weights.
The base model's weights are broadcasted to other ranks using the online
weight update API.

On other ranks, an engine is initialized with the instruct model, and its
parameters are verified against the Hugging Face model. After updating
weights from the distributed system, post-training weights are loaded
and verified again to ensure consistency and accuracy across the
distributed setup.
"""

import gc
import os
import random
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


def verify_params_not_close(params1, params2, error_msg):
    """Verify if two parameter arrays are different enough."""
    assert not np.allclose(np.array(params1), np.array(params2)), error_msg


def init_process(
    rank,
    world_size,
    param_queue,
    truncate_size,
    state_dict_key_to_shape,
    tp_size,
    model_name,
    backend,
    checking_parameters,
    tie_word_embeddings,
):
    torch.cuda.set_device(rank)

    if rank == 0:
        init_process_hf(
            rank,
            world_size,
            param_queue,
            truncate_size,
            model_name,
            checking_parameters,
            tie_word_embeddings,
            state_dict_key_to_shape,
        )
    elif rank in [1, 2]:
        init_process_sgl(
            rank,
            world_size,
            param_queue,
            truncate_size,
            model_name,
            checking_parameters,
            tie_word_embeddings,
            state_dict_key_to_shape,
            backend,
            tp_size,
        )


def init_process_hf(
    rank,
    world_size,
    param_queue,
    truncate_size,
    model_name,
    checking_parameters,
    tie_word_embeddings,
    state_dict_key_to_shape,
):
    # These two environment variables are very important
    # to avoid unexpected behaviors of CUDA and NCCL.
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    # Load model and get parameters
    hf_instruct_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        tie_word_embeddings=tie_word_embeddings,
    ).to("cuda:0")
    base_model_name = model_name.replace("-Instruct", "")
    hf_base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="bfloat16",
        tie_word_embeddings=tie_word_embeddings,
    ).to("cuda:0")

    hf_instruct_params = []
    hf_base_params = []

    print("[hf] get parameter in hf instruct model and base model")
    for parameter_name in checking_parameters:
        hf_instruct_params.append(
            hf_instruct_model.get_parameter(parameter_name)[:truncate_size]
            .cpu()
            .detach()
            .float()
            .numpy()
            .tolist()
        )
        hf_base_params.append(
            hf_base_model.get_parameter(parameter_name)[:truncate_size]
            .cpu()
            .detach()
            .float()
            .numpy()
            .tolist()
        )

    param_queue.put(("hf_instruct_params", hf_instruct_params))
    param_queue.put(("hf_base_params", hf_base_params))

    # Init weight update group for rank 0 (the training engine in RLHF).
    port = 60000 + int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")[0]) * 100
    init_method = f"tcp://localhost:{port}"
    print(f"[hf] {rank=} {world_size=} init custom process group. {init_method=}")
    group = init_custom_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        group_name="test_parameter_update_group",
    )
    dist.barrier(group=group, device_ids=[rank])
    torch.cuda.synchronize()
    time_begin_broadcast = time.time()

    # The last parameter is lm_head.weight, which is tied
    # with embed_tokens.weight. Actually, we only need
    # to broadcast embed_tokens.weight once.
    broadcast_parameters = list(state_dict_key_to_shape.keys())
    if tie_word_embeddings:
        broadcast_parameters.remove("lm_head.weight")

    # Broadcast all the weights from the training
    # engine to other ranks (inference engine).
    for parameter_name in broadcast_parameters:
        torch.distributed.broadcast(
            hf_base_model.get_parameter(parameter_name),
            src=0,
            group=group,
        )
    torch.cuda.synchronize()
    time_end_broadcast = time.time()

    # Measure the latency of broadcasting/weights update.
    broadcast_time = time_end_broadcast - time_begin_broadcast
    print(f"[hf] {rank=} {broadcast_time=:.3f}s")
    param_queue.put(("broadcast_time", broadcast_time))

    # Delete the huggingface models to free up memory.
    del hf_instruct_model
    del hf_base_model
    gc.collect()
    torch.cuda.empty_cache()


def init_process_sgl(
    rank,
    world_size,
    param_queue,
    truncate_size,
    model_name,
    checking_parameters,
    tie_word_embeddings,
    state_dict_key_to_shape,
    backend,
    tp_size,
):
    torch.cuda.set_device(rank)
    torch.cuda.synchronize()
    base_gpu_id = 1 if rank == 1 else 1 + tp_size
    if backend == "Engine":
        print(f"[sgl] rank {rank} init engine")
        engine = sgl.Engine(
            model_path=model_name,
            base_gpu_id=base_gpu_id,
            tp_size=tp_size,
            cuda_graph_max_bs=2,
        )
    else:
        if rank == 1:
            url = DEFAULT_URL_FOR_TEST
        else:
            host, port = DEFAULT_URL_FOR_TEST.split(":")
            url = ":".join(host, str(int(port) + 10000))

        print(f"[sgl] rank {rank} init server on url: {url}")
        process = popen_launch_server(
            model_name,
            url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--base-gpu-id",
                str(base_gpu_id),
                "--tp-size",
                str(tp_size),
                "--cuda-graph-max-bs",
                2,
            ),
        )
    torch.cuda.synchronize()

    # Get weights of instruct model, i.e. pre-training weights.
    instruct_params = []
    for parameter_name in checking_parameters:
        instruct_params.append(
            engine.get_weights_by_name(parameter_name, truncate_size)
            if backend == "Engine"
            else requests.get(
                f"{url}/get_weights_by_name",
                json={"name": parameter_name, "truncate_size": truncate_size},
            ).json()
        )

    param_queue.put((f"sgl_dp_{rank}_instruct_params", instruct_params))

    port = 60000 + int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")[0]) * 100

    # Init weight update group with the training engine.
    if backend == "Engine":
        engine.init_weights_update_group(
            master_address="localhost",
            master_port=str(port),
            rank_offset=base_gpu_id,
            world_size=world_size,
            group_name="test_parameter_update_group",
            backend="nccl",
        )
    else:
        requests.post(
            f"{url}/init_weights_update_group",
            json={
                "master_address": "localhost",
                "master_port": str(port),
                "rank_offset": base_gpu_id,
                "world_size": world_size,
                "group_name": "test_parameter_update_group",
                "backend": "nccl",
            },
        )

    torch.cuda.synchronize()
    time_begin_update = time.time()

    # The last parameter is lm_head.weight, which is tied
    # with embed_tokens.weight. Actually, we only need
    # to update embed_tokens.weight once.
    tie_word_embeddings = (
        True if model_name == DEFAULT_SMALL_MODEL_NAME_FOR_TEST else False
    )
    update_parameters = list(state_dict_key_to_shape.keys())
    if tie_word_embeddings:
        update_parameters.remove("lm_head.weight")

    # Get weights from the training engine and update the inference engine.
    for parameter_name in update_parameters:
        if backend == "Engine":
            engine.update_weights_from_distributed(
                parameter_name,
                dtype=torch.bfloat16,
                shape=state_dict_key_to_shape[parameter_name],
            )
        else:
            requests.post(
                f"{url}/update_weights_from_distributed",
                json={
                    "name": parameter_name,
                    "dtype": "bfloat16",
                    "shape": state_dict_key_to_shape[parameter_name],
                },
            )
    torch.cuda.synchronize()
    time_end_update = time.time()

    # Measure the latency of broadcast/weights update.
    update_time = time_end_update - time_begin_update
    print(
        f"[sgl] fully update model_name {model_name} rank {rank} parameter from distributed time: {update_time:.3f}s"
    )
    param_queue.put((f"update_sgl_dp_{rank}_time", update_time))

    # Get the weights of post-training model after weights update for correctness check.
    base_params = []
    for parameter_name in checking_parameters:
        if backend == "Engine":
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

    # Shutdown the engine or terminate the server process.
    if backend == "Engine":
        engine.shutdown()
    else:
        terminate_process(process)


def assert_tied_weights(params_list, message, should_be_tied):
    for params in params_list:
        if should_be_tied:
            assert np.allclose(params[0], params[-1]), message
        else:
            assert not np.allclose(params[0], params[-1]), message


def test_update_weights_from_distributed(
    tp_size,
    dp_size,
    model_name,
    backend,
    state_dict_key_to_shape,
    truncate_size,
    checking_parameters,
):
    tie_word_embeddings = (
        True if model_name == DEFAULT_SMALL_MODEL_NAME_FOR_TEST else False
    )

    print(
        f"Testing model: {model_name} tp_size: {tp_size}, dp_size: {dp_size} backend: {backend}"
    )
    param_queue = mp.Queue()
    results = {}

    context = mp.spawn(
        init_process,
        args=(
            1 + tp_size * dp_size,
            param_queue,
            truncate_size,
            state_dict_key_to_shape,
            tp_size,
            model_name,
            backend,
            checking_parameters,
            tie_word_embeddings,
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

    params = {
        "hf_instruct": results.get("hf_instruct_params"),
        "hf_base": results.get("hf_base_params"),
        "sgl_dp_1_instruct": results.get("sgl_dp_1_instruct_params"),
        "sgl_dp_1_base": results.get("sgl_dp_1_base_params"),
        "broadcast_time": results.get("broadcast_time"),
        "update_sgl_dp_1_time": results.get("update_sgl_dp_1_time"),
    }

    if dp_size == 2:
        dp2_params = {
            "sgl_dp_2_instruct": results.get("sgl_dp_2_instruct_params"),
            "sgl_dp_2_base": results.get("sgl_dp_2_base_params"),
            "update_sgl_dp_2_time": results.get("update_sgl_dp_2_time"),
        }
        assert all(v is not None for v in dp2_params.values())
        params.update(dp2_params)

    # Check the correctness of weights update by verifying
    # the weights of instruct model and base model.
    for i in range(len(params["hf_instruct"])):
        verify_params_close(
            params["hf_instruct"][i],
            params["sgl_dp_1_instruct"][i],
            f"sgl_dp_1_instruct_params rank {i}",
        )

        verify_params_close(
            params["hf_base"][i],
            params["sgl_dp_1_base"][i],
            f"sgl_dp_1_base_params rank {i}",
        )

        verify_params_not_close(
            params["hf_instruct"][i],
            params["hf_base"][i],
            f"hf_instruct_params rank {i}",
        )

        if dp_size == 2:
            verify_params_close(
                params["hf_base"][i],
                params["sgl_dp_2_base"][i],
                f"sgl_dp_2_base_params rank {i}",
            )
            verify_params_close(
                params["hf_instruct"][i],
                params["sgl_dp_2_instruct"][i],
                f"sgl_dp_2_instruct_params rank {i}",
            )

    assert len(params["hf_instruct"]) == len(
        params["hf_base"]
    ), "hf_instruct_params and hf_base_params have different lengths"

    # Check if the weights of lm_head are tied with embed_tokens.
    params_to_check = [
        (
            params["hf_instruct"],
            "lm_head.weight is not tied with embed_tokens.weight",
        ),
        (
            params["hf_base"],
            "lm_head.weight is not tied with embed_tokens.weight",
        ),
        (
            params["sgl_dp_1_instruct"],
            "lm_head.weight is not tied with embed_tokens.weight",
        ),
        (
            params["sgl_dp_1_base"],
            "lm_head.weight is not tied with embed_tokens.weight",
        ),
    ]

    if dp_size == 2:
        params_to_check.extend(
            [
                (
                    params["sgl_dp_2_instruct"],
                    "lm_head.weight is not tied with embed_tokens.weight",
                ),
                (
                    params["sgl_dp_2_base"],
                    "lm_head.weight is not tied with embed_tokens.weight",
                ),
            ]
        )

    assert_tied_weights(
        [params for params, _ in params_to_check],
        (
            "lm_head.weight is not tied with embed_tokens.weight"
            if tie_word_embeddings
            else "lm_head.weight is tied with embed_tokens.weight"
        ),
        tie_word_embeddings,
    )

    # Time limit for broadcast and update on CI is 3 / 6
    # On local H100, it's 1 / 2
    time_limit = 3 if model_name == DEFAULT_SMALL_MODEL_NAME_FOR_TEST else 6

    assert (
        params["broadcast_time"] < time_limit
    ), f"broadcast_time exceeds time limit {time_limit}s"

    assert (
        params["update_sgl_dp_1_time"] < time_limit
    ), f"update_sgl_dp_one_time exceeds time limit {time_limit}s"

    if dp_size == 2:
        assert (
            params["update_sgl_dp_2_time"] < time_limit
        ), f"update_sgl_dp_two_time exceeds time limit {time_limit}s"

    # Delete the context and close the parameter queue.
    del context
    param_queue.close()
    param_queue.join_thread()
    gc.collect()
    torch.cuda.empty_cache()


class TestUpdateWeightsFromDistributed(CustomTestCase):

    def test_update_weights_from_distributed(self):

        assert torch.cuda.device_count() >= 2, "At least 2 GPUs are required"
        # test_suits : tp, dp, model_name, backend
        if is_in_ci():
            mode = random.choice(["Engine", "Server"])
            test_suits = [
                (1, 1, DEFAULT_SMALL_MODEL_NAME_FOR_TEST, mode),
            ]
        else:
            test_suits = [
                (1, 1, DEFAULT_SMALL_MODEL_NAME_FOR_TEST, "Engine"),
                (1, 1, DEFAULT_MODEL_NAME_FOR_TEST, "Sever"),
            ]

            if torch.cuda.device_count() >= 4:
                test_suits.extend(
                    [
                        (2, 1, DEFAULT_SMALL_MODEL_NAME_FOR_TEST, "Engine"),
                        (1, 2, DEFAULT_MODEL_NAME_FOR_TEST, "Server"),
                    ]
                )

            if torch.cuda.device_count() >= 5:
                test_suits.extend(
                    [
                        (2, 2, DEFAULT_SMALL_MODEL_NAME_FOR_TEST, "Engine"),
                        (2, 2, DEFAULT_MODEL_NAME_FOR_TEST, "Server"),
                    ]
                )

        model_state_dict_shapes = {}
        test_models = [test_suit[2] for test_suit in test_suits]

        for model_name in test_models:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="bfloat16"
            ).to("cuda:0")
            state_dict = model.state_dict()
            state_dict_keys = list(state_dict.keys())
            model_state_dict_shapes[model_name] = {
                key: state_dict[key].shape for key in state_dict_keys
            }
            del model
            gc.collect()
            torch.cuda.empty_cache()

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
            "lm_head.weight",
        ]

        for tp_size, dp_size, model_name, backend in test_suits:
            test_update_weights_from_distributed(
                tp_size,
                dp_size,
                model_name,
                backend,
                model_state_dict_shapes[model_name],
                truncate_size,
                checking_parameters,
            )


if __name__ == "__main__":
    unittest.main()
