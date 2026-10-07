"""Manual XPU distributed refit coverage using the Llama-3.2-1B model pair.

Run from the repository root, selecting a device tier that fits the visible cards:
	ZE_AFFINITY_MASK=2,3 python -m pytest \
		test/manual/rl/test_xpu_rl.py -k tp1dp1 -vv

The refit harness below, an XPU copy of the registered
test_update_weights_from_distributed.py helpers, supplies the full-model transfer,
sampled weight checks, tied-weight checks, and original three-second timing assertions.
Its server flow waits for generation before pausing, so these cases do not prove live
refitting.
"""

import gc
import json
import random
import shutil
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import requests
import torch
import torch.multiprocessing as mp
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

import sglang as sgl
from sglang.srt.configs.model_config import get_num_indexer_layers
from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.state_capturer.indexer_topk import extract_indexer_topk_from_meta_info
from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)
from sglang.srt.utils import (
    get_device_count,
    init_custom_process_group,
    is_xpu,
    kill_process_tree,
)
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)
from sglang.utils import terminate_process

# Ranks must be spawned, not forked: a forked child cannot reinitialize the device runtime.
mp.set_start_method("spawn", force=True)


# ===== Distributed refit harness =====
# XPU-only copy of test/registered/rl/test_update_weights_from_distributed.py's helpers, so the
# registered test stays untouched. Differences: XPU devices and XCCL, attention_backend, and
# free ports chosen in the parent instead of fixed ones.


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
    load_format,
    barrier,
    pause_generation_mode,
    attention_backend,
    ports,
):
    torch.xpu.set_device(rank)

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
            load_format,
            barrier,
            ports["group"],
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
            load_format,
            barrier,
            pause_generation_mode,
            attention_backend,
            ports,
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
    load_format,
    barrier,
    group_port,
):
    # Load model and get parameters
    hf_instruct_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        tie_word_embeddings=tie_word_embeddings,
    ).to("xpu:0")
    base_model_name = model_name.replace("-Instruct", "")
    hf_base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="bfloat16",
        tie_word_embeddings=tie_word_embeddings,
    ).to("xpu:0")

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
    init_method = f"tcp://localhost:{group_port}"
    print(f"[hf] {rank=} {world_size=} init custom process group. {init_method=}")
    group = init_custom_process_group(
        backend="xccl",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        group_name="test_parameter_update_group",
    )
    torch.xpu.synchronize()
    barrier.wait()

    time_begin_broadcast = time.perf_counter()

    # The last parameter is lm_head.weight, which is tied
    # with embed_tokens.weight. Actually, we only need
    # to broadcast embed_tokens.weight once.
    broadcast_parameters = list(state_dict_key_to_shape.keys())
    if tie_word_embeddings:
        broadcast_parameters.remove("lm_head.weight")

    if load_format == "flattened_bucket":
        named_tensors = [
            (parameter_name, hf_base_model.get_parameter(parameter_name))
            for parameter_name in broadcast_parameters
        ]
        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flattened_tensor = bucket.get_flattened_tensor()
        torch.distributed.broadcast(flattened_tensor, src=0, group=group)
    else:
        # Broadcast all the weights from the training
        # engine to other ranks (inference engine).
        for parameter_name in broadcast_parameters:
            torch.distributed.broadcast(
                hf_base_model.get_parameter(parameter_name),
                src=0,
                group=group,
            )
    torch.xpu.synchronize()
    time_end_broadcast = time.perf_counter()

    # Measure the latency of broadcasting/weights update.
    broadcast_time = time_end_broadcast - time_begin_broadcast
    print(f"[hf] {rank=} {broadcast_time=:.3f}s")
    param_queue.put(("broadcast_time", broadcast_time))

    # Destroy process group and release related resource
    torch.distributed.destroy_process_group(group)

    # Delete the huggingface models to free up memory.
    del hf_instruct_model
    del hf_base_model
    gc.collect()
    torch.xpu.empty_cache()


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
    load_format,
    barrier,
    pause_generation_mode,
    attention_backend,
    ports,
):
    torch.xpu.set_device(rank)
    torch.xpu.synchronize()
    base_gpu_id = 1 if rank == 1 else 1 + tp_size
    if backend == "Engine":
        print(f"[sgl] rank {rank} init engine")
        engine = sgl.Engine(
            model_path=model_name,
            base_gpu_id=base_gpu_id,
            tp_size=tp_size,
            attention_backend=attention_backend,
        )
    else:
        url = f"http://127.0.0.1:{ports[rank]}"

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
                *(
                    ("--attention-backend", attention_backend)
                    if attention_backend is not None
                    else ()
                ),
            ),
        )
    torch.xpu.synchronize()

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

    # Init weight update group with the training engine.
    if backend == "Engine":
        engine.init_weights_update_group(
            master_address="localhost",
            master_port=str(ports["group"]),
            rank_offset=base_gpu_id,
            world_size=world_size,
            group_name="test_parameter_update_group",
            backend="xccl",
        )
    else:
        requests.post(
            f"{url}/init_weights_update_group",
            json={
                "master_address": "localhost",
                "master_port": str(ports["group"]),
                "rank_offset": base_gpu_id,
                "world_size": world_size,
                "group_name": "test_parameter_update_group",
                "backend": "xccl",
            },
        )

    if pause_generation_mode in ["in_place", "retract"]:

        def run_decode(max_new_tokens=32):
            response = requests.post(
                url + "/generate",
                json={
                    "text": f"Question: {random.randint(0, 100)},The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": True,
                    },
                },
            )
            return response.json()

        with ThreadPoolExecutor(32) as executor:
            for _ in range(32):
                executor.submit(run_decode, 1000)
            time.sleep(2)

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
    names = [parameter_name for parameter_name in update_parameters]
    dtypes = [torch.bfloat16 if backend == "Engine" else "bfloat16"] * len(names)
    shapes = [state_dict_key_to_shape[parameter_name] for parameter_name in names]

    if pause_generation_mode in ["in_place", "retract"]:
        requests.post(
            url + "/pause_generation",
            json={"mode": pause_generation_mode},
        )
    torch.xpu.synchronize()
    barrier.wait()

    time_begin_update = time.perf_counter()
    if backend == "Engine":
        engine.begin_weight_update()
        engine.update_weights_from_distributed(
            names,
            dtypes=dtypes,
            shapes=shapes,
            group_name="test_parameter_update_group",
            load_format=load_format,
        )
        engine.end_weight_update()
    else:
        requests.post(f"{url}/begin_weight_update", json={})
        requests.post(
            f"{url}/update_weights_from_distributed",
            json={
                "names": names,
                "dtypes": dtypes,
                "shapes": shapes,
                "group_name": "test_parameter_update_group",
                "load_format": load_format,
                "flush_cache": not (pause_generation_mode == "in_place"),
            },
        )
        requests.post(f"{url}/end_weight_update", json={})
    torch.xpu.synchronize()
    time_end_update = time.perf_counter()
    if pause_generation_mode in ["in_place", "retract"]:
        requests.post(
            url + "/continue_generation",
            json={},
        )

        # discard unfinished requests to save test overhead
        time.sleep(2)
        requests.post(
            url + "/pause_generation",
            json={"mode": "abort"},
        )
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

    if backend == "Engine":
        success, _ = engine.destroy_weights_update_group(
            group_name="test_parameter_update_group",
        )
        assert success is True
    else:
        response = requests.post(
            f"{url}/destroy_weights_update_group",
            json={
                "group_name": "test_parameter_update_group",
            },
        )
        assert response.status_code == 200

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


# Chosen once in the parent so every spawned rank agrees. Fixed ports collide on shared hosts,
# where a foreign server on the port answers our health check and receives our updates.
def _pick_free_ports(count):
    ports = []
    while len(ports) < count:
        port = find_available_port(20000 + 2000 * len(ports))
        if port not in ports:
            ports.append(port)
    return ports


def _run_distributed_refit(
    tp_size,
    dp_size,
    model_name,
    backend,
    state_dict_key_to_shape,
    truncate_size,
    checking_parameters,
    load_format=None,
    pause_generation_mode=None,
    attention_backend=None,
):
    tie_word_embeddings = (
        True if model_name == DEFAULT_SMALL_MODEL_NAME_FOR_TEST else False
    )

    print(
        f"Testing model: {model_name} tp_size: {tp_size}, dp_size: {dp_size} backend: {backend}"
    )
    param_queue = mp.Queue()
    results = {}
    barrier = mp.Barrier(1 + dp_size)
    group_port, *server_ports = _pick_free_ports(1 + dp_size)
    ports = {"group": group_port, **dict(zip((1, 2), server_ports))}

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
            load_format,
            barrier,
            pause_generation_mode,
            attention_backend,
            ports,
        ),
        nprocs=1 + dp_size,
        join=False,
    )

    while len(results) < 3 * (1 + dp_size):
        try:
            key, value = param_queue.get(timeout=5)
            results[key] = value
        except Exception:
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

    assert len(params["hf_instruct"]) == len(params["hf_base"]), (
        "hf_instruct_params and hf_base_params have different lengths"
    )

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

    assert params["broadcast_time"] < time_limit, (
        f"broadcast_time exceeds time limit {time_limit}s"
    )

    assert params["update_sgl_dp_1_time"] < time_limit, (
        f"update_sgl_dp_one_time exceeds time limit {time_limit}s"
    )

    if dp_size == 2:
        assert params["update_sgl_dp_2_time"] < time_limit, (
            f"update_sgl_dp_two_time exceeds time limit {time_limit}s"
        )

    # Delete the context and close the parameter queue.
    del context
    param_queue.close()
    param_queue.join_thread()
    gc.collect()
    torch.xpu.empty_cache()


@unittest.skipUnless(is_xpu(), "XPU-only distributed weight-update tests")
class TestRefitFromDistributedXPU(CustomTestCase):
    """Explicit Engine/Server cases; each needs 1 + TP * DP visible devices."""

    @classmethod
    def setUpClass(cls):
        if get_device_count() < 2:
            raise unittest.SkipTest("At least two XPU devices are required")

        cls.model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        model = AutoModelForCausalLM.from_pretrained(
            cls.model_path, torch_dtype="bfloat16"
        )
        state_dict = model.state_dict()
        cls.shapes = {name: tensor.shape for name, tensor in state_dict.items()}
        del state_dict
        del model
        gc.collect()

    def _run(self, tp_size, dp_size, backend, load_format=None, pause_mode=None):
        required = 1 + tp_size * dp_size
        available = get_device_count()
        if available < required:
            self.skipTest(f"{required} XPU devices required, {available} visible")

        _run_distributed_refit(
            tp_size=tp_size,
            dp_size=dp_size,
            model_name=self.model_path,
            backend=backend,
            state_dict_key_to_shape=self.shapes,
            truncate_size=10,
            checking_parameters=[
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
            ],
            load_format=load_format,
            pause_generation_mode=pause_mode,
            attention_backend="intel_xpu",
        )

    def test_tp1dp1_engine_default(self):
        self._run(1, 1, "Engine")

    def test_tp1dp1_engine_flattened_bucket(self):
        self._run(1, 1, "Engine", load_format="flattened_bucket")

    def test_tp1dp1_server_in_place_default(self):
        self._run(1, 1, "Server", pause_mode="in_place")

    def test_tp1dp1_server_in_place_flattened_bucket(self):
        self._run(1, 1, "Server", load_format="flattened_bucket", pause_mode="in_place")

    def test_tp1dp1_server_retract_default(self):
        self._run(1, 1, "Server", pause_mode="retract")

    def test_tp1dp1_server_retract_flattened_bucket(self):
        self._run(1, 1, "Server", load_format="flattened_bucket", pause_mode="retract")

    def test_tp1dp2_engine_default(self):
        self._run(1, 2, "Engine")

    def test_tp1dp2_engine_flattened_bucket(self):
        self._run(1, 2, "Engine", load_format="flattened_bucket")

    def test_tp1dp2_server_in_place_default(self):
        self._run(1, 2, "Server", pause_mode="in_place")

    def test_tp1dp2_server_in_place_flattened_bucket(self):
        self._run(1, 2, "Server", load_format="flattened_bucket", pause_mode="in_place")

    def test_tp1dp2_server_retract_default(self):
        self._run(1, 2, "Server", pause_mode="retract")

    def test_tp1dp2_server_retract_flattened_bucket(self):
        self._run(1, 2, "Server", load_format="flattened_bucket", pause_mode="retract")

    def test_tp2dp1_engine_default(self):
        self._run(2, 1, "Engine")

    def test_tp2dp1_engine_flattened_bucket(self):
        self._run(2, 1, "Engine", load_format="flattened_bucket")

    def test_tp2dp1_server_in_place_default(self):
        self._run(2, 1, "Server", pause_mode="in_place")

    def test_tp2dp1_server_in_place_flattened_bucket(self):
        self._run(2, 1, "Server", load_format="flattened_bucket", pause_mode="in_place")

    def test_tp2dp1_server_retract_default(self):
        self._run(2, 1, "Server", pause_mode="retract")

    def test_tp2dp1_server_retract_flattened_bucket(self):
        self._run(2, 1, "Server", load_format="flattened_bucket", pause_mode="retract")

    def test_tp2dp2_engine_default(self):
        self._run(2, 2, "Engine")

    def test_tp2dp2_engine_flattened_bucket(self):
        self._run(2, 2, "Engine", load_format="flattened_bucket")

    def test_tp2dp2_server_in_place_default(self):
        self._run(2, 2, "Server", pause_mode="in_place")

    def test_tp2dp2_server_in_place_flattened_bucket(self):
        self._run(2, 2, "Server", load_format="flattened_bucket", pause_mode="in_place")

    def test_tp2dp2_server_retract_default(self):
        self._run(2, 2, "Server", pause_mode="retract")

    def test_tp2dp2_server_retract_flattened_bucket(self):
        self._run(2, 2, "Server", load_format="flattened_bucket", pause_mode="retract")


@unittest.skipUnless(is_xpu(), "XPU-only disk weight-update tests")
class TestUpdateWeightsFromDisk(CustomTestCase):
    """BF16 TP1/TP2 refits, memory handoff, and failed-update integrity.

    Run with ZE_AFFINITY_MASK=2,3 python -m pytest
    test/manual/rl/test_xpu_rl.py::TestUpdateWeightsFromDisk -vv.
    Requires the Llama-3.2-1B Instruct/base pair and XPU torch_memory_saver.
    Idle pause checks do not establish live partial-rollout correctness.
    """

    backend_test_suites = ({"tp_size": 1}, {"tp_size": 2})
    parameter_names = (
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "lm_head.weight",
    )

    # Request/decode helpers and the offload-refit-resume test, copied from the registered
    # CUDA base in test_update_weights_from_disk_blackwell.py so this file imports no CUDA test.
    request_timeout = 120
    update_timeout = 120
    idle_timeout = 30
    decode_payload = {
        "text": "The capital of France is",
        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
    }
    update_test_suites = (
        {"flush_cache": True, "abort_all_requests": False},
        {"flush_cache": False, "abort_all_requests": False},
    )

    def _get_json(self, endpoint, timeout=None):
        response = requests.get(
            f"{self.base_url}{endpoint}",
            timeout=timeout or self.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    def _post_json(self, endpoint, payload, timeout=None):
        response = requests.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            timeout=timeout or self.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    def _run_decode(self):
        return self._post_json("/generate", self.decode_payload)["text"]

    def _wait_until_idle(self):
        deadline = time.monotonic() + self.idle_timeout
        last_loads = None
        while time.monotonic() < deadline:
            last_loads = self._get_json("/v1/loads?include=core")["loads"]
            if last_loads and all(
                load["num_running_reqs"] == 0 and load["num_waiting_reqs"] == 0
                for load in last_loads
            ):
                return
            time.sleep(0.1)
        self.fail(f"Server did not become idle before weight update: {last_loads=}")

    def _assert_non_empty_decode(self):
        self.assertTrue(len(self._run_decode()) > 0)

    def _get_decode_logprob_signature(self):
        ret = self._post_json(
            "/generate",
            {**self.decode_payload, "return_logprob": True},
        )
        output_token_logprobs = ret["meta_info"].get("output_token_logprobs")
        self.assertIsNotNone(output_token_logprobs)
        self.assertGreater(
            len(output_token_logprobs),
            0,
            "Expected non-empty output_token_logprobs.",
        )
        return {
            "text": ret["text"],
            "token_ids": [int(x[1]) for x in output_token_logprobs],
            "logprobs": [float(x[0]) for x in output_token_logprobs],
        }

    def _assert_decode_logprob_unchanged(self, before, after, atol=1e-4):
        self.assertEqual(after["text"], before["text"])
        self.assertEqual(after["token_ids"], before["token_ids"])
        self.assertEqual(len(after["logprobs"]), len(before["logprobs"]))
        for idx, (a, b) in enumerate(zip(after["logprobs"], before["logprobs"])):
            self.assertLessEqual(
                abs(a - b),
                atol,
                f"Output token logprob changed at idx={idx}: before={b}, after={a}",
            )

    def _get_model_info(self):
        return self._get_json("/get_model_info")["model_path"]

    def _run_update_weights(
        self,
        model_path,
        flush_cache=True,
        abort_all_requests=False,
    ):
        return self._post_json(
            "/update_weights_from_disk",
            {
                "model_path": model_path,
                "flush_cache": flush_cache,
                "abort_all_requests": abort_all_requests,
            },
            timeout=self.update_timeout,
        )

    def _offload_engine_and_resume_weights(self):
        self._post_json("/release_memory_occupation", {})
        self._post_json(
            "/resume_memory_occupation", {"tags": [GPU_MEMORY_TYPE_WEIGHTS]}
        )

    def _resume_kv_cache_and_graph_memory(self):
        # "cuda_graph" is SGLang's platform-generic graph-memory tag; release frees it on XPU too.
        self._post_json(
            "/resume_memory_occupation",
            {"tags": [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]},
        )

    def test_parameterized_update_weights_from_disk(self):
        for backend_test_suite in self.backend_test_suites:
            case_name = backend_test_suite.get("name", "default")
            with self.subTest(model=self.model, case_name=case_name):
                process = self._launch_server(backend_test_suite)
                try:
                    origin_model_path = self._get_model_info()
                    self.assertEqual(origin_model_path, self.model)
                    self._assert_non_empty_decode()
                    baseline_sig = self._get_decode_logprob_signature()

                    for update_test_suite in self.update_test_suites:
                        with self.subTest(case_name=case_name, **update_test_suite):
                            self._wait_until_idle()
                            self._offload_engine_and_resume_weights()
                            ret = self._run_update_weights(
                                self.model,
                                flush_cache=update_test_suite["flush_cache"],
                                abort_all_requests=update_test_suite[
                                    "abort_all_requests"
                                ],
                            )
                            self._resume_kv_cache_and_graph_memory()
                            self.assertTrue(ret.get("success"), f"{ret=}")
                            self.assertEqual(self._get_model_info(), self.model)
                            self._assert_non_empty_decode()
                            updated_sig = self._get_decode_logprob_signature()
                            self._assert_decode_logprob_unchanged(
                                baseline_sig, updated_sig
                            )
                finally:
                    kill_process_tree(process.pid, wait_timeout=60)

    @classmethod
    def setUpClass(cls):
        cls.model = snapshot_download(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
        cls.base_model = snapshot_download(DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE)
        cls.references = {}
        for model_path in (cls.model, cls.base_model):
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="cpu"
            )
            cls.hidden_size = model.config.hidden_size
            parameters = dict(model.named_parameters(remove_duplicate=False))
            cls.references[model_path] = {
                name: parameters[name].detach().flatten()[:16].float().clone()
                for name in cls.parameter_names
            }
            del parameters, model
            gc.collect()
        super().setUpClass()

    def _launch_server(self, backend_test_suite):
        tp_size = backend_test_suite["tp_size"]
        if get_device_count() < tp_size:
            self.skipTest(f"{tp_size} visible XPU devices required")
        self.base_url = f"http://127.0.0.1:{find_available_port(21000)}"
        return popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--device",
                "xpu",
                "--attention-backend",
                "intel_xpu",
                "--dtype",
                "bfloat16",
                "--tp-size",
                str(tp_size),
                "--enable-memory-saver",
                "--mem-fraction-static",
                "0.7",
                # Batch-invariant decode: without it split-KV attention picks
                # kv splits from the batch, so a refit's cache flush changes the
                # reduction order and the 1e-4 logprob check sees 3e-2 drift.
                "--enable-deterministic-inference",
            ],
        )

    @contextmanager
    def _server(self, tp_size=1):
        process = self._launch_server({"tp_size": tp_size})
        try:
            self.assertEqual(self._get_model_info(), self.model)
            yield
        finally:
            kill_process_tree(process.pid, wait_timeout=60)

    def _assert_weights(self, model_path):
        for name, expected in self.references[model_path].items():
            with self.subTest(parameter=name):
                values = self._post_json(
                    "/get_weights_by_name", {"name": name, "truncate_size": 16}
                )
                actual = torch.tensor(values).flatten()[:16]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def _refit(self, model_path, **options):
        result = self._post_json(
            "/update_weights_from_disk",
            {"model_path": model_path, "flush_cache": True, **options},
            timeout=self.update_timeout,
        )
        self.assertTrue(result.get("success"), result)
        self.assertEqual(self._get_model_info(), model_path)
        self._assert_weights(model_path)

    def _check_changed_checkpoint(self, tp_size):
        self.assertTrue(
            any(
                not torch.equal(
                    self.references[self.model][name],
                    self.references[self.base_model][name],
                )
                for name in self.parameter_names
            ),
            "The two checkpoint references must differ",
        )
        with self._server(tp_size):
            self._assert_weights(self.model)
            baseline = self._get_decode_logprob_signature()
            for step in range(2):
                with self.subTest(step=step):
                    self._wait_until_idle()
                    self._refit(self.base_model, load_format="safetensors")
                    changed = self._get_decode_logprob_signature()
                    self.assertNotEqual(changed, baseline)
                    self._wait_until_idle()
                    self._refit(self.model)
                    self._assert_decode_logprob_unchanged(
                        baseline, self._get_decode_logprob_signature()
                    )

    def test_changed_checkpoint_tp1(self):
        self._check_changed_checkpoint(1)

    def test_changed_checkpoint_tp2(self):
        self._check_changed_checkpoint(2)

    def test_pause_refit_continue(self):
        with self._server():
            baseline = self._get_decode_logprob_signature()
            for mode in ("abort", "in_place", "retract"):
                with self.subTest(mode=mode):
                    self._wait_until_idle()
                    self._post_json("/pause_generation", {"mode": mode})
                    try:
                        self._refit(self.base_model)
                    finally:
                        self._post_json("/continue_generation", {})
                    self._assert_non_empty_decode()
                    self._wait_until_idle()
                    self._refit(self.model)
                    self._assert_decode_logprob_unchanged(
                        baseline, self._get_decode_logprob_signature()
                    )

    def _check_failed_update(self, malformed):
        with tempfile.TemporaryDirectory(prefix="sglang-xpu-invalid-") as directory:
            shutil.copyfile(
                Path(self.model) / "config.json", Path(directory) / "config.json"
            )
            if malformed:
                save_file(
                    {
                        "model.layers.0.input_layernorm.weight": torch.full(
                            (self.hidden_size,),
                            2.0,
                            dtype=torch.bfloat16,
                        ),
                        "model.layers.0.self_attn.q_proj.weight": torch.zeros(1, 1),
                    },
                    str(Path(directory) / "model.safetensors"),
                )
            with self._server():
                baseline = self._get_decode_logprob_signature()
                self._wait_until_idle()
                snapshot = self._post_json("/weights_checker", {"action": "snapshot"})
                self.assertTrue(snapshot.get("success"), snapshot)
                response = requests.post(
                    f"{self.base_url}/update_weights_from_disk",
                    json={"model_path": directory, "flush_cache": True},
                    timeout=self.update_timeout,
                )
                self.assertIn(response.status_code, (200, 400), response.text)
                result = response.json()
                self.assertIs(result.get("success"), False, result)
                self.assertTrue(result.get("message"), result)
                self.assertEqual(self._get_model_info(), self.model)
                comparison = self._post_json("/weights_checker", {"action": "compare"})
                self.assertTrue(comparison.get("success"), comparison)
                self._assert_weights(self.model)
                self._assert_decode_logprob_unchanged(
                    baseline, self._get_decode_logprob_signature()
                )
                self._wait_until_idle()
                self._refit(self.model)

    def test_missing_weights_preserves_model(self):
        self._check_failed_update(malformed=False)

    def test_malformed_weights_rolls_back(self):
        self._check_failed_update(malformed=True)


class _CaptureXPUCase(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not is_xpu() or get_device_count() < cls.required_devices:
            raise unittest.SkipTest(
                f"{cls.required_devices} visible XPU devices required"
            )
        cls.config = AutoConfig.from_pretrained(cls.model, trust_remote_code=True)

    @contextmanager
    def _server(self, extra_args=()):
        self.base_url = f"http://127.0.0.1:{find_available_port(21000)}"
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--device",
                "xpu",
                "--dtype",
                "bfloat16",
                "--tp-size",
                str(self.required_devices),
                "--context-length",
                "4096",
                "--max-total-tokens",
                "8192",
                "--chunked-prefill-size",
                "4096",
                "--max-running-requests",
                "4",
                "--mem-fraction-static",
                "0.8",
                "--random-seed",
                "0",
                *self.server_args,
                *extra_args,
            ],
        )
        try:
            yield
        finally:
            kill_process_tree(process.pid, wait_timeout=60)

    def _post(self, endpoint, payload):
        response = requests.post(self.base_url + endpoint, json=payload, timeout=180)
        response.raise_for_status()
        if endpoint == "/flush_cache":
            return response.text
        body = response.json()
        if isinstance(body, dict):
            self.assertNotIn("error", body, body)
        return body

    def _generate(self, text, capture=True, **extra):
        body = self._post(
            "/generate",
            {
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
                self.request_flag: capture,
                **extra,
            },
        )
        self.assertEqual(body["meta_info"]["completion_tokens"], 8)
        if not capture:
            self.assertNotIn(self.metadata_key, body["meta_info"])
        return body

    def _rows(self, body, start_len=0):
        meta = body["meta_info"]
        return max(0, meta["prompt_tokens"] + meta["completion_tokens"] - 1 - start_len)


class TestReturnRoutedExpertsXPU(_CaptureXPUCase):
    """Pretrained BF16 Qwen3 MoE capture on four XPUs; no DeepEP coverage."""

    model = "Qwen/Qwen3-30B-A3B"
    required_devices = 4
    request_flag = "return_routed_experts"
    metadata_key = "routed_experts"
    server_args = (
        # intel_xpu disables the radix cache under deterministic mode; the cache-hit checks need it.
        "--attention-backend",
        "triton",
        "--enable-return-routed-experts",
        "--enable-deterministic-inference",
    )
    prompt = "Explain why the sky appears blue during the day in one sentence."

    def _experts(self, body, start_len=0):
        flat = extract_routed_experts_from_meta_info(body)
        shape = (
            self._rows(body, start_len),
            self.config.num_hidden_layers,
            self.config.num_experts_per_tok,
        )
        self.assertEqual(flat.dtype, np.dtype("int32"))
        self.assertEqual(flat.size, int(np.prod(shape)))
        experts = flat.reshape(shape)
        self.assertGreater(experts.size, 0)
        self.assertTrue(((experts >= 0) & (experts < self.config.num_experts)).all())
        ordered = np.sort(experts, axis=-1)
        self.assertTrue((np.diff(ordered, axis=-1) > 0).all(), "Duplicate expert IDs")
        return experts

    def test_generate_and_request_flags(self):
        with self._server():
            captured = self._generate(self.prompt)
            self._experts(captured)
            plain = self._generate(self.prompt, capture=False)
            self.assertEqual(captured["output_ids"], plain["output_ids"])
            self.assertEqual(captured["text"], plain["text"])

    def test_start_len_and_cache_hit(self):
        with self._server():
            self._post("/flush_cache", {})
            full = self._generate(self.prompt)
            self.assertEqual(full["meta_info"]["cached_tokens"], 0)
            experts = self._experts(full)
            prompt_tokens = full["meta_info"]["prompt_tokens"]
            for start_len in (0, max(1, prompt_tokens // 2), prompt_tokens):
                with self.subTest(start_len=start_len):
                    cropped = self._generate(
                        self.prompt, routed_experts_start_len=start_len
                    )
                    self.assertEqual(full["output_ids"], cropped["output_ids"])
                    np.testing.assert_array_equal(
                        experts[start_len:], self._experts(cropped, start_len)
                    )
                    if start_len < prompt_tokens - 1:
                        self.assertGreater(
                            cropped["meta_info"]["cached_tokens"], start_len
                        )
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": self.prompt,
                    "return_routed_experts": True,
                    "routed_experts_start_len": prompt_tokens + 1,
                    "sampling_params": {"max_new_tokens": 8},
                },
                timeout=180,
            )
            self.assertIn(response.status_code, (200, 400))
            self.assertIn("is higher than the number of input tokens", response.text)

    def test_concurrent_mixed_requests(self):
        texts = [self.prompt, "Compute 7 plus 12."]
        with self._server():
            isolated = [self._generate(text) for text in texts]
            self._post("/flush_cache", {})
            with ThreadPoolExecutor(max_workers=4) as executor:
                jobs = [
                    (index, capture, executor.submit(self._generate, text, capture))
                    for index, text in enumerate(texts)
                    for capture in (False, True)
                ]
                for index, capture, future in jobs:
                    body = future.result()
                    self.assertEqual(body["output_ids"], isolated[index]["output_ids"])
                    if capture:
                        np.testing.assert_array_equal(
                            self._experts(body), self._experts(isolated[index])
                        )

    def test_openai_endpoints(self):
        with self._server():
            for endpoint, prompt_payload in (
                ("/v1/completions", {"prompt": self.prompt}),
                (
                    "/v1/chat/completions",
                    {"messages": [{"role": "user", "content": self.prompt}]},
                ),
            ):
                with self.subTest(endpoint=endpoint):
                    body = self._post(
                        endpoint,
                        {
                            "model": self.model,
                            **prompt_payload,
                            "temperature": 0,
                            "max_tokens": 8,
                            "return_routed_experts": True,
                        },
                    )
                    self.assertGreater(body["usage"]["completion_tokens"], 0)
                    self._experts(
                        {
                            "meta_info": {
                                **body["usage"],
                                "routed_experts": body["sglext"]["routed_experts"],
                            }
                        }
                    )


class TestReturnIndexerTopkXPU(_CaptureXPUCase):
    """Two-card DSA capture checks using a random fixture, not model-quality tests."""

    model = "yujiepan/glm-moe-dsa-tiny-random"
    required_devices = 2
    request_flag = "return_indexer_topk"
    metadata_key = "indexer_topk"
    server_args = (
        "--trust-remote-code",
        "--dp-size",
        "2",
        "--enable-dp-attention",
        "--enable-return-indexer-topk",
        "--page-size",
        "64",
    )
    prompt = "What is the capital of France?"
    long_prompt = "word " * 2600

    def _indexer_args(self, shared):
        layers = get_num_indexer_layers(self.config)
        self.assertGreaterEqual(layers, 2)
        return (
            "--json-model-override-args",
            json.dumps(
                {
                    "indexer_types": ["full"]
                    + ["shared" if shared else "full"] * (layers - 1),
                    "num_nextn_predict_layers": 0,
                }
            ),
        )

    def _topk(self, body):
        flat = extract_indexer_topk_from_meta_info(body)
        shape = (
            self._rows(body),
            get_num_indexer_layers(self.config),
            self.config.index_topk,
        )
        self.assertEqual(flat.dtype, np.dtype("int32"))
        self.assertEqual(flat.size, int(np.prod(shape)))
        topk = flat.reshape(shape)
        self.assertGreater(topk.size, 0)
        positions = np.arange(shape[0])[:, None, None]
        self.assertTrue(
            ((topk >= -1) & (topk <= positions)).all(), "Noncausal or invalid index"
        )
        expected = np.minimum(np.arange(shape[0]) + 1, shape[2])
        np.testing.assert_array_equal(
            (topk >= 0).sum(axis=-1), np.broadcast_to(expected[:, None], shape[:2])
        )
        ordered = np.sort(topk, axis=-1)
        self.assertFalse(
            ((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)).any()
        )
        return topk

    def test_shared_layers_and_mixed_requests(self):
        with self._server(self._indexer_args(shared=True)):
            with ThreadPoolExecutor(max_workers=3) as executor:
                jobs = [
                    executor.submit(self._generate, text, capture)
                    for text, capture in (
                        (self.long_prompt, True),
                        (self.prompt, True),
                        (self.prompt, False),
                    )
                ]
                long_result, short_result, plain = [job.result() for job in jobs]
            self.assertGreater(
                long_result["meta_info"]["prompt_tokens"], self.config.index_topk
            )
            self.assertEqual(short_result["output_ids"], plain["output_ids"])
            for body in (long_result, short_result):
                topk = self._topk(body)
                for layer in range(1, topk.shape[1]):
                    np.testing.assert_array_equal(topk[:, layer], topk[:, layer - 1])

    def test_fresh_layers_and_saturation(self):
        with self._server(self._indexer_args(shared=False)):
            body = self._generate(self.long_prompt)
            self.assertGreater(
                body["meta_info"]["prompt_tokens"], self.config.index_topk
            )
            topk = self._topk(body)
            selected = np.sort(topk[self.config.index_topk :, :2], axis=-1)
            self.assertFalse(
                np.array_equal(selected[:, 0], selected[:, 1]),
                "Independent fixture layers selected identical position sets",
            )
            self.assertEqual(int((topk[-1, 0] >= 0).sum()), self.config.index_topk)

    def test_cache_hit_and_repeated_requests(self):
        cache_prompt = "word " * 128
        with self._server(self._indexer_args(shared=True)):
            for iteration in range(2):
                with self.subTest(iteration=iteration):
                    self._post("/flush_cache", {})
                    cold = self._generate(cache_prompt, routed_dp_rank=0)
                    warm = self._generate(cache_prompt, routed_dp_rank=0)
                    self.assertGreater(cold["meta_info"]["prompt_tokens"], 64)
                    self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
                    self.assertGreater(warm["meta_info"]["cached_tokens"], 0)
                    self.assertEqual(cold["output_ids"], warm["output_ids"])
                    np.testing.assert_array_equal(
                        np.sort(self._topk(cold), axis=-1),
                        np.sort(self._topk(warm), axis=-1),
                    )


if __name__ == "__main__":
    unittest.main()
