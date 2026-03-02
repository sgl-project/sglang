from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=195, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=195, suite="stage-b-test-small-1-gpu-amd")

import gc
import json
import random
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch
from transformers import AutoConfig

import sglang as sgl
from sglang.srt.utils import MultiprocessingSerializer, kill_process_tree
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MXFP8,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_blackwell_system,
    popen_launch_server,
)


def test_update_weights_from_tensor(tp_size):
    assert torch.cuda.device_count() >= tp_size, f"At least {tp_size} GPUs are required"
    torch.cuda.empty_cache()

    engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST, tp_size=tp_size)

    param_names = [f"model.layers.{i}.mlp.up_proj.weight" for i in range(6, 16)]

    _check_param(engine, param_names[0], [0.0087, -0.0214, -0.0004, 0.0039, 0.0110])

    memory_before = torch.cuda.memory_allocated()
    new_tensor = torch.full((16384, 2048), 1.5, device="cuda")

    time_start = time.perf_counter()
    engine.update_weights_from_tensor([(x, new_tensor) for x in param_names])
    print(f"Time delta: {time.perf_counter() - time_start:.03f}")

    for param_name in param_names[:3]:
        _check_param(engine, param_name, [1.5] * 5)

    engine.shutdown()

    del new_tensor
    gc.collect()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    memory_after = torch.cuda.memory_allocated()
    assert (
        memory_after <= memory_before + 1024
    ), f"Memory leak detected: {memory_after - memory_before} bytes"


class TestUpdateWeightsFromTensor(CustomTestCase):
    def test_update_weights_from_tensor(self):
        tp_sizes = [1, 2]
        for tp_size in tp_sizes:
            if torch.cuda.device_count() < tp_size:
                continue

            with self.subTest(tp_size=tp_size):
                test_update_weights_from_tensor(tp_size)

    def test_update_weights_from_tensor_load_format_direct(self):
        engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        write_param_names = [
            f"model.layers.{i}.self_attn.qkv_proj.weight" for i in range(6, 16)
        ]
        read_param_names = [
            f"model.layers.{i}.self_attn.k_proj.weight" for i in range(6, 16)
        ]

        _check_param(
            engine, read_param_names[0], [-0.0198, 0.0227, 0.0168, 0.0232, -0.0178]
        )

        new_tensor = torch.full((3072, 2048), 1.5)
        engine.update_weights_from_tensor(
            [
                (write_param_name, new_tensor.clone())
                for write_param_name in write_param_names
            ],
            load_format="direct",
        )

        for read_param_name in read_param_names[:3]:
            _check_param(engine, read_param_name, [1.5] * 5)

        engine.shutdown()

    def test_update_weights_from_tensor_load_format_custom(self):
        custom_loader_name = (
            "sglang.srt.model_executor.model_runner._model_load_weights_direct"
        )
        engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            custom_weight_loader=[custom_loader_name],
        )

        write_param_names = [
            f"model.layers.{i}.self_attn.qkv_proj.weight" for i in range(6, 16)
        ]
        read_param_names = [
            f"model.layers.{i}.self_attn.k_proj.weight" for i in range(6, 16)
        ]

        _check_param(
            engine, read_param_names[0], [-0.0198, 0.0227, 0.0168, 0.0232, -0.0178]
        )

        new_tensor = torch.full((3072, 2048), 1.5)
        engine.update_weights_from_tensor(
            [
                (write_param_name, new_tensor.clone())
                for write_param_name in write_param_names
            ],
            load_format=custom_loader_name,
        )

        for read_param_name in read_param_names[:3]:
            _check_param(engine, read_param_name, [1.5] * 5)

        engine.shutdown()

    def test_update_weights_from_tensor_load_format_flattened_bucket(self):
        """Test updating weights using flattened_bucket format"""
        engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        # Create a small set of parameters for testing
        param_names = [f"model.layers.{i}.mlp.up_proj.weight" for i in range(6, 10)]

        # Check original values
        _check_param(engine, param_names[0], [0.0087, -0.0214, -0.0004, 0.0039, 0.0110])

        # Create new tensors with different values
        new_tensors = []
        for _, name in enumerate(param_names):
            # Create tensors with different values for each parameter
            value = 2.0  # Different value for each parameter
            new_tensor = torch.full((16384, 2048), value, device="cuda")
            new_tensors.append((name, new_tensor))

        # Create a flattened bucket
        flattened_bucket = FlattenedTensorBucket(named_tensors=new_tensors)

        # Extract the flattened tensor and metadata in the format expected by model_runner
        flattened_tensor = flattened_bucket.get_flattened_tensor()
        metadata = flattened_bucket.get_metadata()

        # Create the dict format expected by _update_weights_from_flattened_bucket
        bucket_dict = {"flattened_tensor": flattened_tensor, "metadata": metadata}

        # Serialize the bucket data
        from sglang.srt.utils import MultiprocessingSerializer

        serialized_bucket = MultiprocessingSerializer.serialize(
            bucket_dict, output_str=True
        )

        # Create a list where each rank contains the same serialized data
        # This simulates the distributed environment where each rank has the same data
        serialized_bucket_list = [serialized_bucket]

        # Update weights using flattened_bucket format
        time_start = time.perf_counter()
        engine.update_weights_from_tensor(
            named_tensors=serialized_bucket_list, load_format="flattened_bucket"
        )
        update_time = time.perf_counter() - time_start
        print(f"Flattened bucket update time: {update_time:.03f}")

        # Verify the weights were updated correctly
        for i, param_name in enumerate(param_names):
            _check_param(engine, param_name, [2.0] * 5)

        engine.shutdown()


class TestServerUpdateWeightsFromTensorNonBlocking(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--max-running-requests", 8],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self, max_new_tokens=32):
        response = requests.post(
            self.base_url + "/generate",
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

    def get_model_info(self):
        response = requests.get(self.base_url + "/get_model_info")
        model_path = response.json()["model_path"]
        print(json.dumps(response.json()))
        return model_path

    def pause_generation(self, mode):
        response = requests.post(
            self.base_url + "/pause_generation",
            json={"mode": mode},
        )
        ret = response.json()
        return ret

    def continue_generation(self):
        response = requests.post(
            self.base_url + "/continue_generation",
            json={},
        )
        ret = response.json()
        return ret

    def run_update_weights(self, named_tensors, flush_cache=True):
        response = requests.post(
            self.base_url + "/update_weights_from_tensor",
            json={
                "serialized_named_tensors": [
                    MultiprocessingSerializer.serialize(named_tensors, output_str=True)
                ],
                "flush_cache": flush_cache,
            },
        )
        ret = response.json()
        return ret

    def test_update_weights(self):
        pause_generation_modes = ["in_place", "retract"]
        for pause_generation_mode in pause_generation_modes:
            num_requests = 32
            with ThreadPoolExecutor(num_requests) as executor:
                futures = [
                    executor.submit(self.run_decode, 3000) for _ in range(num_requests)
                ]

                # ensure the decode has been started
                time.sleep(2)

                param_names = [
                    f"model.layers.{i}.mlp.up_proj.weight" for i in range(6, 16)
                ]
                new_tensor = torch.full((16384, 2048), 1.5, device="cuda")
                named_tensors = [(x, new_tensor) for x in param_names]

                ret = self.pause_generation(pause_generation_mode)
                ret = self.run_update_weights(
                    named_tensors, flush_cache=pause_generation_mode == "retract"
                )
                self.assertTrue(ret["success"])
                ret = self.continue_generation()

                for future in as_completed(futures):
                    self.assertNotEqual(
                        future.result()["meta_info"]["finish_reason"]["type"], "abort"
                    )

                for param_name in param_names[:3]:
                    response = requests.post(
                        self.base_url + "/get_weights_by_name",
                        json={"name": param_name},
                    )
                    actual_values = torch.tensor(response.json())[0, :5]
                    assert torch.allclose(
                        actual_values, torch.tensor([1.5] * 5), atol=0.002
                    ), f"{actual_values=}"


@unittest.skipIf(not is_blackwell_system(), "MXFP8 requires Blackwell (CUDA)")
class TestServerUpdateWeightsFromTensorMXFP8(CustomTestCase):
    model = DEFAULT_MODEL_NAME_FOR_TEST_MXFP8
    base_url = DEFAULT_URL_FOR_TEST
    backend_test_suites = [
        {"fp8_gemm_backend": "triton", "moe_runner_backend": "cutlass"},
        {"fp8_gemm_backend": "auto", "moe_runner_backend": "auto"},
    ]

    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.model, trust_remote_code=True)
        cls.hidden_size = getattr(
            config,
            "hidden_size",
            getattr(getattr(config, "text_config", None), "hidden_size", None),
        )
        if cls.hidden_size is None:
            raise ValueError("Cannot resolve hidden_size for MXFP8 model config.")

    def _launch_server(self, fp8_gemm_backend, moe_runner_backend):
        return popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--fp8-gemm-backend",
                fp8_gemm_backend,
                "--moe-runner-backend",
                moe_runner_backend,
            ],
        )

    def _run_update_weights(self, serialized_named_tensors, load_format=None):
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "flush_cache": True,
        }
        if load_format is not None:
            payload["load_format"] = load_format
        response = requests.post(
            self.base_url + "/update_weights_from_tensor",
            json=payload,
        )
        ret = response.json()
        print(json.dumps(ret))
        return ret

    def _run_decode(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
        )
        response.raise_for_status()
        return response.json()["text"]

    def test_parameterized_update_weights_from_tensor_mxfp8(self):
        param_name = "model.norm.weight"
        update_test_suites = [
            {"load_format": None, "target_value": 1.125},
            {"load_format": "flattened_bucket", "target_value": 1.25},
        ]

        for backend_test_suite in self.backend_test_suites:
            with self.subTest(**backend_test_suite):
                process = self._launch_server(
                    backend_test_suite["fp8_gemm_backend"],
                    backend_test_suite["moe_runner_backend"],
                )
                try:
                    for update_test_suite in update_test_suites:
                        with self.subTest(
                            fp8_gemm_backend=backend_test_suite["fp8_gemm_backend"],
                            moe_runner_backend=backend_test_suite["moe_runner_backend"],
                            load_format=update_test_suite["load_format"],
                            target_value=update_test_suite["target_value"],
                        ):
                            origin_response = self._run_decode()
                            self.assertTrue(len(origin_response) > 0)

                            new_tensor = torch.full(
                                (self.hidden_size,),
                                update_test_suite["target_value"],
                                device="cuda",
                                dtype=torch.bfloat16,
                            )

                            if update_test_suite["load_format"] == "flattened_bucket":
                                bucket = FlattenedTensorBucket(
                                    named_tensors=[(param_name, new_tensor)]
                                )
                                bucket_dict = {
                                    "flattened_tensor": bucket.get_flattened_tensor(),
                                    "metadata": bucket.get_metadata(),
                                }
                                serialized_named_tensors = [
                                    MultiprocessingSerializer.serialize(
                                        bucket_dict, output_str=True
                                    )
                                ]
                            else:
                                serialized_named_tensors = [
                                    MultiprocessingSerializer.serialize(
                                        [(param_name, new_tensor)], output_str=True
                                    )
                                ]

                            ret = self._run_update_weights(
                                serialized_named_tensors=serialized_named_tensors,
                                load_format=update_test_suite["load_format"],
                            )
                            self.assertTrue(ret["success"])
                            updated_response = self._run_decode()
                            self.assertTrue(len(updated_response) > 0)
                finally:
                    kill_process_tree(process.pid)


def _check_param(engine, param_name, expect_values):
    actual_values = torch.tensor(engine.get_weights_by_name(param_name))[0, :5]
    assert torch.allclose(
        actual_values, torch.tensor(expect_values), atol=0.002
    ), f"{actual_values=}"


if __name__ == "__main__":
    unittest.main()
