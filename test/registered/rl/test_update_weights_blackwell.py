from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=500, suite="stage-b-test-4-gpu-b200")

import os
import random
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
import torch

from sglang.srt.utils import (
    MultiprocessingSerializer,
    init_custom_process_group,
    kill_process_tree,
)
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)


class _BlackwellMXFP8ServerBase(CustomTestCase):
    model = "zianglih/Qwen3-30B-A3B-Instruct-2507-MXFP8-last-8-BF16"
    base_url = DEFAULT_URL_FOR_TEST
    norm_weight_name = "model.norm.weight"
    norm_weight_size = 2048
    request_timeout = 120
    update_timeout = 240
    decode_payload = {
        "text": "The capital of France is",
        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
    }
    norm_weight_update_test_suites = (
        {"load_format": None, "low_target_value": 0.25, "high_target_value": 2.0},
        {
            "load_format": "flattened_bucket",
            "low_target_value": 0.25,
            "high_target_value": 2.0,
        },
    )
    backend_test_suites = [
        {
            "fp8_gemm_backend": "flashinfer_trtllm",
            "moe_runner_backend": "flashinfer_trtllm_routed",
        },
    ]

    def _launch_server(self, fp8_gemm_backend, moe_runner_backend, env=None):
        return popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--base-gpu-id",
                "1",
                "--fp8-gemm-backend",
                fp8_gemm_backend,
                "--moe-runner-backend",
                moe_runner_backend,
            ],
            env=env,
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

    def _run_decode_signature(self):
        ret = self._post_json(
            "/generate",
            {**self.decode_payload, "return_logprob": True},
        )
        output_token_logprobs = ret["meta_info"].get("output_token_logprobs", [])
        self.assertGreater(
            len(output_token_logprobs),
            0,
            "Expected output token logprobs for decode signature.",
        )
        return {
            "text": ret["text"],
            "first_token_logprob": float(output_token_logprobs[0][0]),
        }

    def _assert_decode_signatures_different(self, sig_a, sig_b):
        text_changed = sig_a["text"] != sig_b["text"]
        logprob_delta = abs(sig_a["first_token_logprob"] - sig_b["first_token_logprob"])
        self.assertTrue(
            text_changed or logprob_delta > 1e-2,
            f"Expected decode signature to change, but got {sig_a=} and {sig_b=}",
        )

    def _assert_non_empty_decode(self):
        response = self._run_decode()
        self.assertTrue(len(response) > 0)

    def _assert_success(self, ret):
        self.assertTrue(ret.get("success"), f"{ret=}")

    def _assert_update_changes_decode_signature(self, run_low_update, run_high_update):
        low_ret = run_low_update()
        self._assert_success(low_ret)
        low_signature = self._run_decode_signature()

        high_ret = run_high_update()
        self._assert_success(high_ret)
        high_signature = self._run_decode_signature()

        self._assert_decode_signatures_different(low_signature, high_signature)


class TestServerUpdateWeightsFromDiskMXFP8(_BlackwellMXFP8ServerBase):
    def _launch_server(self, fp8_gemm_backend, moe_runner_backend, env=None):
        # Exercise disk-based update path under tp=4 on Blackwell.
        return popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--base-gpu-id",
                "0",
                "--tp-size",
                "4",
                "--fp8-gemm-backend",
                fp8_gemm_backend,
                "--moe-runner-backend",
                moe_runner_backend,
            ],
            env=env,
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

    def test_parameterized_update_weights_mxfp8(self):
        update_test_suites = (
            {"flush_cache": True, "abort_all_requests": False},
            {"flush_cache": False, "abort_all_requests": False},
        )
        for backend_test_suite in self.backend_test_suites:
            with self.subTest(**backend_test_suite):
                process = self._launch_server(
                    backend_test_suite["fp8_gemm_backend"],
                    backend_test_suite["moe_runner_backend"],
                )
                try:
                    origin_model_path = self._get_model_info()
                    self.assertEqual(origin_model_path, self.model)
                    self._assert_non_empty_decode()

                    for update_test_suite in update_test_suites:
                        with self.subTest(
                            fp8_gemm_backend=backend_test_suite["fp8_gemm_backend"],
                            moe_runner_backend=backend_test_suite["moe_runner_backend"],
                            flush_cache=update_test_suite["flush_cache"],
                            abort_all_requests=update_test_suite["abort_all_requests"],
                        ):
                            ret = self._run_update_weights(
                                self.model,
                                flush_cache=update_test_suite["flush_cache"],
                                abort_all_requests=update_test_suite[
                                    "abort_all_requests"
                                ],
                            )
                            self._assert_success(ret)
                            self.assertEqual(self._get_model_info(), self.model)
                            self._assert_non_empty_decode()
                finally:
                    kill_process_tree(process.pid)


class TestServerUpdateWeightsFromTensorMXFP8(_BlackwellMXFP8ServerBase):
    def _run_update_weights(self, serialized_named_tensors, load_format=None):
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "flush_cache": True,
        }
        if load_format is not None:
            payload["load_format"] = load_format
        return self._post_json(
            "/update_weights_from_tensor",
            payload,
            timeout=self.update_timeout,
        )

    def _serialize_named_tensors(self, param_name, target_value, load_format):
        new_tensor = torch.full(
            (self.norm_weight_size,),
            target_value,
            device="cuda",
            dtype=torch.bfloat16,
        )
        if load_format == "flattened_bucket":
            bucket = FlattenedTensorBucket(named_tensors=[(param_name, new_tensor)])
            bucket_dict = {
                "flattened_tensor": bucket.get_flattened_tensor(),
                "metadata": bucket.get_metadata(),
            }
            return [MultiprocessingSerializer.serialize(bucket_dict, output_str=True)]
        return [
            MultiprocessingSerializer.serialize(
                [(param_name, new_tensor)], output_str=True
            )
        ]

    def test_parameterized_update_weights_from_tensor_mxfp8(self):
        param_name = self.norm_weight_name
        update_test_suites = self.norm_weight_update_test_suites

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
                            low_target_value=update_test_suite["low_target_value"],
                            high_target_value=update_test_suite["high_target_value"],
                        ):
                            self._assert_non_empty_decode()
                            self._assert_update_changes_decode_signature(
                                run_low_update=lambda: self._run_update_weights(
                                    serialized_named_tensors=self._serialize_named_tensors(
                                        param_name,
                                        update_test_suite["low_target_value"],
                                        update_test_suite["load_format"],
                                    ),
                                    load_format=update_test_suite["load_format"],
                                ),
                                run_high_update=lambda: self._run_update_weights(
                                    serialized_named_tensors=self._serialize_named_tensors(
                                        param_name,
                                        update_test_suite["high_target_value"],
                                        update_test_suite["load_format"],
                                    ),
                                    load_format=update_test_suite["load_format"],
                                ),
                            )
                finally:
                    kill_process_tree(process.pid)


class TestServerUpdateWeightsFromDistributedMXFP8(_BlackwellMXFP8ServerBase):
    @staticmethod
    def _set_nccl_stable_env_for_test():
        # Match existing distributed test setup for stability.
        previous = {
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE"),
            "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE"),
        }
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        return previous

    @staticmethod
    def _restore_env(previous):
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _run_distributed_update(
        self,
        *,
        group,
        group_name,
        param_name,
        param_shape,
        target_value,
        load_format,
    ):
        update_payload = {
            "names": [param_name],
            "dtypes": ["bfloat16"],
            "shapes": [param_shape],
            "group_name": group_name,
            "flush_cache": True,
            "load_format": load_format,
        }
        source_tensor = torch.full(
            tuple(param_shape),
            target_value,
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        with ThreadPoolExecutor(1) as executor:
            update_future = executor.submit(
                self._post_json,
                "/update_weights_from_distributed",
                update_payload,
                self.update_timeout,
            )
            torch.distributed.broadcast(source_tensor, src=0, group=group)
            return update_future.result(timeout=300)

    def test_parameterized_update_weights_from_distributed_mxfp8(self):
        param_name = self.norm_weight_name
        update_test_suites = self.norm_weight_update_test_suites
        previous_nccl_env = self._set_nccl_stable_env_for_test()

        try:
            for backend_test_suite in self.backend_test_suites:
                with self.subTest(**backend_test_suite):
                    process = self._launch_server(
                        backend_test_suite["fp8_gemm_backend"],
                        backend_test_suite["moe_runner_backend"],
                        env={
                            "NCCL_CUMEM_ENABLE": "0",
                            "NCCL_NVLS_ENABLE": "0",
                        },
                    )
                    try:
                        param_shape = [self.norm_weight_size]
                        for update_test_suite in update_test_suites:
                            with self.subTest(
                                fp8_gemm_backend=backend_test_suite["fp8_gemm_backend"],
                                moe_runner_backend=backend_test_suite[
                                    "moe_runner_backend"
                                ],
                                load_format=update_test_suite["load_format"],
                                low_target_value=update_test_suite["low_target_value"],
                                high_target_value=update_test_suite[
                                    "high_target_value"
                                ],
                            ):
                                group_name = (
                                    "test_parameter_update_group_mxfp8_"
                                    f"{random.randint(0, 10**8)}"
                                )
                                master_port = find_available_port(50000)
                                group = None
                                try:
                                    # Match existing distributed test mapping:
                                    # sender rank 0 -> GPU 0, server rank 1 -> GPU 1.
                                    torch.cuda.set_device(0)
                                    self._assert_non_empty_decode()

                                    init_payload = {
                                        "master_address": "127.0.0.1",
                                        "master_port": str(master_port),
                                        "rank_offset": 1,
                                        "world_size": 2,
                                        "group_name": group_name,
                                        "backend": "nccl",
                                    }

                                    with ThreadPoolExecutor(1) as executor:
                                        init_future = executor.submit(
                                            self._post_json,
                                            "/init_weights_update_group",
                                            init_payload,
                                        )
                                        group = init_custom_process_group(
                                            backend="nccl",
                                            init_method=f"tcp://127.0.0.1:{master_port}",
                                            world_size=2,
                                            rank=0,
                                            group_name=group_name,
                                        )
                                        init_ret = init_future.result(timeout=240)

                                    self.assertTrue(init_ret["success"], f"{init_ret=}")

                                    self._assert_update_changes_decode_signature(
                                        run_low_update=lambda: self._run_distributed_update(
                                            group=group,
                                            group_name=group_name,
                                            param_name=param_name,
                                            param_shape=param_shape,
                                            target_value=update_test_suite[
                                                "low_target_value"
                                            ],
                                            load_format=update_test_suite[
                                                "load_format"
                                            ],
                                        ),
                                        run_high_update=lambda: self._run_distributed_update(
                                            group=group,
                                            group_name=group_name,
                                            param_name=param_name,
                                            param_shape=param_shape,
                                            target_value=update_test_suite[
                                                "high_target_value"
                                            ],
                                            load_format=update_test_suite[
                                                "load_format"
                                            ],
                                        ),
                                    )
                                finally:
                                    if group is not None:
                                        torch.distributed.destroy_process_group(group)
                                    try:
                                        self._post_json(
                                            "/destroy_weights_update_group",
                                            {"group_name": group_name},
                                            timeout=120,
                                        )
                                    except Exception:
                                        pass
                    finally:
                        kill_process_tree(process.pid)
        finally:
            self._restore_env(previous_nccl_env)


if __name__ == "__main__":
    unittest.main()
