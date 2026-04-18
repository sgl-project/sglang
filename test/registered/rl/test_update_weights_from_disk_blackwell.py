from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=186, suite="stage-c-test-4-gpu-b200")

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class UpdateWeightsFromDiskBase:
    model = None
    base_url = DEFAULT_URL_FOR_TEST
    request_timeout = 120
    update_timeout = 240
    launch_env = None
    decode_payload = {
        "text": "The capital of France is",
        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
    }
    backend_test_suites = ()
    update_test_suites = (
        {"flush_cache": True, "abort_all_requests": False},
        {"flush_cache": False, "abort_all_requests": False},
    )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if cls.model is None:
            raise NotImplementedError("Subclass must set 'model' attribute")
        if not cls.backend_test_suites:
            raise NotImplementedError(
                "Subclass must set non-empty 'backend_test_suites'"
            )

    def _launch_server(self, backend_test_suite):
        launch_kwargs = {}
        if self.launch_env is not None:
            launch_kwargs["env"] = self.launch_env
        other_args = backend_test_suite.get("other_args")
        return popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            **launch_kwargs,
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
                            ret = self._run_update_weights(
                                self.model,
                                flush_cache=update_test_suite["flush_cache"],
                                abort_all_requests=update_test_suite[
                                    "abort_all_requests"
                                ],
                            )
                            self.assertTrue(ret.get("success"), f"{ret=}")
                            self.assertEqual(self._get_model_info(), self.model)
                            self._assert_non_empty_decode()
                            updated_sig = self._get_decode_logprob_signature()
                            self._assert_decode_logprob_unchanged(
                                baseline_sig, updated_sig
                            )
                finally:
                    kill_process_tree(process.pid)


class TestServerUpdateWeightsFromDiskMXFP8(UpdateWeightsFromDiskBase, CustomTestCase):
    model = "zianglih/Qwen3-30B-A3B-Instruct-2507-MXFP8-last-8-BF16"
    backend_test_suites = (
        {
            "name": "flashinfer_trtllm_routed_mxfp8",
            "other_args": (
                "--base-gpu-id",
                "0",
                "--tp-size",
                "4",
                "--fp8-gemm-backend",
                "flashinfer_trtllm",
                "--moe-runner-backend",
                "flashinfer_trtllm_routed",
            ),
        },
    )


class TestServerUpdateWeightsFromDiskNVFP4(UpdateWeightsFromDiskBase, CustomTestCase):
    model = "nvidia/Qwen3-30B-A3B-NVFP4"
    backend_test_suites = (
        {
            "name": "flashinfer_trtllm_nvfp4",
            "other_args": (
                "--base-gpu-id",
                "0",
                "--tp-size",
                "4",
                "--fp4-gemm-backend",
                "flashinfer_trtllm",
                "--moe-runner-backend",
                "flashinfer_trtllm_routed",
            ),
        },
    )


if __name__ == "__main__":
    unittest.main()
