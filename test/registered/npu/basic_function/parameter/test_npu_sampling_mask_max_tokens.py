import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_pd_server,
    popen_launch_server,
    popen_with_error_check,
)

register_npu_ci(est_time=600, suite="base-b-test-2-npu-a3")
register_npu_ci(est_time=600, suite="nightly-2-npu-a3", nightly=True)

_MAX_NEW_TOKENS = 4
_TOP_K = 8
_TOP_P = 0.9
_SAMPLING_MASK_MAX_TOKENS = 64


def _post_generate(base_url, sampling_params):
    return requests.post(
        base_url + "/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {
                "temperature": 1.0,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
                **sampling_params,
            },
            "return_sampling_mask": True,
        },
        timeout=60,
    )


class TestNpuSamplingMaskMaxTokens(CustomTestCase):
    """Testcase: Verify `--sampling-mask-max-tokens` bounds the returned sampling
    mask and rejects requests whose top_k exceeds the cap on the NPU backend.

    [Test Category] Parameter
    [Test Target] --sampling-mask-max-tokens
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--sampling-mask-max-tokens",
                str(_SAMPLING_MASK_MAX_TOKENS),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_greedy_returns_singleton_mask(self):
        response = _post_generate(self.base_url, {"temperature": 0.0})
        self.assertEqual(response.status_code, 200, response.text)

        output = response.json()
        output_ids = output["output_ids"]
        masks = output["meta_info"]["output_token_sampling_mask"]

        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)
        self.assertEqual(len(masks), _MAX_NEW_TOKENS)
        self.assertTrue(all(len(mask) == 1 for mask in masks))
        for token_id, mask in zip(output_ids, masks):
            self.assertIn(token_id, mask)

    def test_non_greedy_returns_bounded_mask(self):
        response = _post_generate(
            self.base_url, {"temperature": 1.0, "top_k": _TOP_K, "top_p": _TOP_P}
        )
        self.assertEqual(response.status_code, 200, response.text)

        output = response.json()
        output_ids = output["output_ids"]
        meta = output["meta_info"]
        masks = meta["output_token_sampling_mask"]

        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)
        self.assertEqual(meta["output_token_sampling_mask_length"], _MAX_NEW_TOKENS)
        self.assertEqual(len(masks), _MAX_NEW_TOKENS)
        for token_id, mask in zip(output_ids, masks):
            self.assertIn(token_id, mask)
            self.assertEqual(len(mask), len(set(mask)))
            self.assertLessEqual(len(mask), _SAMPLING_MASK_MAX_TOKENS)

    def test_rejects_top_k_exceeding_cap(self):
        response = _post_generate(
            self.base_url, {"top_k": _SAMPLING_MASK_MAX_TOKENS + 1}
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn(
            "return_sampling_mask requires top_k=1 for greedy sampling",
            response.text,
        )


class TestNpuSamplingMaskMaxTokensPD(TestDisaggregationBase):
    """Testcase: Verify `--sampling-mask-max-tokens` bounds the sampling mask that
    is transferred across prefill/decode in PD disaggregation. Both sides must be
    launched with SGLANG_ENABLE_DISAGG_SAMPLING_MASK=1 and the same
    --sampling-mask-max-tokens value.

    [Test Category] Parameter
    [Test Target] --sampling-mask-max-tokens (PD disaggregation)
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.bootstrap_port = f"{int(cls.lb_port) + 500}"
        cls.start_prefill()
        cls.start_decode()
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")
        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            "1",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
            "--sampling-mask-max-tokens",
            str(_SAMPLING_MASK_MAX_TOKENS),
        ]
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env={
                "SGLANG_ENABLE_DISAGG_SAMPLING_MASK": "1",
                "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24664",
            },
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size",
            "1",
            "--base-gpu-id",
            "1",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
            "--sampling-mask-max-tokens",
            str(_SAMPLING_MASK_MAX_TOKENS),
        ]
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env={
                "SGLANG_ENABLE_DISAGG_SAMPLING_MASK": "1",
                "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24664",
            },
        )

    @classmethod
    def launch_lb(cls):
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--prefill",
            cls.prefill_url,
            cls.bootstrap_port,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        print("Starting load balancer:", " ".join(lb_command))
        cls.process_lb = popen_with_error_check(lb_command)
        cls.wait_server_ready(cls.lb_url + "/health")

    def test_pd_greedy_returns_singleton_mask(self):
        response = _post_generate(self.lb_url, {"temperature": 0.0})
        self.assertEqual(response.status_code, 200, response.text)

        output = response.json()
        output_ids = output["output_ids"]
        masks = output["meta_info"]["output_token_sampling_mask"]

        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)
        self.assertEqual(len(masks), _MAX_NEW_TOKENS)
        self.assertTrue(all(len(mask) == 1 for mask in masks))
        for token_id, mask in zip(output_ids, masks):
            self.assertIn(token_id, mask)

    def test_pd_non_greedy_returns_bounded_mask(self):
        response = _post_generate(
            self.lb_url, {"temperature": 1.0, "top_k": _TOP_K, "top_p": _TOP_P}
        )
        self.assertEqual(response.status_code, 200, response.text)

        output = response.json()
        output_ids = output["output_ids"]
        meta = output["meta_info"]
        masks = meta["output_token_sampling_mask"]

        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)
        self.assertEqual(meta["output_token_sampling_mask_length"], _MAX_NEW_TOKENS)
        self.assertEqual(len(masks), _MAX_NEW_TOKENS)
        for token_id, mask in zip(output_ids, masks):
            self.assertIn(token_id, mask)
            self.assertEqual(len(mask), len(set(mask)))
            self.assertLessEqual(len(mask), _SAMPLING_MASK_MAX_TOKENS)

    def test_pd_rejects_top_k_exceeding_cap(self):
        response = _post_generate(
            self.lb_url, {"top_k": _SAMPLING_MASK_MAX_TOKENS + 1}
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn(
            "return_sampling_mask requires top_k=1 for greedy sampling",
            response.text,
        )


if __name__ == "__main__":
    unittest.main()