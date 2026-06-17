import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.mock_model.utils import MOCK_MODEL_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=320, suite="stage-b-test-1-gpu-small-amd")

_MAX_NEW_TOKENS = 4
_TOP_P = 0.99
_TOP_K = 10
_SAMPLING_SEED = 1234
_SERVER_ARGS = (
    "--load-format",
    "dummy",
    "--mem-fraction-static",
    "0.7",
)
_INVALID_SAMPLING_MASK_ERROR = (
    "top_p-only sampling is valid but can return huge masks in the tail"
)


class SamplingMaskTestMixin:
    @classmethod
    def _launch_server(cls, other_args=()):
        cls.model = MOCK_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(*_SERVER_ARGS, *other_args),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post_generate(self, sampling_params, return_sampling_mask=True):
        return requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": sampling_params,
                "return_sampling_mask": return_sampling_mask,
            },
            timeout=60,
        )

    def _generate_sampling_masks(self, sampling_params):
        response = self._post_generate(sampling_params)
        self.assertEqual(response.status_code, 200, response.text)

        output = response.json()
        meta_info = output["meta_info"]
        output_ids = output["output_ids"]
        sampling_masks = meta_info["output_token_sampling_mask"]

        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)
        self.assertEqual(meta_info["completion_tokens"], len(output_ids))
        self.assertEqual(
            meta_info["output_token_sampling_mask_length"], len(output_ids)
        )
        self.assertEqual(len(sampling_masks), len(output_ids))
        return sampling_masks

    def _assert_rejects_unbounded_sampling_mask(self, sampling_params):
        response = self._post_generate(sampling_params)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn(_INVALID_SAMPLING_MASK_ERROR, response.text)


class TestSamplingMask(SamplingMaskTestMixin, CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._launch_server()

    def test_generate_returns_sampling_mask(self):
        top_p_sampling_masks = self._generate_sampling_masks(
            {
                "temperature": 1.0,
                "top_k": _TOP_K,
                "top_p": _TOP_P,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
            }
        )
        for sampling_mask in top_p_sampling_masks:
            self.assertLessEqual(len(sampling_mask), _TOP_K)

        top_k_sampling_masks = self._generate_sampling_masks(
            {
                "temperature": 1.0,
                "top_k": _TOP_K,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
            }
        )
        for sampling_mask in top_k_sampling_masks:
            self.assertEqual(len(sampling_mask), _TOP_K)

        top_k_top_p_one_sampling_masks = self._generate_sampling_masks(
            {
                "temperature": 1.0,
                "top_k": _TOP_K,
                "top_p": 1.0,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
            }
        )
        for sampling_mask in top_k_top_p_one_sampling_masks:
            self.assertEqual(len(sampling_mask), _TOP_K)

    def test_generate_rejects_unbounded_sampling_mask(self):
        self._assert_rejects_unbounded_sampling_mask(
            {
                "temperature": 1.0,
                "top_p": _TOP_P,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
            }
        )
        self._assert_rejects_unbounded_sampling_mask(
            {
                "temperature": 1.0,
                "top_p": 1.0,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
            }
        )


class TestSamplingMaskDeterministic(SamplingMaskTestMixin, CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._launch_server(("--enable-deterministic-inference",))

    def test_return_sampling_mask_preserves_deterministic_sampling(self):
        sampling_params = {
            "temperature": 1.0,
            "top_k": _TOP_K,
            "top_p": 1.0,
            "sampling_seed": _SAMPLING_SEED,
            "max_new_tokens": _MAX_NEW_TOKENS,
            "ignore_eos": True,
        }

        with_mask_response = self._post_generate(
            sampling_params, return_sampling_mask=True
        )
        self.assertEqual(with_mask_response.status_code, 200, with_mask_response.text)

        without_mask_response = self._post_generate(
            sampling_params, return_sampling_mask=False
        )
        self.assertEqual(
            without_mask_response.status_code, 200, without_mask_response.text
        )

        with_mask_output = with_mask_response.json()
        without_mask_output = without_mask_response.json()
        self.assertEqual(
            with_mask_output["output_ids"], without_mask_output["output_ids"]
        )
        self.assertEqual(with_mask_output["text"], without_mask_output["text"])


if __name__ == "__main__":
    unittest.main()
