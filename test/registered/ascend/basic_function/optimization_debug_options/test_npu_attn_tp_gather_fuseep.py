import os
import unittest

import requests

os.environ.setdefault("HCCL_BUFFSIZE", "600")

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_6_35B_A3B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-4-npu-a3", nightly=True)


class TestAttnTpGatherDense(CustomTestCase):
    """Verify --disable-attn-tp-gather does not break non-MOE (dense) models.

    For dense models, attention TP gather is not applicable regardless of
    the flag. This test verifies the server starts and infers correctly
    when the flag is passed to a dense model.

    [Test Category] Parameter
    [Test Target] --disable-attn-tp-gather
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    def test_dense_model_noop(self):
        """Flag is a no-op for non-MOE models — server starts and infers correctly."""
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--disable-attn-tp-gather",
            ],
        )
        try:
            resp = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Paris", resp.text)
        finally:
            kill_process_tree(process.pid)


class TestAttnTpGatherDPAttn(CustomTestCase):
    """Verify --disable-attn-tp-gather under --enable-dp-attention.

    When dp_size == tp_size, gather is already disabled without the flag
    (dp_size < tp_size is false). When dp_size < tp_size, gather is enabled
    by default, and --disable-attn-tp-gather overrides it. Both scenarios
    are tested with and without the flag.

    [Test Category] Parameter
    [Test Target] --disable-attn-tp-gather; --enable-dp-attention
    """

    model = QWEN3_6_35B_A3B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @staticmethod
    def _launch(tp_size, dp_size, disable_gather):
        """Launch server with DP attention configuration.

        Args:
            tp_size: Tensor parallelism size.
            dp_size: Data parallelism size (must divide tp_size).
            disable_gather: Whether to pass --disable-attn-tp-gather.
        """
        args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--moe-a2a-backend",
            "deepep",
            "--enable-dp-attention",
            "--tp-size",
            str(tp_size),
            "--dp-size",
            str(dp_size),
        ]
        if disable_gather:
            args.append("--disable-attn-tp-gather")
        return popen_launch_server(
            TestAttnTpGatherDPAttn.model,
            TestAttnTpGatherDPAttn.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
        )

    def _make_request(self):
        return requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )

    def test_dp_equals_tp(self):
        """dp_size == tp_size: gather is already disabled by default.

        When dp_size equals tp_size, attention TP gather is not needed.
        The flag is a no-op in this configuration. Tests both with and
        without the flag to verify neither breaks.
        """
        # WITHOUT --disable-attn-tp-gather
        process1 = self._launch(tp_size=2, dp_size=2, disable_gather=False)
        try:
            resp1 = self._make_request()
            self.assertEqual(resp1.status_code, 200)
            self.assertIn("Paris", resp1.text)
        finally:
            kill_process_tree(process1.pid)

        # WITH --disable-attn-tp-gather
        process2 = self._launch(tp_size=2, dp_size=2, disable_gather=True)
        try:
            resp2 = self._make_request()
            self.assertEqual(resp2.status_code, 200)
            self.assertIn("Paris", resp2.text)
        finally:
            kill_process_tree(process2.pid)

    def test_dp_less_than_tp(self):
        """dp_size < tp_size: gather is enabled by default, flag overrides it.

        When dp_size is less than tp_size, attention TP gather is enabled
        by default. --disable-attn-tp-gather overrides this behavior.
        Tests both with and without the flag to verify the override works.
        """
        # WITHOUT --disable-attn-tp-gather
        process1 = self._launch(tp_size=4, dp_size=2, disable_gather=False)
        try:
            resp1 = self._make_request()
            self.assertEqual(resp1.status_code, 200)
            self.assertIn("Paris", resp1.text)
        finally:
            kill_process_tree(process1.pid)

        # WITH --disable-attn-tp-gather
        process2 = self._launch(tp_size=4, dp_size=2, disable_gather=True)
        try:
            resp2 = self._make_request()
            self.assertEqual(resp2.status_code, 200)
            self.assertIn("Paris", resp2.text)
        finally:
            kill_process_tree(process2.pid)


if __name__ == "__main__":
    unittest.main()
