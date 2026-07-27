"""
NPU multimodal PD disaggregation test case.

Deploy prefill and decode servers on two NPU cards, launch a
sglang_router, and verify image inference through the router.
"""

import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_5_9B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_multimodal_utils import (
    Color,
    Shape,
    assert_color_and_shape,
    chat_single_image,
    create_test_image,
    launch_router,
    launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=150, suite="full-2-npu-a3", nightly=True)


class TestMultimodalPDDisaggregation(CustomTestCase):
    """Verify PD disaggregation + image inference is correct.

    [Test Category] multimodal
    [Test Target] multimodal + pd-disaggregation (2-NPU)
    """

    _model = QWEN3_5_9B_WEIGHTS_PATH
    _host = "127.0.0.1"
    _bootstrap_port = 18998

    _prefill_url = f"http://{_host}:20010"
    _decode_url = f"http://{_host}:20020"
    _router_port = 20030

    _pd_env = {**os.environ, "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666"}

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls._prompt = "Describe the image"

        cls._prefill_process, _ = launch_server(
            cls._model,
            extra_args=[
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-transfer-backend",
                "ascend",
                "--disaggregation-bootstrap-port",
                str(cls._bootstrap_port),
                "--mamba-radix-cache-strategy",
                "extra_buffer",
                "--mem-fraction-static",
                "0.6",
                "--tp-size",
                "1",
                "--base-gpu-id",
                "0",
            ],
            env=cls._pd_env,
            port=20010,
        )

        cls._decode_process, _ = launch_server(
            cls._model,
            extra_args=[
                "--disaggregation-mode",
                "decode",
                "--disaggregation-transfer-backend",
                "ascend",
                "--disaggregation-bootstrap-port",
                str(cls._bootstrap_port),
                "--cuda-graph-bs-decode",
                "1",
                "2",
                "4",
                "--mamba-radix-cache-strategy",
                "extra_buffer",
                "--dtype",
                "bfloat16",
                "--mamba-ssm-dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.6",
                "--tp-size",
                "1",
                "--base-gpu-id",
                "1",
            ],
            env=cls._pd_env,
            port=20020,
        )

        cls._router_process, cls._router_url = launch_router(
            cls._prefill_url,
            cls._decode_url,
            cls._host,
            cls._router_port,
        )

    @classmethod
    def tearDownClass(cls):
        for proc in [cls._router_process, cls._decode_process, cls._prefill_process]:
            if proc:
                try:
                    kill_process_tree(proc.pid)
                except Exception:
                    pass

    def test_multimodal_pd_disaggregation(self):
        """Two-server PD with sglang_router: prefill+decode."""
        output = chat_single_image(
            self._router_url,
            self._image_b64,
            self._prompt,
            max_tokens=128,
        )

        self.assertTrue(
            output, "test_multimodal_pd_disaggregation: PD router returned empty output"
        )
        self.assertGreater(
            len(output),
            5,
            f"test_multimodal_pd_disaggregation: PD output too short: '{output}'",
        )
        assert_color_and_shape(
            self,
            output,
            Color.BLUE.name.lower(),
            Shape.RECTANGLE.value,
            prefix="test_multimodal_pd_disaggregation: ",
        )


if __name__ == "__main__":
    unittest.main()
