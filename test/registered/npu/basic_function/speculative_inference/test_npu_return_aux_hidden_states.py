"""Tests for --enable-return-aux-hidden-states.

One test layer:
- End-to-end: launch a target-only server (no draft model) that captures EAGLE3
  auxiliary hidden states and returns them packed per decode token (NPU)
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="base-b-test-1-npu-a3")

# Qwen3-8B hidden size; the aux layers are each this wide, packed along the
# last dim by `pack_aux_hidden_states`. Use a non-default layer count (2) so the
# assertion also proves `--return-aux-hidden-state-layer-ids` is honored
# (EAGLE3 defaults to 3 layers).
QWEN3_8B_HIDDEN_SIZE = 4096
NUM_AUX_LAYERS = 2
EXPECTED_PACKED_DIM = QWEN3_8B_HIDDEN_SIZE * NUM_AUX_LAYERS


class TestNpuReturnAuxHiddenStates(CustomTestCase):
    """Testcase: Verify --enable-return-aux-hidden-states returns the packed
       auxiliary hidden states of the target model without loading a draft model

    [Test Category] Parameter
    [Test Target] Verify the service returns num_aux_layers x hidden_size packed
                  hidden states per decode token
    """

    model = QWEN3_8B_WEIGHTS_PATH
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    _OTHER_ARGS = [
        "--device",
        "npu",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "ascend",
        "--enable-return-aux-hidden-states",
        "--speculative-algorithm",
        "EAGLE3",
        "--return-aux-hidden-state-layer-ids",
        "1",
        "2",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.7",
    ]

    def setUp(self):
        self._process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=self.timeout,
            other_args=list(self._OTHER_ARGS),
        )
        self.assertIsNone(
            self._process.poll(),
            "Server exited prematurely",
        )

    def tearDown(self):
        kill_process_tree(self._process.pid)

    def test_return_aux_hidden_states(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                "return_hidden_states": True,
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["meta_info"]["completion_tokens"], 4)
        hidden_states = body["meta_info"]["hidden_states"]
        self.assertIsNotNone(hidden_states)
        # "full" returns the prefill chunk followed by one packed aux vector per
        # decode token; the trailing decode rows are 1-D packed aux vectors.
        self.assertGreaterEqual(len(hidden_states), 2)
        for row in hidden_states[1:]:
            self.assertEqual(len(row), EXPECTED_PACKED_DIM)


if __name__ == "__main__":
    unittest.main()
