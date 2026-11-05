import unittest

import requests

from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="base-b-test-1-npu-a3")

# Qwen3-8B hidden size; the three EAGLE3 aux layers are each this wide, packed
# along the last dim by `pack_aux_hidden_states`.
QWEN3_8B_HIDDEN_SIZE = 4096
NUM_AUX_LAYERS = 3
EXPECTED_PACKED_DIM = QWEN3_8B_HIDDEN_SIZE * NUM_AUX_LAYERS


class TestNpuReturnAuxHiddenStates(CustomTestCase):
    """NPU e2e smoke test for ``--enable-return-aux-hidden-states``.

    Captures EAGLE3 auxiliary hidden states on the target model *without* a
    draft, then returns them to the client. A ``return_hidden_states`` request
    should get ``num_aux_layers x hidden_size`` packed per token instead of the
    ordinary last-layer hidden state.
    """

    model = QWEN3_8B_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
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
                "3",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.7",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_return_aux_hidden_states_last(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                "return_hidden_states": "last",
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)
        hidden_states = response.json()["meta_info"]["hidden_states"]
        self.assertIsNotNone(hidden_states)
        # "last" returns a single packed aux vector: num_aux_layers * hidden_size.
        self.assertEqual(len(hidden_states), EXPECTED_PACKED_DIM)

    def test_return_aux_hidden_states_full(self):
        response = requests.post(
            f"{self.base_url}/generate",
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
