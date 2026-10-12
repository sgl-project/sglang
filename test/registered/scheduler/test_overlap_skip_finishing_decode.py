"""The overlap scheduler skips the decode of requests whose queued result
already carries their max_new_tokens-th token."""

import re
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

BATCH_SIZE = 8
MAX_NEW_TOKENS = 4


def _forward_pass_count(base_url: str, mode_prefix: str) -> float:
    metrics = requests.get(base_url + "/metrics", timeout=30).text
    matches = re.findall(
        r'^sglang:cuda_graph_passes_total\{[^}]*mode="' + mode_prefix + r'_[^"]*"'
        r"[^}]*\}\s+([0-9.eE+-]+)$",
        metrics,
        re.MULTILINE,
    )
    return sum(map(float, matches), 0.0)


class TestOverlapSkipFinishingDecode(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1):
            cls.process = popen_launch_server(
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--enable-metrics"],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, batch_size: int) -> list:
        # Pre-tokenized input reaches the scheduler as one batched request.
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": [[1000 + i] * 16 for i in range(batch_size)],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "ignore_eos": True,
                },
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def test_batch_decodes_once_per_non_prefill_token(self):
        self._generate(1)
        prefill_before = _forward_pass_count(self.base_url, "prefill")
        decode_before = _forward_pass_count(self.base_url, "decode")

        outputs = self._generate(BATCH_SIZE)

        self.assertEqual(len(outputs), BATCH_SIZE)
        for output in outputs:
            self.assertEqual(output["meta_info"]["completion_tokens"], MAX_NEW_TOKENS)
        # One prefill emits the first token; each later token needs one decode,
        # and no decode runs for the token already in flight.
        self.assertEqual(
            _forward_pass_count(self.base_url, "prefill") - prefill_before, 1
        )
        self.assertEqual(
            _forward_pass_count(self.base_url, "decode") - decode_before,
            MAX_NEW_TOKENS - 1,
        )


if __name__ == "__main__":
    unittest.main()
