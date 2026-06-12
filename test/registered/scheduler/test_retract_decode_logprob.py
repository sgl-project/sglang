"""Regression test for stream_output_generation logprob off-by-one under retract.

Bug: under overlap scheduling, if a request is retracted between the moment
its prefill batch is launched and the moment that batch's result is
processed, the in-flight batch comes back through
`process_batch_result_prefill` with the now-retracted request still in
`batch.reqs`. The `if req.is_retracted: continue` guard at the top of the
for-loop skips `req.output_ids.append(next_token_id)`, so the subsequent
`self.stream_output(batch.reqs, ...)` sees a req with `len(output_ids) == 0`.
For a non-streaming `return_logprob=True` req,
`should_output = (0 % DEFAULT_FORCE_STREAM_INTERVAL == 0) == True`, so the
slice math in `_GenerationStreamAccumulator.handle_req` (formerly
`stream_output_generation`) fires:

    output_ids_ = req.output_ids_through_stop                  # empty
    output_ids.append(output_ids_[send_token_offset:])         # 0 tokens
    req.send_token_offset = len(output_ids_)                   # 0
    logprob_end = max(len(output_ids_), 1)                     # 1 <-- BUG
    output_token_logprobs_val.append(... [send_lp_off:logprob_end])  # 1 entry
    req.send_output_token_logprobs_offset = logprob_end         # 1

The two send-offsets diverge by 1. Every subsequent stream tick for this
req ships N tokens and N-1 logprobs. The final response delivered to the
client has `len(meta_info["output_token_logprobs"]) == len(output_ids) - 1`.

Fix: drop the `max(..., 1)` floor:

    logprob_end = len(output_ids_)

This test forces the trigger reliably via `SGLANG_TEST_RETRACT=True`
(retract every two forward steps) and asserts the 1:1 invariant across
many concurrent `return_logprob=True` requests.

Run:
    python -m unittest test_retract_decode_logprob.TestRetractDecodeLogprob
"""

import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


N_REQUESTS = 32
MAX_NEW_TOKENS = 256


class TestRetractDecodeLogprob(CustomTestCase):
    """python -m unittest test_retract_decode_logprob.TestRetractDecodeLogprob"""

    other_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--chunked-prefill-size", "128",
            "--max-running-requests", "8",
            "--mem-fraction-static", "0.3",
        ] + cls.other_args
        with envs.SGLANG_TEST_RETRACT.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=launch_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _one_request(self, idx: int) -> dict:
        # NOTE: no `stream` field -> non-streaming. The bug affects the
        # non-streaming path too because _stream_output_generation runs for
        # both (non-streaming reqs get internally force-flushed every
        # DEFAULT_FORCE_STREAM_INTERVAL decoded tokens).
        payload = {
            "text": f"Once upon a time #{idx},",
            "sampling_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
        }
        r = requests.post(f"{self.base_url}/generate", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        meta = data.get("meta_info", {})
        return {
            "idx": idx,
            "n_tokens": len(data.get("output_ids") or []),
            "n_logprobs": len(meta.get("output_token_logprobs") or []),
        }

    def test_output_logprobs_aligned_under_test_retract(self):
        """Every non-streaming return_logprob=True response must have
        len(output_ids) == len(meta_info["output_token_logprobs"]).

        Without the fix, with SGLANG_TEST_RETRACT=True forcing retraction
        every two forward steps, ~6% of responses come back with one fewer
        logprob than tokens. With the fix, all responses are 1:1."""

        with ThreadPoolExecutor(max_workers=N_REQUESTS) as pool:
            futs = [pool.submit(self._one_request, i) for i in range(N_REQUESTS)]
            results = [f.result() for f in as_completed(futs)]

        mismatches = [r for r in results if r["n_tokens"] != r["n_logprobs"]]
        self.assertEqual(
            mismatches,
            [],
            msg=(
                f"{len(mismatches)}/{N_REQUESTS} responses have "
                f"len(output_ids) != len(output_token_logprobs). "
                f"Sample: {mismatches[:5]}. "
                "This is the _GenerationStreamAccumulator logprob_end "
                "off-by-one (max(len(output_ids_), 1) in "
                "scheduler_components/output_streamer.py)."
            ),
        )

        assert self.process.poll() is None, "Server crashed during test"


if __name__ == "__main__":
    unittest.main()
