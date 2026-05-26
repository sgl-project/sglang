"""Disagg × chunked: naive ScriptedRuntime smoke.

Submit a long-prompt request to a prefill-engine running with
``--chunked-prefill-size 256``. The prefill engine must chunk the
request and hand off KV to the decode engine cleanly.

The current ScriptedRuntime only spins up a single Engine; the
disagg topology requires running two Engines and routing through the
PD router (wishlist §4 P3 (16)). This test encodes the *intent* —
when disagg support lands the script body stays unchanged.

Requires 2-4 GPUs in PD configuration.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until_finished,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, CustomTestCase


class TestScriptedDisagg(CustomTestCase):
    def test_naive_disagg_chunked(self):
        """Disagg x chunked: prefill engine chunks long prompt and hands off to decode."""
        # When ScriptedRuntime grows disagg topology support, the
        # decode-side engine kwargs go through ``decode_engine_kwargs=``
        # (or similar) — see wishlist §4 P3 (16).
        execute_scripted_runtime(
            self._script_naive_disagg_chunked,
            **base_engine_kwargs(
                model_path=DEFAULT_MODEL_NAME_FOR_TEST,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
            ),
        )

    @staticmethod
    def _script_naive_disagg_chunked(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2


if __name__ == "__main__":
    unittest.main()
