"""Core chunked-prefill smoke test (registered, CUDA CI).

Smallest possible scripted-runtime exercise of the chunked-prefill code
path: boot an engine with a tiny ``chunked_prefill_size``, submit one
request whose prompt comfortably exceeds the chunk size, and drive it
to ``finished``. Mirrors the structure of the manual scripted tests in
``test/manual/chunked_prefill/`` but keeps a single short script so it
fits in the per-commit CI budget.

The full chunked corner-case grid lives in the manual suite — this
file's job is just to fail loudly if the chunked path stops working
end-to-end at all.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    base_engine_kwargs,
    run_until_finished,
)

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


# Small enough to keep the test fast, large enough to give the
# scheduler multiple chunks worth of work so the chunked path is
# actually entered.
_CHUNK_SIZE = 16
_PROMPT_LEN = 64  # = 4 * _CHUNK_SIZE — forces multi-chunk prefill


class TestScriptedCoreSmoke(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=_CHUNK_SIZE)

    def test_chunked_prefill_smoke(self):
        """Engine boots with small chunk_size and a multi-chunk req finishes cleanly."""
        self.runtime.run(self._script_chunked_prefill_smoke)

    @staticmethod
    def _script_chunked_prefill_smoke(t: ScriptedRuntime):
        r = t.start_req(prompt_len=_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished, f"req did not finish, status={r.status!r}"


if __name__ == "__main__":
    unittest.main()
