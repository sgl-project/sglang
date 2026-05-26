"""Minimal end-to-end smoke test for ScriptedRuntime."""

import unittest

from sglang.test.scripted_runtime import ScriptedRuntime, execute_scripted_runtime
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase


def _smoke_script(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=10, max_new_tokens=4)
    yield
    yield
    assert r1.status in ("waiting", "running", "unknown")


class TestScriptedRuntimeSmoke(CustomTestCase):
    def test_smoke(self):
        execute_scripted_runtime(
            _smoke_script,
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            tp_size=1,
            dp_size=1,
            pp_size=1,
            disable_overlap_schedule=True,
            disable_cuda_graph=True,
        )


if __name__ == "__main__":
    unittest.main()
