"""Minimal end-to-end smoke test for ScriptedRuntime."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def _smoke_script(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=10, max_new_tokens=4)
    yield
    yield
    assert r1.status in ("waiting", "running", "unknown")


class TestScriptedRuntimeSmoke(CustomTestCase):
    def test_smoke(self):
        """Minimal end-to-end smoke for ScriptedRuntime launch and one req."""
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
