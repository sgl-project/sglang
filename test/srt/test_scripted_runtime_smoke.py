"""Smoke test for ScriptedRuntime.

Verifies the basic generator-driven mechanism end-to-end:
1. An Engine launches with a scripted_runtime config.
2. The scheduler subprocess constructs a ScriptedRuntime, runs the
   provided script generator.
3. The script can submit a request via ``start_req``, advance the
   scheduler with ``yield``, and read back status via ReqHandle.
4. When the generator returns, all ranks exit cleanly and the test
   process unblocks without an exception.

The script function is intentionally underscore-prefixed so pytest
does not collect it as a test directly.
"""

from sglang.test.scripted_runtime import ScriptedRuntime, execute_scripted_runtime
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


def _smoke_script(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=10, max_new_tokens=4)
    yield  # let the scheduler pick up the injected request
    yield  # one more step so it has a chance to enter running
    assert r1.status in (
        "waiting",
        "running",
        "unknown",
    ), f"unexpected status: {r1.status!r}"


def test_smoke_scripted_runtime():
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
    test_smoke_scripted_runtime()
