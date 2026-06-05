"""Unit tests for cuda_coredump.get_dump_dir() directory resolution.

get_dump_dir() must resolve the dump directory identically to the bash logic in
.github/actions/upload-cuda-coredumps/action.yml so the producer (the server
process) and the uploader (the composite action) always agree on the path:

  explicit (non-empty) SGLANG_CUDA_COREDUMP_DIR
    > per-job RUNNER_TEMP base (wiped between CI jobs)
    > /tmp default,
  then a per-(run, attempt) subdir under GITHUB_RUN_ID.

The per-(run, attempt) subdir is what prevents a coredump left on a shared
self-hosted runner by one job from being mis-attributed to a later, unrelated
job (a distinct GITHUB_RUN_ID yields a distinct dir).
"""

import os
import unittest
from contextlib import ExitStack
from unittest import mock

from sglang.srt.debug_utils import cuda_coredump
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# External (runner / GitHub Actions) env vars, cleared per case so the ambient
# CI environment does not leak into the assertions.
_CI_ENV_KEYS = ("RUNNER_TEMP", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")


class TestCudaCoredumpDir(CustomTestCase):
    def _resolve(self, *, explicit=None, runner_temp=None, run_id=None, attempt=None):
        """Resolve get_dump_dir() under a fully controlled environment.

        explicit -> SGLANG_CUDA_COREDUMP_DIR override (None = unset);
        runner_temp / run_id / attempt -> external CI vars (None = unset).
        """
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_CUDA_COREDUMP_DIR.override(explicit))
            stack.enter_context(mock.patch.dict(os.environ, {}, clear=False))
            for key in _CI_ENV_KEYS:
                os.environ.pop(key, None)
            if runner_temp is not None:
                os.environ["RUNNER_TEMP"] = runner_temp
            if run_id is not None:
                os.environ["GITHUB_RUN_ID"] = run_id
            if attempt is not None:
                os.environ["GITHUB_RUN_ATTEMPT"] = attempt
            return cuda_coredump.get_dump_dir()

    def test_local_default_when_nothing_set(self):
        self.assertEqual(self._resolve(), "/tmp/sglang_cuda_coredumps")

    def test_runner_temp_base_without_run_id(self):
        self.assertEqual(self._resolve(runner_temp="/rt"), "/rt/sglang_cuda_coredumps")

    def test_per_run_subdir_under_runner_temp(self):
        self.assertEqual(
            self._resolve(runner_temp="/rt", run_id="42", attempt="3"),
            "/rt/sglang_cuda_coredumps/42-3",
        )

    def test_run_attempt_defaults_to_one(self):
        self.assertEqual(
            self._resolve(runner_temp="/rt", run_id="42"),
            "/rt/sglang_cuda_coredumps/42-1",
        )

    def test_run_id_subdir_under_tmp_default(self):
        self.assertEqual(
            self._resolve(run_id="42", attempt="2"),
            "/tmp/sglang_cuda_coredumps/42-2",
        )

    def test_explicit_override_wins_over_runner_temp(self):
        self.assertEqual(
            self._resolve(explicit="/custom", runner_temp="/rt", run_id="42"),
            "/custom/42-1",
        )

    def test_empty_explicit_falls_through_to_runner_temp(self):
        # Mirrors the action's `[ -n "$SGLANG_CUDA_COREDUMP_DIR" ]`: an empty
        # value counts as unset, not as an explicit "" base.
        self.assertEqual(
            self._resolve(explicit="", runner_temp="/rt", run_id="42"),
            "/rt/sglang_cuda_coredumps/42-1",
        )


if __name__ == "__main__":
    unittest.main()
