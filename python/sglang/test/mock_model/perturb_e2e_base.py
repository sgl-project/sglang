from __future__ import annotations

import io
import os
from typing import ClassVar

from sglang.test.kv_canary.e2e_base import CapturedServerE2EBase
from sglang.test.kv_canary.utils import post_parallel_generate
from sglang.test.mock_model_utils import (
    MOCK_MODEL_PATH,
    mock_model_server_args,
    mock_model_server_env,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


class MockModelPerturbE2EBase(CapturedServerE2EBase):
    """Base for mock-model self-test perturb e2e tests.

    ``setUpClass`` launches the mock-model + canary server with subclass-provided
    extra env / extra server args. Server lifecycle, log capture, and the generic
    violation-log assertions are inherited from ``CapturedServerE2EBase``.

    The canary mode is ``log`` (not ``raise``) so the server stays alive after
    the first violation — clients get their responses, log keeps accumulating,
    and the test's log-based assertions run without races.
    """

    extra_env: ClassVar[dict[str, str]] = {}
    extra_server_args: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def setUpClass(cls) -> None:
        server_env = os.environ.copy()
        server_env.update(mock_model_server_env())
        server_env.update(cls.extra_env)

        cls._stdout_buf = io.StringIO()
        cls._stderr_buf = io.StringIO()

        cls.process = popen_launch_server(
            MOCK_MODEL_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=mock_model_server_args(
                *cls.extra_server_args, canary_mode="log"
            ),
            env=server_env,
            return_stdout_stderr=(cls._stdout_buf, cls._stderr_buf),
        )

    def send_parallel_requests(
        self,
        n: int = 4,
        *,
        max_new_tokens: int = 256,
        timeout: float = 30.0,
    ) -> list[dict]:
        prompts = ["hello world " * 50] * n
        return post_parallel_generate(
            url=self.base_url + "/generate",
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            timeout=timeout,
        )

    def assert_any_launch_tag_violation_reported(
        self,
        *,
        fail_reason: str,
        flush_wait_seconds: float = 3.0,
        max_retries: int = 10,
    ) -> None:
        """Wildcard-launch_tag wrapper around ``assert_violation_logged_any``:
        mock-model self-test doesn't care which HEAD/TAIL/FULL/SWA produced the
        violation, only that some line with the given fail_reason landed."""
        self.assert_violation_logged_any(
            launch_tag_patterns=("*",),
            fail_reason=fail_reason,
            flush_wait_seconds=flush_wait_seconds,
            max_retries=max_retries,
        )

    def assert_any_launch_tag_violation_absent(self, *, fail_reason: str) -> None:
        self.assert_no_violation_matching(
            launch_tag_patterns=("*",), fail_reason=fail_reason
        )
