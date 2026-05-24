from __future__ import annotations

import io
import os
import re
import time
from typing import ClassVar, Optional

from sglang.srt.utils import kill_process_tree
from sglang.test.kv_canary.utils import post_parallel_generate
from sglang.test.mock_model_utils import (
    MOCK_MODEL_PATH,
    mock_model_server_args,
    mock_model_server_env,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_VIOLATION_LINE_RE = re.compile(
    r"kv_canary violation: launch_tag=(\S+) fail_reason=(\S+)"
)


class MockModelPerturbE2EBase(CustomTestCase):
    """Base for mock-model self-test perturb e2e tests.

    Launches the mock-model + canary server with subclass-provided extra env
    and extra server args, then exposes helpers to send parallel requests and
    assert kv_canary violation lines (or their absence) in the captured log.
    """

    extra_env: ClassVar[dict[str, str]] = {}
    extra_server_args: ClassVar[tuple[str, ...]] = ()

    process: ClassVar[Optional[object]] = None
    base_url: ClassVar[str] = DEFAULT_URL_FOR_TEST
    _stdout_buf: ClassVar[Optional[io.StringIO]] = None
    _stderr_buf: ClassVar[Optional[io.StringIO]] = None

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
            other_args=mock_model_server_args(*cls.extra_server_args),
            env=server_env,
            return_stdout_stderr=(cls._stdout_buf, cls._stderr_buf),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
        for buf in (cls._stdout_buf, cls._stderr_buf):
            if buf is not None:
                buf.close()
        cls._stdout_buf = None
        cls._stderr_buf = None

    def _captured_log_text(self) -> str:
        stdout_text = (
            self._stdout_buf.getvalue() if self._stdout_buf is not None else ""
        )
        stderr_text = (
            self._stderr_buf.getvalue() if self._stderr_buf is not None else ""
        )
        return stdout_text + stderr_text

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

    def assert_violation_reported(
        self,
        *,
        fail_reason: str,
        flush_wait_seconds: float = 3.0,
        max_retries: int = 10,
    ) -> None:
        for _ in range(max_retries):
            time.sleep(flush_wait_seconds)
            if _log_contains_violation(
                log_text=self._captured_log_text(), fail_reason=fail_reason
            ):
                return
        raise AssertionError(
            f"No kv_canary violation line with fail_reason={fail_reason!r} found "
            f"after {max_retries} retries (wait={flush_wait_seconds}s each). "
            f"Log tail:\n{self._captured_log_text()[-2000:]}"
        )

    def assert_violation_absent(self, *, fail_reason: str) -> None:
        log_text = self._captured_log_text()
        if _log_contains_violation(log_text=log_text, fail_reason=fail_reason):
            raise AssertionError(
                f"Unexpected kv_canary violation line with fail_reason={fail_reason!r}. "
                f"Log tail:\n{log_text[-2000:]}"
            )

    def assert_log_contains(self, substring: str) -> None:
        log_text = self._captured_log_text()
        if substring not in log_text:
            raise AssertionError(
                f"Expected substring {substring!r} not found in captured log. "
                f"Log tail:\n{log_text[-2000:]}"
            )


def _log_contains_violation(*, log_text: str, fail_reason: str) -> bool:
    for match in _VIOLATION_LINE_RE.finditer(log_text):
        reason_field = match.group(2)
        if fail_reason in reason_field.split("+"):
            return True
    return False
