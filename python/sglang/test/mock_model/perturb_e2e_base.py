from __future__ import annotations

import io
import os
from typing import ClassVar

from sglang.test.kv_canary.e2e_base import CapturedServerE2EBase
from sglang.test.kv_canary.utils import post_parallel_generate
from sglang.test.mock_model.utils import (
    MOCK_MODEL_PATH,
    mock_model_server_args,
    mock_model_server_env,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


class MockModelPerturbE2EBase(CapturedServerE2EBase):
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
