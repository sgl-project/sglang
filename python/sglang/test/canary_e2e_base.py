"""Shared base class for canary e2e tests.

Server-launch / teardown / parallel-burst boilerplate factored out of
``test_e2e_qwen3.py`` and ``test_e2e_dsv4_flash.py`` so each test only
declares its model + extra server args + the actual assertions.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import ClassVar, Dict, List, Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_DEFAULT_PROMPTS: List[str] = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Explain in one sentence what a transformer is.",
    "1 + 1 =",
]


class CanaryE2EBase(CustomTestCase):
    """Common scaffold for canary server-launch e2e tests.

    Subclasses set ``model`` and (optionally) ``extra_server_args``,
    ``perturb_prob``, ``perturb_seed``. ``setUpClass`` launches the server
    with ``--kv-cache-canary=raise`` and ``--mem-fraction-static=0.65``
    (shadow K/V tensors need headroom); subclasses can layer more args on
    via ``extra_server_args``.

    Use :meth:`send_parallel_requests` to fan out N concurrent
    /generate calls — that's the only way to exercise concurrent
    prefill+decode batches the canary may behave differently on.

    Set ``allow_launch_failure = True`` if the test expects the server to
    fail to come up (e.g. perturb+raise tripping in warmup); in that case
    ``launch_failed`` will be ``True`` and the launch exception is stored
    in ``launch_exception``.
    """

    model: ClassVar[str] = ""
    extra_server_args: ClassVar[List[str]] = []
    perturb_prob: ClassVar[float] = 0.0
    perturb_seed: ClassVar[int] = 0
    allow_launch_failure: ClassVar[bool] = False

    base_url: ClassVar[str] = ""
    process: ClassVar[Optional[object]] = None
    launch_failed: ClassVar[bool] = False
    launch_exception: ClassVar[Optional[BaseException]] = None

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.model:
            raise RuntimeError(
                f"{cls.__name__}: subclass must set `model` to a HF model id"
            )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = None
        cls.launch_failed = False
        cls.launch_exception = None

        env = os.environ.copy()
        if cls.perturb_prob > 0:
            env["SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB"] = str(cls.perturb_prob)
            env["SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_SEED"] = str(cls.perturb_seed)
        else:
            env.pop("SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB", None)
            env.pop("SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_SEED", None)

        other_args: List[str] = [
            "--kv-cache-canary",
            "raise",
            "--mem-fraction-static",
            "0.65",
            *cls.extra_server_args,
        ]

        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                env=env,
            )
        except Exception as exc:
            if not cls.allow_launch_failure:
                raise
            cls.launch_failed = True
            cls.launch_exception = exc

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.process:
            kill_process_tree(cls.process.pid)

    def send_parallel_requests(
        self,
        n: int,
        *,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 16,
        max_workers: int = 16,
        timeout: float = 60.0,
    ) -> List[Dict[str, object]]:
        """Fire N /generate requests concurrently; return raw response dicts.

        Each thread sends one request; ``prompts`` is round-robined so the
        server sees varied input lengths. ``status_code`` and ``text`` are
        preserved on each entry so the caller can drive its own assertions.
        """
        if prompts is None:
            prompts = _DEFAULT_PROMPTS
        if not prompts:
            raise ValueError("prompts must be non-empty")

        def _one(i: int) -> Dict[str, object]:
            payload = {
                "text": f"{prompts[i % len(prompts)]} {i}",
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                },
            }
            try:
                resp = requests.post(
                    self.base_url + "/generate", json=payload, timeout=timeout
                )
                return {"index": i, "status_code": resp.status_code, "text": resp.text}
            except requests.exceptions.RequestException as exc:
                return {"index": i, "error": repr(exc)}

        results: List[Dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_one, i) for i in range(n)]
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda r: r["index"])
        return results

    def assert_health_ok(self, *, timeout: float = 10.0) -> None:
        """Sanity-check /health responds 200 (post-traffic liveness)."""
        resp = requests.get(self.base_url + "/health", timeout=timeout)
        self.assertEqual(resp.status_code, 200, resp.text)
