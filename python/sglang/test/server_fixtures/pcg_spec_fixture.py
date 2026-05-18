"""Piecewise CUDA Graph + speculative decoding test fixture.

Each variant tests PCG coexisting with one speculative-decoding algorithm
(EAGLE3 / NEXTN / STANDALONE / NGRAM). Variants differ widely on model /
server args / thresholds, so the base only abstracts the common shape:
  - launch a server with `server_args` (variant-supplied list)
  - run gsm8k, assert `score > accuracy_threshold`
  - read `avg_spec_accept_length` from /server_info, assert
    `> speedup_threshold`

Pure mixin (does NOT inherit `TestCase`), so unittest does not collect
the base itself.
"""

from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class PCGSpecBase:
    # Subclasses must set:
    model: str = ""
    server_args: list = []

    # Optional knobs (variant defaults override):
    timeout_mult: int = 2
    server_env: dict = None  # passed to popen_launch_server `env=...`
    accuracy_threshold: float = 0.70
    speedup_threshold: float = 1.5
    max_tokens: int = 512
    thinking_mode: str = ""  # set to e.g. "qwen3" if needed

    @classmethod
    def setUpClass(cls):
        assert (
            cls.model and cls.server_args
        ), f"{cls.__name__} must set `model` and `server_args`"
        cls.base_url = DEFAULT_URL_FOR_TEST
        kwargs = dict(
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * cls.timeout_mult,
            other_args=cls.server_args,
        )
        if cls.server_env:
            kwargs["env"] = cls.server_env
        cls.process = popen_launch_server(cls.model, cls.base_url, **kwargs)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        eval_kwargs = dict(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            max_tokens=self.max_tokens,
            num_examples=200,
            num_threads=200,
        )
        if self.thinking_mode:
            eval_kwargs["thinking_mode"] = self.thinking_mode
        args = SimpleNamespace(**eval_kwargs)
        metrics = run_eval(args)
        print(metrics)
        self.assertGreater(metrics["score"], self.accuracy_threshold)

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, self.speedup_threshold)
