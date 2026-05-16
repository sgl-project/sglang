"""STANDALONE speculative-decoding server fixture.

Variants combine this base with `CustomTestCase` and override class
attributes (`attention_backend`, plus optional `speculative_eagle_topk` /
`speculative_num_draft_tokens` / `enable_spec_v2`) to select a backend
and the V1 / V2 spec engine.

Pure mixin (does NOT inherit `TestCase`), so unittest does not collect
the base itself.
"""

from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_STANDALONE,
    DEFAULT_TARGET_MODEL_STANDALONE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

GSM_DATASET_PATH = None


class StandaloneServerBase:
    model = DEFAULT_TARGET_MODEL_STANDALONE
    draft_model = DEFAULT_DRAFT_MODEL_STANDALONE
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.69
    spec_decode_threshold = 3.6

    # Subclasses set these:
    attention_backend: str = ""
    # V2 defaults; V1 subclasses override to (2, 7, False).
    speculative_num_steps: int = 4
    speculative_eagle_topk: int = 1
    speculative_num_draft_tokens: int = 5
    enable_spec_v2: bool = True

    @classmethod
    def get_server_args(cls):
        assert cls.attention_backend, f"{cls.__name__} must set `attention_backend`"
        return [
            "--trust-remote-code",
            "--cuda-graph-max-bs",
            "8",
            "--speculative-algorithm",
            "STANDALONE",
            "--speculative-draft-model-path",
            DEFAULT_DRAFT_MODEL_STANDALONE,
            "--speculative-num-steps",
            str(cls.speculative_num_steps),
            "--speculative-eagle-topk",
            str(cls.speculative_eagle_topk),
            "--speculative-num-draft-tokens",
            str(cls.speculative_num_draft_tokens),
            "--mem-fraction-static",
            0.7,
            "--attention-backend",
            cls.attention_backend,
        ]

    @classmethod
    def setUpClass(cls):
        # disable deep gemm precompile to make launch server faster
        # please don't do this if you want to make your inference workload faster
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)
        if not cls.enable_spec_v2:
            envs.SGLANG_ENABLE_SPEC_V2.set(False)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        if not cls.enable_spec_v2:
            envs.SGLANG_ENABLE_SPEC_V2.clear()

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=128,
            num_shots=4,
            gsm8k_data_path=GSM_DATASET_PATH,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        metric_key = "score"
        self.assertGreaterEqual(metrics[metric_key], self.accuracy_threshold)

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, self.spec_decode_threshold)
