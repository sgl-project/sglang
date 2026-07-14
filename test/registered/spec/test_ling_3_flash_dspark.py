import os

os.environ.setdefault("SGLANG_RAGGED_VERIFY_MODE", "static")

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=7200, stage="base-c", runner_config="4-gpu-h20")

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

TARGET_MODEL = "/root/models/ling-3.0-flash"
DRAFT_MODEL = "/root/models/ling-3.0-flash-dspark-draft"
GSM8K_DATA_PATH = "/root/datasets/gsm8k/test.jsonl"
GSM8K_NUM_EXAMPLES: int | None = None
GSM8K_SCORE_THRESHOLD = 0.90

# Shared target launch knobs for both runs.
COMMON_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "4",
    "--mem-fraction-static",
    "0.55",
    "--max-running-requests",
    "4",
]

# Strict config-mutation guard rejects the bare ``server_args.cuda_graph_bs``
# assignment that the cuda-graph path performs during init_cuda_graphs.
COMMON_ENV = {
    "SGLANG_STRICT_CONFIG_MUTATION": "0",
}


def _gsm8k_args(base_url: str) -> SimpleNamespace:
    return SimpleNamespace(
        base_url=base_url,
        model=TARGET_MODEL,
        eval_name="gsm8k",
        api="completion",
        max_tokens=512,
        num_examples=GSM8K_NUM_EXAMPLES,
        num_threads=128,
        gsm8k_data_path=GSM8K_DATA_PATH,
    )


def _run_gsm8k(test_case) -> dict:
    metrics = run_eval(_gsm8k_args(test_case.base_url))
    print(f"[{type(test_case).__name__}] {metrics=}")
    if is_in_ci():
        write_github_step_summary(
            f"### {type(test_case).__name__}\n"
            f"score={metrics['score']:.4f}\n"
        )
    return metrics


class _Ling3FlashServerMixin:
    """Shared server lifecycle for the ling-3.0-flash runs.

    Not a TestCase (no subclass of unittest.TestCase); the two ``Test*``
    classes mix it in so the base is never collected on its own.
    """

    # Subclasses set these.
    _server_extra_args: list[str] = []
    _server_env: dict = {}

    @classmethod
    def setUpClass(cls):
        cls.model = TARGET_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = dict(COMMON_ENV)
        env.update(cls._server_env)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=COMMON_ARGS + cls._server_extra_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def _run_gsm8k_and_assert(self) -> dict:
        metrics = _run_gsm8k(self)
        self.assertGreaterEqual(metrics["score"], GSM8K_SCORE_THRESHOLD)
        return metrics


class TestLing3FlashBaseline(_Ling3FlashServerMixin, CustomTestCase):
    """Pure-target GSM8K baseline (no spec decoding).

    Establishes the model's standalone accuracy floor against which the DSpark
    spec run is compared.
    """

    def test_gsm8k(self):
        self._run_gsm8k_and_assert()


class TestLing3FlashDSpark(_Ling3FlashServerMixin, CustomTestCase):
    """DSpark spec decoding on the ling-3.0-flash (hybrid KDA) target.

    Guards the DSpark worker's hybrid linear-attention state commit: without
    ``commit_mamba_states_after_verify`` after accept, the Bailing-MoE-V3 target
    diverges from pure-target greedy decoding on multi-step outputs. The score
    must match the no-spec baseline (both >= 0.90), so spec decoding leaves
    accuracy intact.
    """

    _server_extra_args = [
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-draft-model-path",
        DRAFT_MODEL,
    ]
    _server_env = {"SGLANG_RAGGED_VERIFY_MODE": "static"}

    def test_gsm8k(self):
        self._run_gsm8k_and_assert()


if __name__ == "__main__":
    unittest.main()
