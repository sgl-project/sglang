import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_rust_server_built,
    popen_launch_server,
)

# Two classes run from this file: the default server plus the Rust-frontend
# variant (when the embedded extension is built), each launches a server + eval.
register_cuda_ci(est_time=106, stage="base-b", runner_config="1-gpu-large")


class TestModeloptFP8(CustomTestCase):
    # Extra server env; the Rust-frontend subclass sets SGLANG_RUST_SERVER here.
    env = None
    # Eval endpoint. The Rust server exposes only the native `/generate`, so its
    # subclass overrides this to "generate".
    api = "completion"

    @classmethod
    def setUpClass(cls):
        cls.model = "nvidia/Llama-3.1-8B-Instruct-FP8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--quantization",
                "modelopt_fp8",
                "--tokenizer-worker-num",
                "2",
                "--detokenizer-worker-num",
                "2",
            ],
            env=cls.env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        parsed_url = urlparse(self.base_url)
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api=self.api,
            max_tokens=512,
            num_examples=200,
            num_threads=200,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.70)


@unittest.skipUnless(
    is_rust_server_built(),
    "embedded rust server extension not built",
)
class TestModeloptFP8WithRustServer(TestModeloptFP8):
    """Same model + eval, but served through the embedded Rust frontend
    (`SGLANG_RUST_SERVER`). Guards the Rust tokenizer/detokenizer/completions path
    against accuracy regressions: a bug there drops gsm8k score below the same
    0.70 bar the default frontend must clear. Uses the native `/generate` endpoint
    (the only API the Rust server exposes)."""

    env = {"SGLANG_RUST_SERVER": "1"}
    api = "generate"


if __name__ == "__main__":
    unittest.main()
