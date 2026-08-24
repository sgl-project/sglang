import time
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_GEMMA4_PLE_PP,
    DEFAULT_MODEL_NAME_FOR_TEST_GEMMA4_PP,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
)

# tp=1 pp=2 -- two GPUs.
register_cuda_ci(est_time=203, stage="base-b", runner_config="2-gpu-large")


@unittest.skipIf(
    is_in_amd_ci(),
    "Gemma4 PP not yet validated on AMD",
)
class TestGemma4PPAccuracy(unittest.TestCase):
    """End-to-end PP=2 accuracy gate for Gemma4 multimodal.

    Gemma4 has full-attention layers with head_dim=512 (FA's max is 256), so
    sglang auto-selects the triton attention backend; no manual flag needed.
    The 26B BF16 model splits to ~26 GB per stage under PP=2, well within an
    H100's 80 GB.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_GEMMA4_PP
        cls.base_url = "http://127.0.0.1:23333"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                1,
                "--pp-size",
                2,
                "--trust-remote-code",
                "--enable-multimodal",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        # Gemma4 is instruction-tuned and doesn't follow few-shot completion
        # prompts well — use the chat API (default in run_eval), which scores
        # ~0.98 on this model vs ~0.44 with api="completion".
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=200,
            num_threads=32,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        # Chat-API baseline ~0.98; gate well below to absorb sample-noise
        # without missing a real PP-routing regression (pre-PP-fix the model
        # produced garbage outputs scoring ≈ 0).
        self.assertGreaterEqual(metrics["score"], 0.90)
        # Wait a little bit so that the memory check happens.
        time.sleep(4)

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
    def test_mmmu(self):
        # Multimodal accuracy gate covering the vision_tower → embed_vision
        # (first rank) → PP-proxy handoff → LM tail (last rank) chain.
        # Measured 0.71 on 200 examples; full eval (~900 questions) takes
        # ~5-7 min on H100 so this is manual-only.
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmmu",
            num_examples=None,
            num_threads=32,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        # Measured 0.72 on this setup; published Gemma-4-26B MMMU lies in
        # 0.69-0.73.  Gate 0.65 leaves ~5 SE of headroom (SE on 900 binary
        # samples ≈ 0.015) while still catching mid-grade vision/PP
        # regressions, not just complete breakage.
        self.assertGreater(metrics["score"], 0.65)


@unittest.skipIf(
    is_in_amd_ci(),
    "Gemma4 PP not yet validated on AMD",
)
class TestGemma4PLEPPAccuracy(unittest.TestCase):
    """PP=2 coverage for Gemma4 PLE variants (per_layer_inputs proxy path).

    26B-A4B has ``hidden_size_per_layer_input=0`` so the default Gemma4 PP
    test never crosses the PLE branch.  Cuda graph + PLE corrupts outputs
    (the runner's hardcoded ``{hidden_states, residual}`` PP-proxy schema
    drops ``per_layer_inputs``), so this test pins the eager configuration.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_GEMMA4_PLE_PP
        cls.base_url = "http://127.0.0.1:23339"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                1,
                "--pp-size",
                2,
                "--trust-remote-code",
                "--enable-multimodal",
                # Required for PLE under PP — see Gemma4TextModel guard.
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        # Eager-path baseline ~0.92; gate 0.80 catches PLE breakage
        # (corruption collapses score to ~0).
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=100,
            num_threads=32,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["score"], 0.80)
        time.sleep(4)


if __name__ == "__main__":
    unittest.main()
