"""
Usage:
python3 -m unittest test_pp_single_node_extra.TestQwenVLPPAccuracy.test_gsm8k
python3 -m unittest test_pp_single_node_extra.TestQwenPPAccuracy.test_pp_consistency
python3 -m unittest test_pp_single_node_extra.TestQwenPPTieWeightsAccuracy.test_pp_consistency
python3 -m unittest test_pp_single_node_extra.TestQwenMoePPAccuracy.test_pp_consistency
python3 -m unittest test_pp_single_node_extra.TestQwen35PPAccuracy.test_pp_consistency
python3 -m unittest test_pp_single_node_extra.TestGLM41VPPAccuracy.test_mmmu
"""

import time
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP,
    DEFAULT_MODEL_NAME_FOR_TEST_VL_PP,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=368, stage="extra-b", runner_config="4-gpu-h100")
register_amd_ci(est_time=350, suite="stage-c-test-4-gpu-amd")


@unittest.skipIf(
    is_in_amd_ci(),
    "VLM PP accuracy too low on AMD (0.48-0.50 with both aiter and triton)",
)
class TestQwenVLPPAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_VL_PP
        cls.base_url = "http://127.0.0.1:23333"
        cls.process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST_VL_PP,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                1,
                "--pp-size",
                4,
                "--chunked-prefill-size",
                8192,
                "--enable-multimodal",
            ],
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreaterEqual(metrics["score"], 0.65)
        # Wait a little bit so that the memory check happens.
        time.sleep(4)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
    def test_mmmu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmmu",
            num_examples=None,
            num_threads=32,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.26)


class TestQwenPPAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23334"  # different ports to avoid conflicts
        cls.model_name = "Qwen/Qwen3-8B"  # replace with your Qwen Model if needed

    def run_gsm8k_test(self, pp_size):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model_name,
                eval_name="gsm8k",
                api="completion",
                max_tokens=512,
                num_examples=512,
                num_threads=128,
            )
            metrics = run_eval(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=2)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["score"], 0.74)
        self.assertGreaterEqual(
            pp_metrics["score"],
            baseline["score"] - 0.02,
            msg=(
                f"PP accuracy dropped more than 2% compared to baseline. "
                f"Baseline: {baseline['score']:.2%}, PP: {pp_metrics['score']:.2%}"
            ),
        )


@unittest.skipIf(is_in_amd_ci(), "PP consistency too flaky on AMD 4-GPU runners")
class TestQwenPPTieWeightsAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23335"  # different ports to avoid conflicts
        cls.model_name = (
            "Qwen/Qwen3-0.6B"  # qwen3 < 8B all have tie_word_embeddings = True
        )

    def run_gsm8k_test(self, pp_size):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model_name,
                eval_name="gsm8k",
                api="completion",
                max_tokens=512,
                num_examples=512,
                num_threads=128,
            )
            metrics = run_eval(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=2)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["score"], 0.38)
        self.assertGreaterEqual(
            pp_metrics["score"],
            baseline["score"] - 0.02,
            msg=(
                f"PP accuracy dropped more than 2% compared to baseline. "
                f"Baseline: {baseline['score']:.2%}, PP: {pp_metrics['score']:.2%}"
            ),
        )


class TestQwenMoePPAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23336"  # different ports to avoid conflicts
        cls.model_name = "Qwen/Qwen3-30B-A3B"  # replace with your Qwen Model if needed

    def run_gsm8k_test(self, pp_size):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model_name,
                eval_name="gsm8k",
                api="completion",
                max_tokens=512,
                num_examples=512,
                num_threads=128,
            )
            metrics = run_eval(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=2)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["score"], 0.74)
        self.assertGreaterEqual(
            pp_metrics["score"],
            baseline["score"] - 0.02,
            msg=(
                f"PP accuracy dropped more than 2% compared to baseline. "
                f"Baseline: {baseline['score']:.2%}, PP: {pp_metrics['score']:.2%}"
            ),
        )


@unittest.skipIf(
    is_in_ci(), "Qwen35 PP consistency too flaky on H100 and AMD 4-GPU runners"
)
class TestQwen35PPAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23337"  # different ports to avoid conflicts
        cls.model_name = (
            "Qwen/Qwen3.5-35B-A3B"  # replace with your Qwen Model if needed
        )

    def run_gsm8k_test(self, tp_size, pp_size):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                tp_size,
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model_name,
                eval_name="gsm8k",
                api="completion",
                max_tokens=512,
                num_examples=512,
                num_threads=128,
            )
            metrics = run_eval(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(tp_size=2, pp_size=1)
        pp_metrics = self.run_gsm8k_test(tp_size=1, pp_size=2)

        print(f"[Qwen35 PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["score"], 0.83)
        self.assertGreaterEqual(
            pp_metrics["score"],
            baseline["score"] - 0.05,
            msg=(
                f"PP accuracy dropped more than 5% compared to baseline. "
                f"Baseline: {baseline['score']:.2%}, PP: {pp_metrics['score']:.2%}"
            ),
        )


@unittest.skipIf(
    is_in_ci(), "Skipping GLM41V PP accuracy test before it gets more stable"
)
class TestGLM41VPPAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                1,
                "--pp-size",
                2,
                "--chunked-prefill-size",
                8192,
                "--enable-multimodal",
                "--reasoning-parser",
                "glm45",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmmu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmmu",
            num_examples=None,
            num_threads=32,
            response_answer_regex=r"<\|begin_of_box\|>(.*)<\|end_of_box\|>",
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.45)


if __name__ == "__main__":
    unittest.main()
