"""
Usage:
python3 -m unittest test_pp_single_node.TestPPAccuracy.test_gsm8k
python3 -m unittest test_pp_single_node.TestDPAttentionDP2PP2.test_gsm8k
python3 -m unittest test_pp_single_node.TestGemma4PPAccuracy.test_gsm8k
python3 -m unittest test_pp_single_node.TestGemma4PPAccuracy.test_mmmu
python3 -m unittest test_pp_single_node.TestGemma4PLEPPAccuracy.test_gsm8k
python3 -m unittest test_pp_single_node.TestPPMixedChunk.test_gsm8k
python3 -m unittest test_pp_single_node.TestFixedBugs.test_chunked_prefill_with_small_bs
"""

import time
import unittest
from types import SimpleNamespace

import requests

from sglang.bench_one_batch_server import BenchArgs as OneBatchBenchArgs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_GEMMA4_PLE_PP,
    DEFAULT_MODEL_NAME_FOR_TEST_GEMMA4_PP,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
    run_bench_one_batch_server,
)

register_cuda_ci(est_time=507, stage="base-c", runner_config="4-gpu-h100")
register_amd_ci(est_time=500, suite="stage-c-test-4-gpu-amd")


class TestPPAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23333"
        cls.process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                2,
                "--pp-size",
                2,
                "--chunked-prefill-size",
                256,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_amd_ci():
            # AMD triton backend produces slightly lower accuracy than FA3 on NVIDIA
            self.assertGreater(metrics["score"], 0.70)
        else:
            self.assertGreater(metrics["score"], 0.74)
        # Wait a little bit so that the memory check happens.
        time.sleep(4)

    def test_logprob(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
                "return_logprob": True,
                "top_logprobs_num": 5,
                "logprob_start_len": 0,
            },
        )
        response_json = response.json()
        input_token_logprobs = response_json["meta_info"]["input_token_logprobs"]
        output_token_logprobs = response_json["meta_info"]["output_token_logprobs"]
        output_top_logprobs = response_json["meta_info"]["output_top_logprobs"]

        assert len(input_token_logprobs) == 6
        assert len(output_token_logprobs) == 16
        assert len(output_top_logprobs) == 16


@unittest.skipIf(is_in_amd_ci(), "MLA model with DP attention not yet supported on AMD")
class TestDPAttentionDP2PP2(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--pp-size",
                "2",
                "--enable-dp-attention",
                "--dp",
                "2",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.8)


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


class TestPPMixedChunk(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = "http://127.0.0.1:23338"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                2,
                "--pp-size",
                2,
                "--chunked-prefill-size",
                256,
                "--enable-mixed-chunk",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

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

        if is_in_amd_ci():
            # AMD triton backend produces slightly lower accuracy than FA3 on NVIDIA
            self.assertGreater(metrics["score"], 0.70)
        else:
            self.assertGreater(metrics["score"], 0.74)
        # Wait a little bit so that the memory check happens.
        time.sleep(4)


class TestFixedBugs(unittest.TestCase):
    def test_chunked_prefill_with_small_bs(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST
        server_args = ServerArgs(model_path=model)
        bench_args = OneBatchBenchArgs(
            batch_size=(1,),
            input_len=(1,),
            output_len=(1,),
            base_url=DEFAULT_URL_FOR_TEST,
        )
        other_server_args = [
            "--tp-size",
            2,
            "--pp-size",
            2,
            "--chunked-prefill-size",
            256,
            "--max-running-requests",
            2,
        ]
        run_bench_one_batch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            server_args,
            bench_args,
            other_server_args,
        )


if __name__ == "__main__":
    unittest.main()
