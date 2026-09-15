"""Native Beam serving on the deterministic fixture at TP=1 and TP=2.

Prefill, cached decode and tensor parallelism are compared against the eager
BF16 oracle and against each other. Single-GPU precision cases live in
test/registered/e2e/models/test_beam_numerics.py.
"""

import tempfile
import unittest

import torch

import sglang as sgl
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import (
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
)
from sglang.test.beam_fixture import (
    assert_logprobs_close,
    eager_beam_logits,
    engine_token_logprobs,
    make_beam_fixture,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Cover FMHA-v2/XQA on Hopper and TRT-LLM-GEN on Blackwell at TP1/TP2.
# ci-cost-override: the smallest Blackwell runner has four GPUs.
register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")
register_cuda_ci(est_time=300, stage="base-b", runner_config="4-gpu-b200")


class TestBeamNative(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("Beam TP differential test requires two CUDA GPUs")
        if not is_flashinfer_available() or not (
            is_sm90_supported() or is_sm100_supported() or is_sm120_supported()
        ):
            raise unittest.SkipTest(
                "Beam TRT-LLM MHA test requires FlashInfer and SM90, SM100, or SM120"
            )
        cls.checkpoint = tempfile.TemporaryDirectory(prefix="beam-fixture-")
        cls.config, state = make_beam_fixture(cls.checkpoint.name)
        cls.state = {name: value.cuda() for name, value in state.items()}

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "state"):
            del cls.state
        if hasattr(cls, "checkpoint"):
            cls.checkpoint.cleanup()

    def logprobs(self, engine, token_ids, count):
        return engine_token_logprobs(engine, token_ids, count, self.config.vocab_size)

    def test_prefill_cached_decode_and_tensor_parallelism(self):
        tp1_results = {}
        for tp_size in (1, 2):
            engine = sgl.Engine(
                model_path=self.checkpoint.name,
                skip_tokenizer_init=True,
                dtype="bfloat16",
                tp_size=tp_size,
                attention_backend="trtllm_mha",
                page_size=64,
                moe_runner_backend="auto" if tp_size == 1 else "triton",
                disable_cuda_graph=True,
                disable_radix_cache=True,
                chunked_prefill_size=-1,
                max_total_tokens=4096,
                mem_fraction_static=0.2,
            )
            try:
                # 510 and 513 straddle the 512-token sliding window.
                for prompt_length in (6, 510, 513):
                    with self.subTest(tp_size=tp_size, prompt_length=prompt_length):
                        prompt = [1 + (i * 17) % 127 for i in range(prompt_length)]
                        generated, actual = self.logprobs(engine, prompt, 4)
                        self.assertEqual(len(generated), 4)
                        if tp_size == 1:
                            tp1_results[prompt_length] = generated, actual
                        logits = eager_beam_logits(
                            self.config, self.state, prompt + generated[:-1]
                        )
                        expected = logits[prompt_length - 1 :].log_softmax(-1)
                        tag = f"tp={tp_size} prompt={prompt_length}"
                        with self.subTest(comparison="reference"):
                            assert_logprobs_close(
                                self, actual, expected, tag=f"{tag} reference"
                            )
                        for step in range(1, 4):
                            with self.subTest(comparison="cached_vs_fresh", step=step):
                                _, fresh = self.logprobs(
                                    engine, prompt + generated[:step], 1
                                )
                                assert_logprobs_close(
                                    self,
                                    actual[step],
                                    fresh[0],
                                    tag=f"{tag} cached step={step}",
                                )
                        if tp_size == 2:
                            baseline_ids, baseline_logprobs = tp1_results[prompt_length]
                            # Teacher-force TP1's continuation so TP never
                            # compares distributions for different prefixes.
                            for step in range(4):
                                with self.subTest(comparison="tp1_vs_tp2", step=step):
                                    _, fresh = self.logprobs(
                                        engine, prompt + baseline_ids[:step], 1
                                    )
                                    assert_logprobs_close(
                                        self,
                                        fresh[0],
                                        baseline_logprobs[step],
                                        tag=f"{tag} TP step={step}",
                                    )
            finally:
                engine.shutdown()


if __name__ == "__main__":
    unittest.main()
