"""Hybrid radix cache, chunked prefill, CUDA graphs and concurrency for native Beam.

Every comparison flips one variable against the eager BF16 oracle or against
another engine run on the deterministic fixture, and every gate is the fixture's
per-token KL bound. Random weights carry no quality signal; these cases guard
the serving paths a single-request eager test never enters: a cold prefix
restored from the radix cache, a partial hit that stops inside the sliding
window, a prefix evicted under pool pressure and recomputed, a long decode
across the window and the SWA eviction interval, chunk boundaries, graph
replay, and a mixed-length concurrent batch.
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
    make_beam_fixture,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Cover FMHA-v2/XQA on Hopper and TRT-LLM-GEN graph replay on Blackwell.
# ci-cost-override: the smallest Blackwell runner has four GPUs.
register_cuda_ci(est_time=420, stage="base-b", runner_config="1-gpu-large")
register_cuda_ci(est_time=420, stage="base-b", runner_config="4-gpu-b200")

# The fixture's window is 512 tokens; 510/513 straddle it and 1200 is long
# enough for chunked prefill (256), page 64 alignment and partial hits.
PROMPT_LENGTHS = (6, 510, 513, 1200)
LONG_PROMPT = 1200
PARTIAL_PREFIX = 700
# Four SWA eviction intervals (128 tokens) past the window.
LONG_DECODE = 600
BATCH_LENGTHS = (6, 100, 300, 510, 513, 700, 900, 1200)
# One request that cannot fit beside the cached prompts in a 4096-token pool.
EVICTION_FILLER = 3800


def prompt(length, offset=0):
    return [1 + ((i + offset) * 17) % 127 for i in range(length)]


class TestBeamCacheParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Beam cache test requires a CUDA GPU")
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

    def launch(self, **overrides):
        kwargs = dict(
            model_path=self.checkpoint.name,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            tp_size=1,
            attention_backend="trtllm_mha",
            page_size=64,
            chunked_prefill_size=256,
            max_total_tokens=4096,
            mem_fraction_static=0.2,
        )
        kwargs.update(overrides)
        return sgl.Engine(**kwargs)

    def generate(self, engine, token_ids, count):
        """Greedy ids, full-vocab logprobs and the radix hit for one or many prompts."""
        batched = isinstance(token_ids[0], list)
        results = engine.generate(
            input_ids=token_ids,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": count,
                "ignore_eos": True,
            },
            return_logprob=True,
            token_ids_logprob=list(range(self.config.vocab_size)),
        )
        if not batched:
            results = [results]
        outputs = []
        for result in results:
            logprobs = torch.tensor(
                [
                    [entry[0] for entry in row]
                    for row in result["meta_info"]["output_token_ids_logprobs"]
                ],
                device="cuda",
            )
            outputs.append(
                (result["output_ids"], logprobs, result["meta_info"]["cached_tokens"])
            )
        return outputs if batched else outputs[0]

    def oracle(self, token_ids, generated):
        logits = eager_beam_logits(self.config, self.state, token_ids + generated[:-1])
        return logits[len(token_ids) - 1 :].log_softmax(-1)

    def teacher_forced(self, engine, token_ids, ids):
        """Logprobs of ``ids`` after ``token_ids``: one greedy run when it agrees."""
        generated, logprobs, _ = self.generate(engine, token_ids, len(ids))
        if generated == ids:
            return logprobs
        rows = []
        for step in range(len(ids)):
            _, fresh, _ = self.generate(engine, token_ids + ids[:step], 1)
            rows.append(fresh[0])
        return torch.stack(rows)

    def test_cache_chunking_graphs_and_concurrency(self):
        eager = self.launch(disable_cuda_graph=True)
        try:
            cold = self.run_eager_cases(eager)
        finally:
            eager.shutdown()
        serving = self.launch()
        try:
            self.run_graph_cases(serving, cold)
        finally:
            serving.shutdown()

    def run_eager_cases(self, engine):
        """Radix hits, eviction, chunking, long decode and batching on eager execution."""
        # Warm the engine once so the first flush never races startup.
        self.generate(engine, prompt(6), 1)
        cold = {}
        for length in PROMPT_LENGTHS:
            with self.subTest(case="cold_chunked_vs_oracle", prompt=length):
                engine.flush_cache()
                ids, logprobs, cached = self.generate(engine, prompt(length), 4)
                self.assertEqual(cached, 0, "flushed cache still reports a hit")
                # test_beam.py runs the same oracle against unchunked prefill.
                assert_logprobs_close(
                    self,
                    logprobs,
                    self.oracle(prompt(length), ids),
                    tag=f"eager chunked cold prompt={length}",
                )
                cold[length] = (ids, logprobs)

        long_ids, long_logprobs = cold[LONG_PROMPT]
        with self.subTest(case="full_hit_vs_cold"):
            ids, logprobs, cached = self.generate(engine, prompt(LONG_PROMPT), 4)
            self.assertGreater(cached, 0, "second submission missed the radix cache")
            self.assertEqual(ids, long_ids)
            assert_logprobs_close(self, logprobs, long_logprobs, tag="full radix hit")

        with self.subTest(case="partial_hit_vs_oracle"):
            branch = prompt(PARTIAL_PREFIX) + prompt(64, offset=5000)
            ids, logprobs, cached = self.generate(engine, branch, 4)
            # The radix cache hands back whole pages of the shared prefix.
            self.assertEqual(cached, PARTIAL_PREFIX // 64 * 64)
            assert_logprobs_close(
                self, logprobs, self.oracle(branch, ids), tag="partial radix hit"
            )

        with self.subTest(case="evicted_prefix_vs_cold"):
            filler = prompt(EVICTION_FILLER, offset=9000)
            _, _, cached = self.generate(engine, filler, 4)
            self.assertEqual(cached, 0)
            ids, logprobs, cached = self.generate(engine, prompt(LONG_PROMPT), 4)
            self.assertEqual(cached, 0, "filler did not evict the prompt")
            self.assertEqual(ids, long_ids)
            assert_logprobs_close(
                self, logprobs, long_logprobs, tag="recomputed after eviction"
            )

        with self.subTest(case="long_decode_vs_oracle"):
            ids, logprobs, _ = self.generate(engine, prompt(LONG_PROMPT), LONG_DECODE)
            self.assertEqual(len(ids), LONG_DECODE)
            assert_logprobs_close(
                self,
                logprobs,
                self.oracle(prompt(LONG_PROMPT), ids),
                tag=f"eager decode {LONG_DECODE} tokens",
            )

        with self.subTest(case="batch_vs_solo"):
            prompts = [
                prompt(length, offset=100 * i) for i, length in enumerate(BATCH_LENGTHS)
            ]
            solo = []
            for tokens in prompts:
                engine.flush_cache()
                solo.append(self.generate(engine, tokens, 8))
            engine.flush_cache()
            batch = self.generate(engine, prompts, 8)
            for length, (ids, logprobs, _), (batch_ids, batch_logprobs, _) in zip(
                BATCH_LENGTHS, solo, batch
            ):
                self.assertEqual(
                    batch_ids, ids, f"batched ids differ at prompt={length}"
                )
                assert_logprobs_close(
                    self, batch_logprobs, logprobs, tag=f"batched prompt={length}"
                )
        return cold

    def run_graph_cases(self, engine, cold):
        """Prefill and decode graph replay against the eager runs and the oracle."""
        self.generate(engine, prompt(6), 1)
        for length in PROMPT_LENGTHS:
            with self.subTest(case="graph_vs_eager", prompt=length):
                engine.flush_cache()
                ids, logprobs = cold[length]
                assert_logprobs_close(
                    self,
                    self.teacher_forced(engine, prompt(length), ids),
                    logprobs,
                    tag=f"graph vs eager prompt={length}",
                )
        with self.subTest(case="graph_long_decode_vs_oracle"):
            engine.flush_cache()
            ids, logprobs, _ = self.generate(engine, prompt(LONG_PROMPT), LONG_DECODE)
            assert_logprobs_close(
                self,
                logprobs,
                self.oracle(prompt(LONG_PROMPT), ids),
                tag=f"graph decode {LONG_DECODE} tokens",
            )


if __name__ == "__main__":
    unittest.main()
