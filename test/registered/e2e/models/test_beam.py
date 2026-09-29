"""Native Beam serving on the deterministic fixture across parallel layouts.

Prefill, cached decode and every parallel layout (TP2, TP2 with EP2, PP2, TP4,
attention DP over TP2/TP4) are compared against the eager BF16 oracle and,
teacher-forced, against the TP1 run. Layouts needing more GPUs than present
are skipped. Single-GPU precision cases live in
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

# Cover FMHA-v2/XQA on Hopper and TRT-LLM-GEN on Blackwell; TP4 runs only on
# the four-GPU Blackwell runner.
# ci-cost-override: the smallest Blackwell runner has four GPUs, and each
# parallel layout needs its own engine launch.
register_cuda_ci(est_time=600, stage="base-b", runner_config="2-gpu-large")
register_cuda_ci(est_time=600, stage="base-b", runner_config="4-gpu-b200")

# The fixture has 12 query heads and 2 KV heads, so TP8 cannot shard it;
# TP8 and EP8 are full-checkpoint evidence, not fixture cases. The graph
# variant of PP2 replays the pipeline proxy tensors through CUDA graphs,
# which fill a residual slot Beam never sends.
PARALLEL_LAYOUTS = (
    ("tp1", dict(tp_size=1)),
    ("tp2", dict(tp_size=2)),
    ("tp2_ep2", dict(tp_size=2, ep_size=2)),
    ("pp2", dict(pp_size=2)),
    ("pp2_graphs", dict(pp_size=2, disable_cuda_graph=False)),
    ("tp4", dict(tp_size=4)),
    # With EP2, TP == DP == EP reduce-scatters the MoE output instead of
    # dp_scatter; DP2 x TP4 keeps attention TP2 inside each DP rank.
    ("dp2_tp2", dict(tp_size=2, dp_size=2, enable_dp_attention=True)),
    (
        "dp2_tp2_ep2_dp_lm_head",
        dict(
            tp_size=2,
            dp_size=2,
            ep_size=2,
            enable_dp_attention=True,
            enable_dp_lm_head=True,
        ),
    ),
    ("dp2_tp4", dict(tp_size=4, dp_size=2, enable_dp_attention=True)),
)


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

    def test_prefill_cached_decode_and_parallel_layouts(self):
        tp1_results = {}
        for index, (layout, parallel) in enumerate(PARALLEL_LAYOUTS):
            gpus = parallel.get("tp_size", 1) * parallel.get("pp_size", 1)
            if gpus > torch.cuda.device_count():
                print(f"{layout}: skipped, needs {gpus} GPUs", flush=True)
                continue
            kwargs = dict(
                model_path=self.checkpoint.name,
                skip_tokenizer_init=True,
                dtype="bfloat16",
                attention_backend="trtllm_mha",
                page_size=64,
                moe_runner_backend="auto" if layout == "tp1" else "triton",
                disable_cuda_graph=True,
                disable_radix_cache=True,
                chunked_prefill_size=-1,
                max_total_tokens=4096,
                mem_fraction_static=0.2,
                # Attention-DP engines bind TCP ports derived from `port`, and
                # shutdown leaves them bound in this process.
                port=30000 + 100 * index,
            )
            kwargs.update(parallel)
            engine = sgl.Engine(**kwargs)
            try:
                # 510 and 513 straddle the 512-token sliding window.
                for prompt_length in (6, 510, 513):
                    with self.subTest(layout=layout, prompt_length=prompt_length):
                        prompt = [1 + (i * 17) % 127 for i in range(prompt_length)]
                        generated, actual = self.logprobs(engine, prompt, 4)
                        self.assertEqual(len(generated), 4)
                        if layout == "tp1":
                            tp1_results[prompt_length] = generated, actual
                        logits = eager_beam_logits(
                            self.config, self.state, prompt + generated[:-1]
                        )
                        expected = logits[prompt_length - 1 :].log_softmax(-1)
                        tag = f"{layout} prompt={prompt_length}"
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
                        if layout != "tp1":
                            baseline_ids, baseline_logprobs = tp1_results[prompt_length]
                            # Teacher-force TP1's continuation so no layout
                            # compares distributions for different prefixes.
                            for step in range(4):
                                with self.subTest(
                                    comparison="tp1_vs_layout", step=step
                                ):
                                    _, fresh = self.logprobs(
                                        engine, prompt + baseline_ids[:step], 1
                                    )
                                    assert_logprobs_close(
                                        self,
                                        fresh[0],
                                        baseline_logprobs[step],
                                        tag=f"{tag} vs tp1 step={step}",
                                    )
            finally:
                engine.shutdown()


if __name__ == "__main__":
    unittest.main()
