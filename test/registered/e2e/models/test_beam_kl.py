"""Prefill-versus-decode logprob consistency for native Beam through the radix cache.

Uses the shared KL helpers on the deterministic Beam fixture: a prefix restored
from the radix cache and a decode continued on a warm cache must reproduce a
fresh prefill's logprobs. This is what catches a sliding-window pool that evicted
something still needed or a cache prefix that does not match a recomputation;
the fixture's precision tests do not exercise the cache at all.

The server prefills in 256-token chunks, so every LongBench prompt crosses the
512-token window and several chunk boundaries before its prefix is reused.
The default breakable prefill CUDA graphs and decode graphs stay enabled; the
eager variant disables both so graph replay and eager execution answer to the
same gate. The scheduler never mixes logprob requests into running decodes, so
test_beam_mixed_chunk.py covers mixed prefill/decode batches.
"""

import tempfile
import unittest

import torch

from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import (
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
)
from sglang.test.beam_fixture import make_beam_fixture
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=900, stage="base-b", runner_config="1-gpu-large")


class TestBeamKL(KLDivergenceMixin, DefaultServerBase):
    kl_div_thres = 0.005
    other_args = [
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "trtllm_mha",
        "--page-size",
        "64",
        "--chunked-prefill-size",
        "256",
        "--mem-fraction-static",
        "0.3",
    ]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Beam KL test requires a CUDA GPU")
        if not is_flashinfer_available() or not (
            is_sm90_supported() or is_sm100_supported() or is_sm120_supported()
        ):
            raise unittest.SkipTest(
                "Beam TRT-LLM MHA test requires FlashInfer and SM90, SM100, or SM120"
            )
        cls.checkpoint = tempfile.TemporaryDirectory(prefix="beam-fixture-")
        make_beam_fixture(cls.checkpoint.name)
        cls.model = cls.checkpoint.name
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            super().tearDownClass()
        if hasattr(cls, "checkpoint"):
            cls.checkpoint.cleanup()


class TestBeamKLEager(TestBeamKL):
    other_args = TestBeamKL.other_args + [
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "disabled",
    ]


class TestBeamKLFlashInferChunked(TestBeamKL):
    """FlashInfer prefix reuse must match fresh prefill across chunk boundaries."""

    other_args = [
        "--attention-backend",
        "flashinfer",
        "--chunked-prefill-size",
        "256",
        "--mem-fraction-static",
        "0.3",
    ]


if __name__ == "__main__":
    unittest.main()
