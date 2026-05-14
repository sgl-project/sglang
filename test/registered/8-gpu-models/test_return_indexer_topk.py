import asyncio
import logging
import unittest

import aiohttp
import numpy as np

from sglang.srt.state_capturer.indexer_topk import (
    extract_indexer_topk_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="stage-c", runner_config="8-gpu-h200")

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2"

# V3.2 config — hardcoded for response decoding (mirrors test_return_routed_experts.py).
NUM_INDEXER_LAYERS = 61
INDEX_TOPK = 2048

# index_topk_freq=2 → layers 2,4,6,... reuse layer L-1's topk; exercises the
# forward_mla.py skip_topk capture path.
INDEX_TOPK_FREQ = 2

logger = logging.getLogger(__name__)


class TestReturnIndexerTopk(CustomTestCase):
    """Indexer-topk capture e2e test for DSv3.2 (NSA).

    Single server with `--enable-return-indexer-topk` and `index_topk_freq=2`.
    Validates the native `/generate` endpoint only — OpenAI-protocol surface
    (`SglExt.indexer_topk`) not yet wired up; follow-up PR.

    Per response, validates:
      1. Captured tensor decodes to (seqlen-1, num_indexer_layers, index_topk).
      2. Indices are positional sentinels in [-1, +inf); -1 marks padding.
      3. With freq=2, layers L in {2,4,6,...} byte-equal layer L-1's slot —
         regression-protects the skip_topk capture path in forward_mla.py.
    """

    @classmethod
    def setUpClass(cls):
        cls.other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--dp",
            "8",
            "--enable-dp-attention",
            "--enable-return-indexer-topk",
            # Cap KV pool so the indexer-topk host buffer (488 KB / token for
            # V3.2) stays bounded; with the default ~600k tokens × 8 procs the
            # pinned allocation runs into TB-scale and OOMs the CI host.
            "--max-total-tokens",
            "32768",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
            "--json-model-override-args",
            f'{{"index_topk_freq": {INDEX_TOPK_FREQ}}}',
        ]
        cls.sampling_args = {"temperature": 0, "max_new_tokens": 16}
        cls.texts = [
            "What is the capital of France?",
            "Solve: 2 + 3 = ?",
        ]
        cls.process = popen_launch_server(
            DEEPSEEK_V32_MODEL_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )
        try:
            cls.captured = asyncio.run(cls._collect_async())
        except Exception:
            kill_process_tree(cls.process.pid)
            raise

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_indexer_topk_generate(self):
        for topk in self.captured:
            self._check_shape_and_range(topk)
            self._check_skip_topk_equality(topk)

    def _check_shape_and_range(self, topk: np.ndarray):
        self.assertEqual(topk.ndim, 3)
        seqlen_minus_1, num_layers, topk_size = topk.shape
        self.assertGreater(seqlen_minus_1, 0)
        self.assertEqual(num_layers, NUM_INDEXER_LAYERS)
        self.assertEqual(topk_size, INDEX_TOPK)
        # Indices are token positions; valid values are >= -1 (-1 = padding sentinel).
        self.assertTrue((topk >= -1).all(), f"min index {topk.min()} < -1")

    def _check_skip_topk_equality(self, topk: np.ndarray):
        """Layers L in {2, 4, 6, ...} must byte-equal layer L-1 with freq=2.

        With `skip_topk = max(layer_id - 1, 0) % freq != 0`, freq=2 yields
        skip=True for L >= 2 with L-1 odd → L even (>= 2). The forward_mla.py
        skip-path mirrors prev_topk_indices into layer L's slot.
        """
        for L in range(2, NUM_INDEXER_LAYERS, 2):
            np.testing.assert_array_equal(
                topk[:, L, :],
                topk[:, L - 1, :],
                err_msg=f"layer {L} should reuse layer {L - 1}'s topk (skip_topk path)",
            )

    @classmethod
    async def _collect_async(cls):
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(
                    make_request(
                        session,
                        f"{DEFAULT_URL_FOR_TEST}/generate",
                        {
                            "text": text,
                            "sampling_params": cls.sampling_args,
                            "return_indexer_topk": True,
                        },
                    )
                )
                for text in cls.texts
            ]
            http_results = await asyncio.gather(*tasks)
            # Reshape raw int32 bytes into (seqlen-1, num_indexer_layers, index_topk).
            return [
                extract_indexer_topk_from_meta_info(res).reshape(
                    -1, NUM_INDEXER_LAYERS, INDEX_TOPK
                )
                for res in http_results
            ]


async def make_request(session, url, payload):
    async with session.post(url=url, json=payload) as response:
        return await response.json()


if __name__ == "__main__":
    unittest.main()
