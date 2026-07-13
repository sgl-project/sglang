"""Integration tests for the full prefill CUDA graph backend.

The Qwen3-8B test checks end-to-end accuracy with FlashInfer. The smaller
DeepSeek-Coder-V2-Lite test checks that an MLA radix-prefix hit selects the
OSS FA4 cached-prefix graph variant and matches an eager cold request.

The attention backend is pinned to flashinfer: plain EXTEND under full
CUDA graph requires the backend's init_forward_metadata_out_graph to
support extend (capture-stable plan state). flashinfer and the
FlashAttention backend (fa4; fa3 untested — needs SM90 hardware)
implement it; the FA4 case is restricted to Blackwell.
"""

import re
import unittest

import requests

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
)

# OSS FA4 coverage requires Blackwell. Each test still uses only one GPU.
register_cuda_ci(est_time=170, stage="base-b", runner_config="4-gpu-b200")


class TestFullCudaGraphPrefill(CustomTestCase):
    """Integration: Qwen3-8B with --cuda-graph-backend-prefill=full on mgsm_en."""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-8B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--cuda-graph-backend-prefill=full",
                "--attention-backend=flashinfer",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=1319,
            num_threads=1024,
        )

        metrics = run_eval(args)
        score = metrics["score"]
        print(f"mgsm_en accuracy with full prefill CUDA graph: {score:.3f}")

        self.assertGreaterEqual(score, 0.80)


@unittest.skipIf(get_device_sm() < 100, "Test requires CUDA SM 100 or higher")
class TestFullCudaGraphChunkedPrefix(unittest.TestCase):
    """A radix-cache hit replays the OSS FA4 FullCG prefix variant."""

    @classmethod
    def setUpClass(cls):
        cls.model = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={"SGLANG_MAX_KV_CHUNK_CAPACITY": "64"},
            other_args=[
                "--trust-remote-code",
                "--prefill-attention-backend=fa4",
                "--decode-attention-backend=flashinfer",
                "--disable-flashinfer-autotune",
                "--context-length=256",
                "--max-total-tokens=512",
                "--max-running-requests=1",
                "--chunked-prefill-size=128",
                "--skip-server-warmup",
                "--enable-metrics",
                "--cuda-graph-config",
                '{"decode":{"backend":"disabled"},'
                '"prefill":{"backend":"full","bs":[32],"max_bs":32,'
                '"full_prefill_max_req":1}}',
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, input_ids):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 4, "temperature": 0},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def _prefill_graph_count(self):
        metrics = requests.get(self.base_url + "/metrics", timeout=30).text
        match = re.search(
            r'^sglang:cuda_graph_passes_total\{[^}]*mode="prefill_cuda_graph"[^}]*\}'
            r"\s+([0-9.eE+-]+)$",
            metrics,
            re.MULTILINE,
        )
        return float(match.group(1)) if match else 0.0

    def test_cached_prefix_replays_full_cuda_graph(self):
        prefix = list(range(1000, 1048))
        prompt = prefix + list(range(2000, 2032))

        requests.post(self.base_url + "/flush_cache", timeout=30).raise_for_status()
        cold = self._generate(prompt)

        requests.post(self.base_url + "/flush_cache", timeout=30).raise_for_status()
        self._generate(prefix)
        graph_count = self._prefill_graph_count()
        cached = self._generate(prompt)

        self.assertEqual(cached["meta_info"]["cached_tokens"], len(prefix))
        self.assertEqual(cached["output_ids"], cold["output_ids"])
        self.assertEqual(self._prefill_graph_count(), graph_count + 1)


if __name__ == "__main__":
    unittest.main()
