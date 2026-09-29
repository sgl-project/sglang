"""Radix-cache logprob consistency for native Beam under attention DP.

DP2 x TP2 on the deterministic fixture: each attention-DP rank owns its own
hybrid SWA pool and radix tree, and every MLP gathers both ranks' tokens over
TP. A restored prefix and a warm-cache decode must reproduce a fresh prefill,
and a cached-prefix prefill must still replay a CUDA graph while the other
rank runs an idle batch.
"""

import os
import re
import tempfile
import time
import unittest

import torch

from sglang.srt.utils import is_flashinfer_available, kill_process_tree
from sglang.srt.utils.common import (
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
)
from sglang.test.beam_fixture import make_beam_fixture
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kl_test_utils import _flush_cache, _generate
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")

CACHED_PREFIX_GRAPH_REPLAY_PATTERN = re.compile(
    r"Prefill batch.*#cached-token: [1-9][0-9]*.*cuda graph: True"
)


class TestBeamDPAttentionKL(KLDivergenceMixin, CustomTestCase):
    kl_div_thres = 0.005
    base_url = DEFAULT_URL_FOR_TEST
    # DP attention divides the chunk size by dp, so each rank prefills in
    # 256-token chunks as in test_beam_kl.py.
    other_args = [
        "--tp-size",
        "2",
        "--dp-size",
        "2",
        "--enable-dp-attention",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "trtllm_mha",
        "--page-size",
        "64",
        "--chunked-prefill-size",
        "512",
        "--mem-fraction-static",
        "0.3",
    ]

    @classmethod
    def setUpClass(cls):
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("Beam DP attention test requires two CUDA GPUs")
        if not is_flashinfer_available() or not (
            is_sm90_supported() or is_sm100_supported() or is_sm120_supported()
        ):
            raise unittest.SkipTest(
                "Beam TRT-LLM MHA test requires FlashInfer and SM90, SM100, or SM120"
            )
        checkpoint = tempfile.TemporaryDirectory(prefix="beam-fixture-")
        cls.addClassCleanup(checkpoint.cleanup)
        make_beam_fixture(checkpoint.name)
        cls.model = checkpoint.name
        log_dir = tempfile.TemporaryDirectory(prefix="beam-dp-attention-")
        cls.addClassCleanup(log_dir.cleanup)
        cls.stdout_path = os.path.join(log_dir.name, "server.out")
        cls.stderr_path = os.path.join(log_dir.name, "server.err")
        stdout = open(cls.stdout_path, "w")
        cls.addClassCleanup(stdout.close)
        stderr = open(cls.stderr_path, "w")
        cls.addClassCleanup(stderr.close)
        process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            return_stdout_stderr=(stdout, stderr),
        )
        cls.addClassCleanup(kill_process_tree, process.pid)

    def _logs_since(self, offsets):
        chunks = []
        for path, offset in zip((self.stdout_path, self.stderr_path), offsets):
            with open(path, "rb") as log:
                log.seek(offset)
                chunks.append(log.read().decode(errors="replace"))
        return "\n".join(chunks)

    def test_cached_prefix_prefill_replays_cuda_graph(self):
        _flush_cache(self.base_url)
        prompt = [1 + (i * 17) % 127 for i in range(300)]
        # Both requests land on rank 0, so the second reuses the first one's
        # prefix while rank 1 runs an idle batch.
        _generate(self.base_url, prompt, max_new_tokens=0, routed_dp_rank=0)
        offsets = [os.path.getsize(p) for p in (self.stdout_path, self.stderr_path)]
        result = _generate(
            self.base_url, prompt + [5, 7, 9], max_new_tokens=1, routed_dp_rank=0
        )
        self.assertNotIn("error", result)
        deadline = time.monotonic() + 30
        while not CACHED_PREFIX_GRAPH_REPLAY_PATTERN.search(self._logs_since(offsets)):
            if time.monotonic() > deadline:
                self.fail("No cached-prefix prefill CUDA graph replay was logged")
            time.sleep(0.5)


if __name__ == "__main__":
    unittest.main()
