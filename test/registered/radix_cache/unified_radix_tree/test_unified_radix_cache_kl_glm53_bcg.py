"""GLM-5.3 Flash KL regression for KDA prefill graphs and slow HiCache loads.

Like TestUnifiedHybridHiCacheSlowLoadBitExact, exercise branching decode-cache
hits and compare generated-token logprobs against cold-prefill scoring. GLM's
FP4 prefill/decode paths have a numerical floor, so this is a bounded KL guard,
not a bit-exact test. A small device pool makes the same workload reload L2.
"""

import json
import os
import random
import re
import unittest

import numpy as np
import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_multiturn_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _replay_and_compare_kl,
    make_mamba_decode_assert,
)
from sglang.test.kl_test_utils import get_input_ids
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
    unified_radix_tree_server_env,
)

# Full GLM-5.3 weights require the same runner as the existing B200 recipe.
# Provisional wall time including startup; calibrate with the B200 CI timing.
register_cuda_ci(est_time=600, stage="extra-b", runner_config="4-gpu-b200")

MODEL_PATH = "zai-org/GLM-5.3-Flash"
TRACK_INTERVAL = 256
DELAY_CYCLES = 50_000_000
MAX_TOTAL_TOKENS = 8192
MAX_NEW_TOKENS = 512
# The original NVFP4 slow-load experiment measured mean KL 0.007733 with
# dynamic waits versus 0.019290 with completed-placeholder waits. Never use
# Inkling's bit-exact threshold for GLM's different prefill/decode arithmetic.
KL_DIV_THRESHOLD = 0.01


def _random_suffixes(count, length, seed):
    rng = random.Random(seed)
    return [[rng.randint(1, 30000) for _ in range(length)] for _ in range(count)]


class TestUnifiedGLM53HiCacheSlowLoadBCG(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get("GLM53_HICACHE_TEST_MODEL", MODEL_PATH)
        # Attaching is for the remote NVFP4 validation; CI owns its server.
        cls.base_url = os.environ.get("GLM53_HICACHE_TEST_URL")
        if not cls.base_url:
            cls.model = try_cached_model(cls.model)
            cls.base_url = f"http://127.0.0.1:{find_available_port(32180)}"
            hardware = json.loads(
                os.environ.get(
                    "GLM53_HICACHE_TEST_SERVER_ARGS",
                    '["--tp-size", "4", "--dsa-prefill-backend", "trtllm", '
                    '"--dsa-decode-backend", "trtllm", "--kv-cache-dtype", '
                    '"fp8_e4m3", "--moe-runner-backend", "flashinfer_trtllm"]',
                )
            )
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=3600,
                other_args=hardware
                + [
                    "--trust-remote-code",
                    "--linear-attn-backend",
                    "triton",
                    "--linear-attn-prefill-backend",
                    "flashinfer",
                    "--cuda-graph-backend-prefill",
                    "breakable",
                    "--cuda-graph-bs-prefill",
                    "512",
                    "1024",
                    "2048",
                    "--cuda-graph-max-bs-decode",
                    "8",
                    "--chunked-prefill-size",
                    "2048",
                    "--max-total-tokens",
                    str(MAX_TOTAL_TOKENS),
                    "--max-running-requests",
                    "1",
                    "--page-size",
                    "256",
                    "--mamba-radix-cache-strategy",
                    "extra_buffer",
                    "--mamba-track-interval",
                    str(TRACK_INTERVAL),
                    "--enable-hierarchical-cache",
                    "--hicache-ratio",
                    "4",
                    "--hicache-write-policy",
                    "write_through",
                    "--hicache-io-backend",
                    "direct",
                    "--hicache-mem-layout",
                    "page_first_direct",
                    "--enable-cache-report",
                    "--enable-metrics",
                ],
                env=unified_radix_tree_server_env(
                    "python",
                    SGLANG_TEST_HICACHE_H2D_DELAY_CYCLES=str(DELAY_CYCLES),
                    SGLANG_ENABLE_CUDA_GRAPH_DEDUP="0",
                    SGLANG_DISABLE_KDA_PREFILL_GRAPH_EXTEND="0",
                    SGLANG_KDA_PREFILL_GRAPH_MAX_SEQS="1",
                ),
            )
            cls.addClassCleanup(
                terminate_and_kill_process_tree, cls.process, wait_timeout=60
            )
        cls.input_ids = get_input_ids(
            tokenizer_path=cls.model, num_samples=9, trust_remote_code=True
        )

    def _graph_count(self):
        response = requests.get(self.base_url + "/metrics", timeout=30)
        response.raise_for_status()
        return sum(
            map(
                float,
                re.findall(
                    r'^sglang:cuda_graph_passes_total\{[^}]*mode="prefill_cuda_graph"'
                    r"[^}]*\}\s+([0-9.eE+-]+)$",
                    response.text,
                    re.MULTILINE,
                ),
            )
        )

    def test_multiturn_decode_cache_hit_branching(self):
        response = requests.get(self.base_url + "/server_info", timeout=30)
        response.raise_for_status()
        info = response.json()
        self.assertTrue(info["enable_hierarchical_cache"])
        self.assertEqual(info["cuda_graph_config"]["prefill"]["backend"], "breakable")
        self.assertEqual(info["linear_attn_prefill_backend"], "flashinfer")
        self.assertEqual(info["max_total_tokens"], MAX_TOTAL_TOKENS)
        assert_decode_hit = make_mamba_decode_assert(TRACK_INTERVAL)
        groups, branches = 3, 3
        n = groups * branches
        replay_ids, output_logprobs, evidence = [], [], []
        for repeat in range(2):
            _flush_cache(self.base_url)
            histories = [
                list(self.input_ids[g][:512])
                for g in range(groups)
                for _ in range(branches)
            ]
            suffixes = [
                _random_suffixes(n, 512, 300 + repeat),
                _random_suffixes(n, 256, 400 + repeat),
            ]
            previous_lengths = [(0, 0)] * n
            for turn in range(3):
                # Interleave groups so branches compete for the small L1 pool.
                order = [
                    g * branches + b for b in range(branches) for g in range(groups)
                ]
                for i in order:
                    before = self._graph_count()
                    result = _generate(
                        self.base_url,
                        [histories[i]],
                        MAX_NEW_TOKENS,
                        return_logprob=True,
                        temperature=0,
                    )[0]
                    graph_delta = self._graph_count() - before
                    meta = result["meta_info"]
                    if turn:
                        assert_decode_hit(
                            result,
                            *previous_lengths[i],
                            f"repeat{repeat}/turn{turn}/sample{i}",
                        )
                    host_tokens = (meta.get("cached_tokens_details") or {}).get(
                        "host", 0
                    )
                    if host_tokens:
                        self.assertGreater(
                            graph_delta, 0, "L2-hit request bypassed the prefill graph"
                        )
                        evidence.append(
                            {
                                "repeat": repeat,
                                "turn": turn,
                                "sample": i,
                                "host_tokens": host_tokens,
                                "graph_replays": graph_delta,
                            }
                        )
                    lps = _extract_output_logprobs(result)
                    self.assertEqual(len(lps), MAX_NEW_TOKENS)
                    self.assertTrue(np.isfinite(np.asarray(lps, dtype=float)).all())
                    if turn == 2:
                        replay_ids.append(histories[i] + result["output_ids"])
                        output_logprobs.append(lps)
                    else:
                        previous_lengths[i] = (
                            len(histories[i]),
                            len(result["output_ids"]),
                        )
                        histories[i] += result["output_ids"] + suffixes[turn][i]

        # Six or more actual loads exercise reuse beyond three event slots.
        self.assertGreaterEqual(
            len(evidence), 6, "Small pool did not produce enough real L2 graph replays"
        )
        self.assertEqual(len(replay_ids), 18)
        print("host_graph_replays=" + json.dumps(evidence), flush=True)
        _replay_and_compare_kl(
            self.base_url,
            self.model,
            KL_DIV_THRESHOLD,
            replay_ids,
            output_logprobs,
            label="glm53_bcg_hicache_slow_load",
            batch_size=1,
            sampling_temperature=0,
        )


if __name__ == "__main__":
    unittest.main()
