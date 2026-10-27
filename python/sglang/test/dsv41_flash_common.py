"""Shared DeepSeek-V4.1-Flash + DSPARK accuracy recipe for the GB300 tests.

Each server configuration lives in its own file under
``test/registered/e2e/models/test_deepseek_v41_flash_gb300_*.py`` so the
configurations partition independently across GB300 runners. All use default
decoder bounded replay and explicitly enabled breakable prefill graphs.

They run on two GB300 GPUs (TP2) with the engram tables in host memory
(SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE). The two engram tables are ~189 GiB in FP8
(2 x ~384M rows x 264 B), so at TP2 offloading them frees ~94 GiB per GPU and the
remaining weights (~143 GiB/GPU) leave room for KV cache, graphs and DSPARK.
"""

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin, _run_accuracy_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    try_cached_model,
)

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"

# Tensor-parallel attention and expert-parallel MoE, without DP attention.
TP2_EP2_ARGS = ["--attn-dp-size", "1", "--ep-size", "2"]


class DSV41FlashAccuracyMixin(GSM8KMixin):
    # Use the zero-shot chat evaluator, not the legacy five-shot completion path.
    gsm8k_backend = "sgl_eval"
    gsm8k_score_threshold = 0.95
    gsm8k_num_examples = 200
    gsm8k_num_threads = 64
    gsm8k_max_tokens = 4096
    mmmu_pro_score_threshold = 0.75
    mmmu_pro_num_examples = 200
    mmmu_pro_num_threads = 64
    # Bound long reasoning responses consistently across all CI recipes.
    mmmu_pro_max_tokens = 32768
    mmmu_pro_load_preset_from_model_id = MODEL

    mem_fraction_static = "0.8"
    max_running_requests = "32"
    graph_args = [
        "--cuda-graph-backend-prefill",
        "breakable",
        "--cuda-graph-max-bs-prefill",
        "8192",
        "--cuda-graph-max-seq-len-prefill",
        "16384",
        "--chunked-prefill-size",
        "8192",
    ]
    parallel_args = []
    # Keep the engram tables in host memory so DSV4.1-Flash fits on two GB300s.
    server_env = {"SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1"}

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-dspark-block-size",
                "5",
                "--mem-fraction-static",
                cls.mem_fraction_static,
                "--cuda-graph-max-bs-decode",
                "64",
                "--max-running-requests",
                cls.max_running_requests,
                "--reasoning-parser",
                "auto",
                "--tool-call-parser",
                "auto",
                "--disable-flashinfer-autotune",
                *cls.parallel_args,
                *cls.graph_args,
            ],
            env=cls.server_env,
        )
        cls.addClassCleanup(kill_process_tree, cls.process.pid)

    def assert_dspark_accepts_draft_tokens(self):
        response = requests.get(self.base_url + "/server_info", timeout=30)
        response.raise_for_status()
        states = response.json()["internal_states"]
        self.assertTrue(states, "Missing scheduler statistics")
        accept_lengths = [state["avg_spec_accept_length"] for state in states]
        self.assertGreater(
            max(accept_lengths),
            1.0,
            f"DSPARK accepted no draft tokens: {accept_lengths}",
        )

    def test_gsm8k(self):
        super().test_gsm8k()
        self.assert_dspark_accepts_draft_tokens()

    def test_mmmu_pro(self):
        _run_accuracy_eval(
            self,
            eval_name="mmmu_pro",
            score_threshold=self.mmmu_pro_score_threshold,
            num_examples=self.mmmu_pro_num_examples,
            num_threads=self.mmmu_pro_num_threads,
            model=None,
            load_preset_from_model_id=self.mmmu_pro_load_preset_from_model_id,
            max_tokens=self.mmmu_pro_max_tokens,
        )
        self.assert_dspark_accepts_draft_tokens()
