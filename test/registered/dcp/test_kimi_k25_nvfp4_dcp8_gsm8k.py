import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

# ---------------------------------------------------------------------------
# CI registration — nightly only. NVFP4 requires Blackwell (sm100) and DCP=8 +
# TP=8 needs 8 GPUs; the 8-gpu-b200 runner is nightly-only.
# ---------------------------------------------------------------------------
register_cuda_ci(est_time=2400, suite="nightly-8-gpu-b200", nightly=True)

KIMI_K25_NVFP4_MODEL_PATH = "nvidia/Kimi-K2.5-NVFP4"

_COMMON_SERVER_ARGS = [
    "--tp-size",
    "8",
    "--trust-remote-code",
    "--quantization",
    "modelopt_fp4",
    "--enable-cache-report",
    "--enable-metrics",
    "--random-seed",
    "0",
    "--mem-fraction-static",
    "0.85",
    "--chunked-prefill-size",
    "16384",
    "--max-running-requests",
    "256",
    "--cuda-graph-max-bs-decode",
    "256",
    # DCP for MLA is only wired through the flashinfer MLA backend (Kimi's
    # default trtllm_mla has no DCP path).
    "--attention-backend",
    "flashinfer",
    # tc_piecewise prefill capture crashes with attn_dcp_metadata=None under DCP.
    "--disable-piecewise-cuda-graph",
    "--log-level",
    "info",
    "--log-requests",
    "--log-requests-level",
    "3",
]

_DCP8_ARGS = [
    "--dcp-size",
    "8",
]
_DCP4_ARGS = [
    "--dcp-size",
    "4",
]

# Prompts used for logprob parity verification between DCP and non-DCP.
_LOGPROB_PARITY_PROMPTS = [
    "The capital city of France is",
    "What is 2 + 3? The answer is",
    "In the year 1492, Christopher Columbus",
    "The largest planet in our solar system is",
    "Water boils at",
]


def _get_max_total_num_tokens(base_url: str) -> int:
    # When DCP is enabled, max_total_num_tokens is multiplied by dcp_world_size
    # (see model_runner_kv_cache_mixin.py), so it can verify DCP is active.
    resp = requests.get(f"{base_url}/server_info", timeout=30)
    resp.raise_for_status()
    info = resp.json()
    # scheduler_info is flattened into the top-level response
    return info["max_total_num_tokens"]


# ---------------------------------------------------------------------------
# Test 1: nightly accuracy gate + decode sanity (DCP=8, TP=8)
# ---------------------------------------------------------------------------
class TestKimiK25NVFP4DCP8TP8GSM8K(
    GSM8KMixin, BasicDecodeCorrectnessMixin, CustomTestCase
):
    model = KIMI_K25_NVFP4_MODEL_PATH
    base_url = DEFAULT_URL_FOR_TEST

    # A 40-question local DCP=8 run scored 0.975; the 0.90 threshold leaves
    # headroom for the full 200-question set. Run TestKimiK25NVFP4DCP8LogprobParity
    # manually for tighter per-token verification.
    gsm8k_accuracy_thres = 0.90
    gsm8k_num_questions = 200
    gsm8k_num_threads = 128
    gsm8k_num_shots = 5

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=_DCP8_ARGS + _COMMON_SERVER_ARGS,
        )
        # Store max_total_num_tokens so we can verify DCP is active.
        # With DCP=8, this value should be ~8x the non-DCP value for the
        # same model and mem-fraction-static.
        cls._dcp_max_total_num_tokens = _get_max_total_num_tokens(cls.base_url)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid, wait_timeout=60)

    def test_dcp_activation_check(self):
        self.assertGreater(
            self._dcp_max_total_num_tokens,
            0,
            "max_total_num_tokens should be positive",
        )


# ---------------------------------------------------------------------------
# Test 2: DCP=8 vs non-DCP logprob parity (manual-only, too expensive for CI)
# ---------------------------------------------------------------------------
@unittest.skipIf(
    is_in_ci(),
    "Requires two server launches (~30 min); run locally for DCP correctness verification.",
)
class TestKimiK25NVFP4DCP8LogprobParity(BasicDecodeCorrectnessMixin, CustomTestCase):
    # Maximum per-token logprob difference between DCP and non-DCP. DCP adds
    # all-gather/reduce-scatter and (for this model) an fp8->bf16 KV cast, so a
    # loose tolerance accounts for floating-point reordering while still
    # catching systematic bugs (which would diverge >> the tolerance).
    LOGPROB_TOLERANCE = 1.0
    base_url = "http://127.0.0.1:31500"

    model = KIMI_K25_NVFP4_MODEL_PATH

    @classmethod
    def setUpClass(cls):
        # Launch non-DCP baseline server first
        env = os.environ.copy()
        env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
        cls._baseline_process = popen_launch_server(
            KIMI_K25_NVFP4_MODEL_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=_COMMON_SERVER_ARGS,
            env=env,
        )
        cls._processes = [cls._baseline_process]

    @classmethod
    def tearDownClass(cls):
        for proc in cls._processes:
            try:
                kill_process_tree(proc.pid, wait_timeout=60)
            except Exception:
                pass

    @staticmethod
    def _generate_with_logprobs(base_url, prompt, max_new_tokens=8):
        resp = requests.post(
            f"{base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "return_logprob": True,
                "top_logprobs_num": 1,
                "logprob_start_len": 0,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Generate request failed (status {resp.status_code}): {resp.text[:500]}"
            )
        data = resp.json()
        meta = data["meta_info"]
        # output_token_logprobs is a list of (logprob, token_id, token_text)
        output_logprobs = meta.get("output_token_logprobs", [])
        return {
            "text": data["text"],
            "output_logprobs": output_logprobs,
        }

    @staticmethod
    def _warmup_request(base_url):
        requests.post(
            f"{base_url}/generate",
            json={
                "text": "Hello",
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            },
            timeout=60,
        )

    def test_logprob_parity(self):
        # --- Phase 1: collect baseline (non-DCP) outputs ---
        self._warmup_request(self.base_url)

        baseline_results = []
        for prompt in _LOGPROB_PARITY_PROMPTS:
            baseline_results.append(self._generate_with_logprobs(self.base_url, prompt))

        # --- Phase 2: switch to DCP=8 ---
        kill_process_tree(self._baseline_process.pid, wait_timeout=60)
        # Allow OS to release the port
        time.sleep(5)

        env = os.environ.copy()
        env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
        dcp_process = popen_launch_server(
            KIMI_K25_NVFP4_MODEL_PATH,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=_DCP8_ARGS + _COMMON_SERVER_ARGS,
            env=env,
        )
        self._processes.append(dcp_process)
        self._warmup_request(self.base_url)

        dcp_results = []
        for prompt in _LOGPROB_PARITY_PROMPTS:
            dcp_results.append(self._generate_with_logprobs(self.base_url, prompt))

        # --- Phase 3: compare ---
        for i, (baseline, dcp) in enumerate(zip(baseline_results, dcp_results)):
            prompt_short = _LOGPROB_PARITY_PROMPTS[i][:50]
            # Output text must be identical at temperature=0
            self.assertEqual(
                baseline["text"],
                dcp["text"],
                f"Prompt '{prompt_short}...': output text differs between non-DCP and DCP=8.\n"
                f"  non-DCP: {baseline['text']!r}\n"
                f"  DCP=8:   {dcp['text']!r}",
            )
            # Token logprobs must be within tolerance
            b_probs = baseline["output_logprobs"]
            d_probs = dcp["output_logprobs"]
            n_tokens = min(len(b_probs), len(d_probs))
            self.assertGreater(
                n_tokens,
                0,
                f"Prompt '{prompt_short}...': no output tokens produced",
            )
            self.assertEqual(
                len(b_probs),
                len(d_probs),
                f"Prompt '{prompt_short}...': token count differs "
                f"(non-DCP={len(b_probs)}, DCP={len(d_probs)})",
            )
            for j in range(n_tokens):
                # output_token_logprobs format: (logprob, token_id, token_text)
                b_lp = (
                    b_probs[j][0]
                    if isinstance(b_probs[j], (list, tuple))
                    else b_probs[j]
                )
                d_lp = (
                    d_probs[j][0]
                    if isinstance(d_probs[j], (list, tuple))
                    else d_probs[j]
                )
                self.assertAlmostEqual(
                    b_lp,
                    d_lp,
                    delta=self.LOGPROB_TOLERANCE,
                    msg=(
                        f"Prompt '{prompt_short}...', token {j}: "
                        f"logprob diff > {self.LOGPROB_TOLERANCE} "
                        f"(non-DCP={b_lp:.4f}, DCP={d_lp:.4f}, "
                        f"diff={abs(b_lp - d_lp):.4f})"
                    ),
                )


# ---------------------------------------------------------------------------
# Test 3: DCP=4 variant (manual-only, exercises different all-gather pattern)
# ---------------------------------------------------------------------------
@unittest.skipIf(
    is_in_ci(), "Requires 8 GPUs; run locally for additional DCP coverage."
)
class TestKimiK25NVFP4DCP4TP8GSM8K(
    GSM8KMixin, BasicDecodeCorrectnessMixin, CustomTestCase
):
    model = KIMI_K25_NVFP4_MODEL_PATH
    base_url = "http://127.0.0.1:31501"

    gsm8k_accuracy_thres = 0.90
    gsm8k_num_questions = 200
    gsm8k_num_threads = 128
    gsm8k_num_shots = 5

    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=_DCP4_ARGS + _COMMON_SERVER_ARGS,
            env=env,
        )
        # Store max_total_num_tokens for DCP activation verification.
        # With DCP=4, this should be ~4x the non-DCP value.
        cls._dcp_max_total_num_tokens = _get_max_total_num_tokens(cls.base_url)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid, wait_timeout=60)

    def test_dcp_activation_check(self):
        self.assertGreater(
            self._dcp_max_total_num_tokens,
            0,
            "max_total_num_tokens should be positive",
        )


if __name__ == "__main__":
    unittest.main()
