"""
DCP (Decode Context Parallelism) correctness tests.

Dual-platform: the same test IDs run on CUDA and on Intel XPU. The CUDA
configuration is unchanged (DeepSeek-V3.1, flashinfer, DCP=8/TP=8 on 8xH200);
XPU substitutes a platform-appropriate backend, model and shape via ``_PLATFORM``
below, because flashinfer is CUDA-only and V3.1 does not fit. Everything else --
the probe set, the class names, the assertions -- is shared.

Test classes:
    TestDSV31DCP8TP8GSM8K          — CI gate: DCP=8 + TP=8 GSM8K accuracy + decode sanity
    TestDSV31DCP8LogprobParity     — (manual) DCP=8 vs non-DCP logprob equivalence
    TestDSV31DCP4TP8GSM8K          — (manual) DCP=4 + TP=8, different all-gather path

CI coverage & known gaps
------------------------
What the CI test (TestDSV31DCP8TP8GSM8K) covers:
  - DCP decode path: N-way KV-shard all-gather + LSE correction + reduce-scatter
  - DCP extend path: prefix KV all-gather (MLA on CUDA; the staged per-rank
    prefix + current-chunk LSE merge on the XPU Triton path)
  - Basic decode correctness: factual recall, math, no-repetition, temp=0
    determinism, max_new_tokens=1 edge case (catches CUDA graph capture bugs)
  - GSM8K accuracy gate (completion API, 5-shot)
  - DCP activation verification

What the CI test does NOT cover (and the manual tests address):
  - Exact parity with non-DCP outputs (TestDSV31DCP8LogprobParity)
  - A second DCP shape (TestDSV31DCP4TP8GSM8K: DCP=4 at TP=8 on CUDA; skipped on
    XPU, where the KV-head replication rule leaves DCP=2 as the only valid width
    at TP=4 — see _XPU_PLATFORM)
  - MHA extend path on CUDA (all_gather_kv_cache_for_mha_extend /
    mha_chunk_extend) — DeepSeek-V3.1 uses MLA, so those paths are never
    exercised there. The XPU config is GQA and does exercise them.
  - The multi-KV-head branch of the DCP query all-gather reorder. The replication
    rule leaves one KV head per rank for every model whose qkv projection is not
    DCP-aware, so that branch is unreachable from here;
    test/registered/xpu/test_xpu_dcp.py TestDCPGatheredQHeadOrder covers it as a
    kernel-level test.

Future improvements:
  - Add dcp_world_size to /server_info so tests can assert DCP is active
    directly (see _get_max_total_num_tokens: the token budget cannot serve
    this purpose)
  - Tighten gsm8k_accuracy_thres to 0.92+ once baseline numbers are established.
    The non-DCP V3.1 baseline on 200 questions typically scores ~0.93–0.94; the
    current 0.90 threshold provides ~3–4% headroom for initial DCP validation.
    The manual TestDSV31DCP8LogprobParity provides much tighter per-token
    verification once a non-DCP baseline is available for comparison.
"""

import os
import time
import unittest

import requests

from sglang.srt.utils import is_xpu, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

# ---------------------------------------------------------------------------
# CI registration — only TestDSV31DCP8TP8GSM8K runs in CI
# ---------------------------------------------------------------------------
register_cuda_ci(est_time=213, stage="extra-b", runner_config="8-gpu-h200")
register_xpu_ci(est_time=1200, suite="nightly-xpu-4-gpu", nightly=True)

DEEPSEEK_V31_MODEL_PATH = "deepseek-ai/DeepSeek-V3.1"

_IS_XPU = is_xpu()


class _PlatformConfig:
    """Per-platform DCP shape, model and backend.

    Only these knobs differ between CUDA and XPU; the probes and assertions are
    identical. ``dcp_alt`` is the second shape for the DCP4 variant class, or
    None when the platform admits only one valid shape.
    """

    def __init__(
        self,
        *,
        model,
        backend,
        tp_size,
        dcp_size,
        dcp_alt,
        gsm8k_threshold,
        gsm8k_questions,
        gsm8k_threads,
        launch_timeout_mult,
        extra_args,
    ):
        self.model = model
        self.backend = backend
        self.tp_size = tp_size
        self.dcp_size = dcp_size
        self.dcp_alt = dcp_alt
        self.gsm8k_threshold = gsm8k_threshold
        self.gsm8k_questions = gsm8k_questions
        self.gsm8k_threads = gsm8k_threads
        self.launch_timeout_mult = launch_timeout_mult
        self.extra_args = extra_args


_CUDA_PLATFORM = _PlatformConfig(
    model=DEEPSEEK_V31_MODEL_PATH,
    backend="flashinfer",
    tp_size=8,
    dcp_size=8,
    dcp_alt=4,
    gsm8k_threshold=0.90,
    gsm8k_questions=200,
    gsm8k_threads=128,
    launch_timeout_mult=5,
    extra_args=[
        "--enable-cache-report",
        "--enable-metrics",
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.88",
        "--chunked-prefill-size",
        "16384",
        "--max-running-requests",
        "256",
        "--cuda-graph-max-bs-decode",
        "256",
        "--log-requests",
        "--log-requests-level",
        "3",
    ],
)

# XPU deviations, each forced rather than chosen:
#   backend: flashinfer is CUDA-only. The Triton DCP path (LSE merge + sharded
#     KV index build + per-rank masked KV write) is the XPU path; the intel_xpu
#     backend cannot serve DCP at all -- it has no DCP implementation and its
#     decode kernels return no softmax LSE, rejected by ServerArgs.
#   model: DeepSeek-V3.1 is ~700GB. GLM-4.1V-9B-Thinking is the largest dense-
#     attention model that fits 4 B60s *and* satisfies the DCP replication rule,
#     total_num_kv_heads <= tp_size / dcp_size: DCP shards KV by position, so each
#     rank of a group must hold the same KV heads, and only Qwen3.5 shards its qkv
#     projection that way (ServerArgs rejects everything else that violates it).
#     Its 2 KV heads at tp // dcp_size = 2 clear the bar. It is GQA rather than MLA,
#     so the XPU run covers cp_lse_ag_out_rs_mha and the MHA extend path instead of
#     the MLA merge.
#   shape: tp must stay a multiple of dcp_size, and the replication rule caps
#     dcp_size at tp_size // 2 = 2 here, so dcp=2 is the only DCP width available --
#     dcp_alt=None and the second-shape class skips. Varying it needs a
#     DCP-aware-qkv model (Qwen3.5), which needs GDN kernels sgl-kernel-xpu does not
#     ship yet.
#   gsm8k threshold: 0.80, against a measured 0.880 at DCP=2 and 0.880 without DCP
#     (200 questions, 5 shots, invalid 0.000 in both) -- so DCP costs no accuracy
#     here and the gate has ~8 points of headroom for run-to-run variation. It is
#     deliberately not tighter: the failure this guards against is a broken LSE
#     merge or a KV write landing at the wrong stride, which collapses accuracy
#     toward zero rather than shaving a point off it.
#   mem-fraction-static: 0.60, well under the 0.88 the CUDA config uses. The
#     per-layer xccl all-gather / reduce-scatter want Level Zero scratch from
#     outside the static pool; 0.85 already dies with
#     UR_RESULT_ERROR_OUT_OF_RESOURCES from the TP all-reduce at dcp=2.
_XPU_PLATFORM = _PlatformConfig(
    model=DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP,
    backend="triton",
    tp_size=4,
    dcp_size=2,
    dcp_alt=None,
    gsm8k_threshold=0.80,
    gsm8k_questions=100 if is_in_ci() else 200,
    gsm8k_threads=32,
    launch_timeout_mult=3,
    extra_args=[
        "--device",
        "xpu",
        "--mem-fraction-static",
        "0.60",
        "--chunked-prefill-size",
        "2048",
        "--disable-radix-cache",
    ],
)

_PLATFORM = _XPU_PLATFORM if _IS_XPU else _CUDA_PLATFORM

_COMMON_SERVER_ARGS = [
    "--tp-size",
    str(_PLATFORM.tp_size),
    "--random-seed",
    "0",
    "--attention-backend",
    _PLATFORM.backend,
    "--cuda-graph-backend-prefill=disabled",
    "--log-level",
    "info",
    *_PLATFORM.extra_args,
]

_DCP8_ARGS = ["--dcp-size", str(_PLATFORM.dcp_size)]
_DCP4_ARGS = ["--dcp-size", str(_PLATFORM.dcp_alt or _PLATFORM.dcp_size)]

# Prompts used for logprob parity verification between DCP and non-DCP.
_LOGPROB_PARITY_PROMPTS = [
    "The capital city of France is",
    "What is 2 + 3? The answer is",
    "In the year 1492, Christopher Columbus",
    "The largest planet in our solar system is",
    "Water boils at",
]


def _get_max_total_num_tokens(base_url: str) -> int:
    """Fetch max_total_num_tokens from /server_info.

    NOTE: this is NOT a DCP activation signal, in either direction. The scheduler
    reports ``max_total_num_tokens * dcp_size``, so what it means depends on how the
    per-rank pool responded to dcp_size, which depends on the model's KV head count:
    on XPU the per-rank pool is unchanged between DCP=1 and DCP=2 (measured: 487859
    both for GLM-4.1V-9B, 1015888 both for Qwen2.5-1.5B at TP=4), so the reported
    number doubles; for a model whose KV heads get replicated instead it stays flat.
    Neither tells you the DCP code path ran. Use it for liveness only.
    """
    resp = requests.get(f"{base_url}/server_info", timeout=30)
    resp.raise_for_status()
    # scheduler_info is flattened into the top-level response
    return resp.json()["max_total_num_tokens"]


# ---------------------------------------------------------------------------
# Test 1: CI accuracy gate + decode sanity (DCP=8, TP=8 on CUDA; DCP=2/TP=4 XPU)
# ---------------------------------------------------------------------------
class TestDSV31DCP8TP8GSM8K(GSM8KMixin, BasicDecodeCorrectnessMixin, CustomTestCase):
    """CI accuracy gate + basic decode probes under DCP.

    This test exercises the full DCP decode and extend paths:
      - Decode: query all-gather → attention on local KV shard → LSE
        correction (cp_lse_ag_out_rs_mla for MLA / cp_lse_ag_out_rs_mha for
        MHA-GQA) → reduce-scatter or all-reduce
      - Extend (prefill): the sharded prefix KV is combined across DCP ranks and
        merged with the current chunk by LSE

    Inherits:
      - GSM8KMixin.test_gsm8k: accuracy gate
      - BasicDecodeCorrectnessMixin: cheap sanity probes (factual recall,
        no-repetition, temp=0 determinism, max_new_tokens=1)

    Platform: CUDA runs DeepSeek-V3.1 (MLA) at DCP=8/TP=8 with flashinfer; XPU
    runs GLM-4.1V-9B-Thinking (GQA) at DCP=2/TP=4 with Triton. See _PLATFORM.
    """

    model = _PLATFORM.model
    base_url = DEFAULT_URL_FOR_TEST

    # CUDA: non-DCP V3.1 on 200 questions typically scores ~0.93–0.94, so 0.90
    # leaves ~3–4% headroom. XPU: 0.80 against a measured 0.880 both with and
    # without DCP; see _PLATFORM.
    # For tighter verification, run TestDSV31DCP8LogprobParity manually.
    gsm8k_accuracy_thres = _PLATFORM.gsm8k_threshold
    gsm8k_num_questions = _PLATFORM.gsm8k_questions
    gsm8k_num_threads = _PLATFORM.gsm8k_threads
    gsm8k_num_shots = 5

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * _PLATFORM.launch_timeout_mult,
            other_args=_DCP8_ARGS + _COMMON_SERVER_ARGS,
        )
        cls._dcp_max_total_num_tokens = _get_max_total_num_tokens(cls.base_url)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid, wait_timeout=60)

    def test_dcp_activation_check(self):
        """Verify the server is live and reports a sane token budget.

        A stronger check that compares DCP vs non-DCP outputs is in
        TestDSV31DCP8LogprobParity.test_logprob_parity.
        """
        # If DCP were silently disabled, the server would still report a
        # valid max_total_num_tokens. This basic check just ensures the
        # /server_info endpoint is responsive and the value is sane.
        self.assertGreater(
            self._dcp_max_total_num_tokens,
            0,
            "max_total_num_tokens should be positive",
        )


# ---------------------------------------------------------------------------
# Test 2: DCP vs non-DCP logprob parity (manual-only, too expensive for CI)
# ---------------------------------------------------------------------------
@unittest.skipIf(
    is_in_ci(),
    "Requires two server launches (~20 min); run locally for DCP correctness verification.",
)
class TestDSV31DCP8LogprobParity(BasicDecodeCorrectnessMixin, CustomTestCase):
    """Verify DCP produces output-equivalent results to non-DCP at the same TP.

    Strategy:
      1. Launch a non-DCP baseline server.
      2. Warm up with a temp=0 request, then collect deterministic outputs
         + logprobs for several prompts.
      3. Kill the baseline, launch a DCP server on the same port.
      4. Warm up and collect outputs + logprobs for the same prompts.
      5. Assert:
         - Output text matches exactly (temperature=0 must be deterministic)
         - Token logprobs are within tolerance (floating-point all-gather
           introduces small numerical differences)

    This catches subtle correctness bugs in the DCP LSE correction path that a
    coarse GSM8K accuracy gate cannot detect. For example, if exp2/exp mismatch
    causes a systematic bias in the attention output, logprobs will diverge by
    more than the tolerance. Two such bugs were found this way while enabling
    XPU DCP: a rank-major query all-gather head order (which paired query heads
    with the wrong KV head) and a violated KV-replication invariant -- both
    produced fluent-looking but wrong continuations.
    """

    # Maximum per-token logprob difference between DCP and non-DCP.
    # DCP introduces additional all-gather/reduce-scatter operations;
    # a tolerance of 1.0 accounts for floating-point reordering while
    # still catching systematic bugs (which diverge by many nats).
    LOGPROB_TOLERANCE = 1.0
    base_url = "http://127.0.0.1:31500"

    model = _PLATFORM.model

    @classmethod
    def setUpClass(cls):
        # Launch non-DCP baseline server first
        env = os.environ.copy()
        env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
        cls._baseline_process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * _PLATFORM.launch_timeout_mult,
            other_args=list(_COMMON_SERVER_ARGS),
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
        """Send a temp=0 generation request and return output text + logprobs."""
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
        """Send a warmup request to trigger CUDA graph capture and JIT compilation."""
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
        baseline_max_total = _get_max_total_num_tokens(self.base_url)

        baseline_results = []
        for prompt in _LOGPROB_PARITY_PROMPTS:
            baseline_results.append(self._generate_with_logprobs(self.base_url, prompt))

        # --- Phase 2: switch to DCP ---
        kill_process_tree(self._baseline_process.pid, wait_timeout=60)
        # Allow OS to release the port
        time.sleep(5)

        env = os.environ.copy()
        env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
        dcp_process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * _PLATFORM.launch_timeout_mult,
            other_args=_DCP8_ARGS + _COMMON_SERVER_ARGS,
            env=env,
        )
        self._processes.append(dcp_process)
        self._warmup_request(self.base_url)

        # Liveness only: the reported budget does not move with dcp_size
        # (see _get_max_total_num_tokens). The comparison below is the real
        # evidence that the cross-rank merge ran.
        self.assertGreater(baseline_max_total, 0)
        self.assertGreater(_get_max_total_num_tokens(self.base_url), 0)

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
                f"Prompt '{prompt_short}...': output text differs between non-DCP and DCP.\n"
                f"  non-DCP: {baseline['text']!r}\n"
                f"  DCP:     {dcp['text']!r}",
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
# Test 3: second DCP shape (manual-only, exercises a different all-gather)
# ---------------------------------------------------------------------------
@unittest.skipIf(
    _PLATFORM.dcp_alt is None,
    "Platform admits only one DCP width (XPU: the KV-head replication rule caps "
    "dcp_size at tp_size // total_num_kv_heads = 2), so this would re-run the CI "
    "shape.",
)
@unittest.skipIf(
    is_in_ci(),
    "Second DCP shape; run locally rather than paying for another full model load "
    "in CI.",
)
class TestDSV31DCP4TP8GSM8K(GSM8KMixin, BasicDecodeCorrectnessMixin, CustomTestCase):
    """DCP=4 — exercises a different all-gather width than the CI shape.

    TP=8 on CUDA: 1/4 of the KV cache per rank vs 1/8 at DCP=8. The wider
    all-gather uses a different GroupCoordinator configuration, and the
    token-to-shard mapping exercises different edge cases in:
      - update_local_kv_lens_for_dcp (different div/mod arithmetic)
      - plan_dcp_decode_metadata (different local_kv_lens distribution)
      - create_dcp_kv_indices (different padding/alignment)
      - all_gather_kv_cache_for_dcp (4-way vs 8-way interleave pattern)
    """

    model = _PLATFORM.model
    base_url = "http://127.0.0.1:31501"

    gsm8k_accuracy_thres = _PLATFORM.gsm8k_threshold
    gsm8k_num_questions = _PLATFORM.gsm8k_questions
    gsm8k_num_threads = _PLATFORM.gsm8k_threads
    gsm8k_num_shots = 5

    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * _PLATFORM.launch_timeout_mult,
            other_args=_DCP4_ARGS + _COMMON_SERVER_ARGS,
            env=env,
        )
        cls._dcp_max_total_num_tokens = _get_max_total_num_tokens(cls.base_url)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid, wait_timeout=60)

    def test_dcp_activation_check(self):
        """Verify the server is live and reports a sane token budget."""
        self.assertGreater(
            self._dcp_max_total_num_tokens,
            0,
            "max_total_num_tokens should be positive",
        )


if __name__ == "__main__":
    unittest.main()
