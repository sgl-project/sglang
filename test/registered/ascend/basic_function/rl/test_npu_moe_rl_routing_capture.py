"""
E2E Test Suite: NPU MoE Routing Expert Capture
===============================================
Validates the integrity of the MoE routing expert capture data flow on NPU —
SGLang records the expert IDs selected per token during inference and returns
them to the caller via the HTTP API.

Models: Qwen/Qwen3-30B-A3B (BF16, Unquantized MoE)
        Qwen/Qwen3-30B-A3B-w8a8 (W8A8, INT8 quantized)
Hardware: NPU

## Test Matrix

| Class                          | Model              | TP | Quant | Cases | GPU Counterpart               |
|--------------------------------|--------------------|----|-------|-------|-------------------------------|
| TestNPURoutingCaptureBasic     | Qwen3-30B-A3B      | 1  | BF16  | 6     | TestReturnRoutedExperts       |
| TestNPURoutingCaptureStartLen  | Qwen3-30B-A3B      | 2  | BF16  | 4     | TestRoutedExpertsStartLen     |
| TestNPURoutingCaptureW8A8      | Qwen3-30B-A3B-W8A8 | 1  | W8A8  | 3     | (NPU-only)                    |
| TestNPURoutingCaptureTP2       | Qwen3-30B-A3B      | 2  | BF16  | 3     | (NPU-only)                    |

"""

import logging
import os
import unittest

import numpy as np
import requests

from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH as QWEN3_30B_A3B_INSTRUCT,
)
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_W8A8_WEIGHTS_PATH as QWEN3_30B_A3B_W8A8,
)
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_WEIGHTS_PATH as QWEN3_30B_A3B,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=1800, suite="full-2-npu-a3", nightly=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Model config constants (Qwen3-30B-A3B)
# ──────────────────────────────────────────────────────────────────────
_NUM_LAYERS = 48
_TOPK = 8
_NUM_EXPERTS = 128


# ──────────────────────────────────────────────────────────────────────
# Verification helpers
# ──────────────────────────────────────────────────────────────────────
def _get_decode_logprob_signature(
    base_url,
    *,
    prompt=None,
    max_new_tokens=16,
    temperature=0.0,
    return_rr=True,
    start_len=0,
):
    """
    Get the full probabilistic signature of a single deterministic decode.

    Returns:
        dict: {
            "text": generated text,
            "token_ids": [int, ...],
            "logprobs": [float, ...],
            "routed_experts": np.ndarray | None,  # flat int32 array; reshape to [seqlen-1-start_len, 48, 8]
            "meta_info": dict,
        }
    """
    payload = {
        "text": prompt or "The capital of France is",
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": True,
        "return_routed_experts": return_rr,
    }
    if start_len != 0 or return_rr:
        payload["routed_experts_start_len"] = start_len

    resp = requests.post(
        f"{base_url}/generate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    ret = resp.json()

    output_token_logprobs = ret["meta_info"].get("output_token_logprobs")
    assert (
        output_token_logprobs is not None
    ), "missing output_token_logprobs in response"
    assert len(output_token_logprobs) > 0, "empty output_token_logprobs"

    result = {
        "text": ret["text"],
        "token_ids": [int(x[1]) for x in output_token_logprobs],
        "logprobs": [float(x[0]) for x in output_token_logprobs],
        "meta_info": ret["meta_info"],
    }

    if return_rr:
        result["routed_experts"] = extract_routed_experts_from_meta_info(ret)
    else:
        result["routed_experts"] = None

    return result


def _assert_logprob_signature_equal(a, b, *, atol=1e-4, msg=""):
    """Assert two decode logprob signatures are identical."""
    assert a["text"] == b["text"], f"{msg}text mismatch: {a['text']!r} != {b['text']!r}"
    assert (
        a["token_ids"] == b["token_ids"]
    ), f"{msg}token_ids mismatch: {a['token_ids']} != {b['token_ids']}"
    assert len(a["logprobs"]) == len(
        b["logprobs"]
    ), f"{msg}logprobs length mismatch: {len(a['logprobs'])} != {len(b['logprobs'])}"
    for idx, (la, lb) in enumerate(zip(a["logprobs"], b["logprobs"])):
        assert abs(la - lb) <= atol, (
            f"{msg}logprob diff at idx={idx}: {la} vs {lb} "
            f"(delta={abs(la - lb):.6f}, tol={atol})"
        )
    logger.info(
        f"{msg}logprobs match: text={a['text'][:40]!r}... tokens={len(a['token_ids'])} (PASS)"
    )


def _assert_experts_valid(routed, num_experts=_NUM_EXPERTS):
    """Validate expert IDs are in the legal range."""
    assert isinstance(routed, np.ndarray), f"Expected np.ndarray, got {type(routed)}"
    assert routed.dtype == np.int32, f"Expected int32, got {routed.dtype}"
    assert (routed >= 0).all() and (routed < num_experts).all(), (
        f"Expert IDs out of [0, {num_experts}): "
        f"min={routed.min()}, max={routed.max()}"
    )
    logger.info(
        f"[_assert_experts_valid] shape={routed.shape} dtype={routed.dtype} "
        f"range=[{routed.min()}, {routed.max()}] num_experts={num_experts} (PASS)"
    )


def _assert_experts_shape(
    routed, seqlen, start_len=0, num_layers=_NUM_LAYERS, topk=_TOPK
):
    """Validate the captured tensor shape is correct."""
    expected_rows = seqlen - 1 - start_len
    reshaped = routed.reshape(expected_rows, num_layers, topk)
    assert reshaped.shape == (
        expected_rows,
        num_layers,
        topk,
    ), f"Shape mismatch: {reshaped.shape} != ({expected_rows}, {num_layers}, {topk})"
    logger.info(
        f"[_assert_experts_shape] shape={reshaped.shape} seqlen={seqlen} start_len={start_len} (PASS)"
    )
    return reshaped


def _assert_experts_equal(a, b, *, msg=""):
    """Bit-exact comparison of two expert capture arrays."""
    mismatch = (a != b).sum()
    total = a.size
    if mismatch == 0:
        logger.info(f"{msg}Expert bit-exact match: 0 / {total} mismatches (PASS)")
    else:
        logger.info(f"{msg}Expert MISMATCH: {mismatch} / {total}")
    assert mismatch == 0, f"{msg}Expert mismatch count: {mismatch} / {total}"


def _expert_overlap_rate(a, b):
    """Compute expert assignment overlap rate between two captures."""
    return (a == b).mean()


# ──────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────
class _BaseNPURoutingCaptureTest(CustomTestCase):
    """Base class for NPU MoE routing capture tests.

    Subclasses must set:
        - model: model path
        - tp_size: tensor parallelism size
        - server_extra_args: additional server launch arguments
    """

    model = None
    tp_size = 1
    base_url = DEFAULT_URL_FOR_TEST
    server_process = None
    server_extra_args = []
    _disable_cuda_graph = True
    _mem_fraction_static = "0.95"

    @classmethod
    def setUpClass(cls):
        if cls.model is None:
            raise NotImplementedError("Subclass must set 'model'")

    def _launch_server(self, extra_args=None):
        args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            self._mem_fraction_static,
            "--max-running-requests",
            "8",
            "--tp-size",
            str(self.tp_size),
            "--enable-return-routed-experts",
        ]
        if self._disable_cuda_graph:
            args.append("--disable-cuda-graph")
        args.extend(self.server_extra_args)
        if extra_args:
            args.extend(extra_args)
        logger.info(
            f"[_launch_server] Starting server model={self.model} tp={self.tp_size}..."
        )
        self.server_process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env={**os.environ, "SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT": "1"},
        )

    def tearDown(self):
        if self.server_process is not None:
            logger.info(
                f"[tearDown] Killing server process pid={self.server_process.pid}..."
            )
            kill_process_tree(self.server_process.pid)
            self.server_process = None
        # torch.multiprocessing forkserver workers may escape process tree;
        # kill them explicitly to avoid NPU resource leaks
        try:
            import subprocess as sp

            result = sp.run(
                ["pkill", "-f", "multiprocessing.forkserver"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("[tearDown] Cleaned up forkserver workers.")
        except Exception as e:
            logger.info(f"[tearDown] forkserver cleanup skipped: {e}")

    def _run_decode(self, **kwargs):
        return _get_decode_logprob_signature(self.base_url, **kwargs)

    # ── HTTP helpers ────────────────────────────────────────────────
    def _post(self, endpoint, payload, timeout=300):
        resp = requests.post(
            f"{self.base_url}{endpoint}", json=payload, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()

    def _update_weights_from_disk(self, model_path, flush_cache=True):
        logger.info(
            f"[_update_weights_from_disk] Updating to {model_path} (flush_cache={flush_cache})..."
        )
        return self._post(
            "/update_weights_from_disk",
            {
                "model_path": model_path,
                "flush_cache": flush_cache,
            },
        )


# ══════════════════════════════════════════════════════════════════════
# Test Class 1: Capture Basic — BF16, TP=1
# ══════════════════════════════════════════════════════════════════════
class TestNPURoutingCaptureBasic(_BaseNPURoutingCaptureTest):
    """
    Verify basic functionality of --enable-return-routed-experts on the NPU BF16 path.

    [Test Category] RL Routing Capture
    [Test Target] POST /v1/completions (return_routed_experts)

    """

    model = QWEN3_30B_A3B
    tp_size = 1

    def setUp(self):
        self._launch_server()

    # ── T-CAP-01 ────────────────────────────────────────────────────
    def test_01_generate_endpoint_capture(self):
        """T-CAP-01: /generate endpoint capture basic validation.

        Verifies:
          1. meta_info.routed_experts is non-empty
          2. base64 decode succeeds
          3. returns int32 ndarray
          4. shape is correct: [seqlen-1, 48, 8]
        """
        result = self._run_decode(max_new_tokens=16)
        routed = result["routed_experts"]
        assert (
            routed is not None
        ), "routed_experts must not be None when return_routed_experts=True"

        _assert_experts_valid(routed)
        seqlen = (
            result["meta_info"]["prompt_tokens"]
            + result["meta_info"]["completion_tokens"]
        )
        _assert_experts_shape(routed, seqlen)

    # ── T-CAP-02 ────────────────────────────────────────────────────
    def test_02_expert_ids_valid_range(self):
        """T-CAP-02: Verify all expert IDs are in [0, 128) range."""
        result = self._run_decode(max_new_tokens=32)
        routed = result["routed_experts"]
        _assert_experts_valid(routed, num_experts=_NUM_EXPERTS)

    # ── T-CAP-03 ────────────────────────────────────────────────────
    def test_03_deterministic_routing_same_prompt(self):
        """T-CAP-03: Deterministic routing — same prompt produces same expert assignments.

        Verifies:
          1. Expert assignments are bit-exact across two identical requests
          2. Logprobs are per-token identical
        """
        sig1 = self._run_decode()
        sig2 = self._run_decode()

        _assert_logprob_signature_equal(sig1, sig2, msg="T-CAP-03: ")
        routed1 = sig1["routed_experts"]
        routed2 = sig2["routed_experts"]
        logger.info(f"[T-CAP-03] routed1 shape={routed1.shape} dtype={routed1.dtype}")
        logger.info(f"[T-CAP-03] routed1=\n{routed1}")
        logger.info(f"[T-CAP-03] routed2 shape={routed2.shape} dtype={routed2.dtype}")
        logger.info(f"[T-CAP-03] routed2=\n{routed2}")
        _assert_experts_equal(routed1, routed2, msg="T-CAP-03: ")

    # ── T-CAP-04 ────────────────────────────────────────────────────
    def test_04_no_routed_experts_without_flag(self):
        """T-CAP-04: No routed_experts data returned when flag is not set.

        Verifies:
          1. meta_info lacks routed_experts when return_routed_experts=False
          2. meta_info contains routed_experts when return_routed_experts=True
        """
        # Request WITHOUT flag
        sig_without = self._run_decode(return_rr=False)
        assert (
            "routed_experts" not in sig_without["meta_info"]
        ), "routed_experts should NOT appear when return_routed_experts=False"

        # Request WITH flag
        sig_with = self._run_decode(return_rr=True)
        assert (
            "routed_experts" in sig_with["meta_info"]
        ), "routed_experts SHOULD appear when return_routed_experts=True"

    # ── T-CAP-05 ────────────────────────────────────────────────────
    def test_05_routing_changes_with_weights(self):
        """T-CAP-05: Expert assignments change after router weight update.

        Flow:
          1. baseline = decode(base)
          2. update(instruct) → decode → expert distribution differs significantly
          3. update(base) → decode → expert distribution matches baseline exactly

        Verifies:
          1. Expert overlap < 90% with instruct model
          2. Expert assignments bit-exact restored after switching back to base
        """
        prompt = "The capital of France is"
        logger.info("[T-CAP-05] Phase 1/3: baseline decode (base model)...")
        baseline = self._run_decode(prompt=prompt, max_new_tokens=32)
        baseline_experts = baseline["routed_experts"]

        # Switch to Instruct
        logger.info("[T-CAP-05] Phase 2/3: switching to instruct model...")
        r1 = self._update_weights_from_disk(QWEN3_30B_A3B_INSTRUCT)
        assert r1.get("success"), f"Update to instruct failed: {r1}"

        instruct_sig = self._run_decode(prompt=prompt, max_new_tokens=32)
        instruct_experts = instruct_sig["routed_experts"]

        # Expert assignments must differ significantly
        overlap = _expert_overlap_rate(baseline_experts, instruct_experts)
        logger.info(f"[T-CAP-05] Expert overlap (base vs instruct): {overlap:.2%}")
        assert overlap < 0.90, (
            f"T-CAP-05: Expert overlap too high ({overlap:.2%}) — "
            f"routing should change when weights change"
        )

        # Switch back to Base
        logger.info("[T-CAP-05] Phase 3/3: switching back to base model...")
        r2 = self._update_weights_from_disk(QWEN3_30B_A3B)
        assert r2.get("success"), f"Update back to base failed: {r2}"

        restored_sig = self._run_decode(prompt=prompt, max_new_tokens=32)
        restored_experts = restored_sig["routed_experts"]

        # Expert assignments must be fully restored
        _assert_experts_equal(
            baseline_experts,
            restored_experts,
            msg="T-CAP-05: routing should be bit-exact after roundtrip",
        )
        _assert_logprob_signature_equal(
            baseline,
            restored_sig,
            msg="T-CAP-05: logprobs should match after roundtrip",
        )

    # ── T-CAP-06 ────────────────────────────────────────────────────
    def test_06_multiple_decode_consistent(self):
        """T-CAP-06: Multiple consecutive decodes remain consistent.

        3 consecutive decodes, verifying:
          1. Logprobs are consistent
          2. Expert assignments are consistent
        """
        sig1 = self._run_decode()
        sig2 = self._run_decode()
        sig3 = self._run_decode()

        _assert_logprob_signature_equal(sig1, sig2, msg="T-CAP-06: sig1 vs sig2 ")
        _assert_logprob_signature_equal(sig2, sig3, msg="T-CAP-06: sig2 vs sig3 ")
        _assert_experts_equal(
            sig1["routed_experts"],
            sig2["routed_experts"],
            msg="T-CAP-06: experts sig1 vs sig2 ",
        )
        _assert_experts_equal(
            sig2["routed_experts"],
            sig3["routed_experts"],
            msg="T-CAP-06: experts sig2 vs sig3 ",
        )


# ══════════════════════════════════════════════════════════════════════
# Test Class 2: Capture StartLen — BF16, TP=2
# ══════════════════════════════════════════════════════════════════════
class TestNPURoutingCaptureStartLen(_BaseNPURoutingCaptureTest):
    """
    Verify routed_experts_start_len parameter behavior.

    [Test Category] RL Routing Capture + StartLen
    [Test Target] POST /v1/completions (routed_experts_start_len)

    """

    model = QWEN3_30B_A3B
    tp_size = 2
    _disable_cuda_graph = False
    _mem_fraction_static = "0.50"
    MAX_NEW_TOKENS = 8

    def setUp(self):
        self._launch_server()

    def _build_payload(self, **extra):
        payload = {
            "text": "User: Tell me a fact about cats.\nAssistant:",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": self.MAX_NEW_TOKENS,
                "ignore_eos": True,
            },
            "return_routed_experts": True,
        }
        payload.update(extra)
        return payload

    def _send(self, payload):
        return requests.post(f"{self.base_url}/generate", json=payload, timeout=120)

    def _routed_experts(self, resp_json):
        return extract_routed_experts_from_meta_info(resp_json).reshape(
            -1, _NUM_LAYERS, _TOPK
        )

    def _seqlen(self, resp_json):
        meta = resp_json["meta_info"]
        return meta["prompt_tokens"] + meta["completion_tokens"]

    # ── T-SL-01 ────────────────────────────────────────────────────
    def test_01_start_len_zero_is_default(self):
        """T-SL-01: Default is equivalent to routed_experts_start_len=0.

        Verifies:
          1. Omitting the field and explicit start_len=0 return the same row count
          2. Both responses have identical expert assignments
        """
        resp_default = self._send(self._build_payload()).json()
        resp_zero = self._send(self._build_payload(routed_experts_start_len=0)).json()

        rows_default = self._routed_experts(resp_default)
        rows_zero = self._routed_experts(resp_zero)
        seqlen_default = self._seqlen(resp_default)
        seqlen_zero = self._seqlen(resp_zero)

        assert seqlen_default == seqlen_zero
        assert rows_default.shape[0] == seqlen_default - 1
        assert rows_zero.shape[0] == seqlen_zero - 1
        assert np.array_equal(
            rows_default, rows_zero
        ), "default and explicit 0 must produce identical routed experts"

    # ── T-SL-02 ────────────────────────────────────────────────────
    def test_02_start_len_controls_row_count(self):
        """T-SL-02: start_len=N returns seqlen-1-N rows, tail matches full sequence."""
        full_resp = self._send(self._build_payload()).json()
        full_rows = self._routed_experts(full_resp)
        seqlen = self._seqlen(full_resp)
        assert full_rows.shape[0] == seqlen - 1

        start_len = max(1, full_resp["meta_info"]["prompt_tokens"] // 2)

        cropped_resp = self._send(
            self._build_payload(routed_experts_start_len=start_len)
        ).json()
        cropped_rows = self._routed_experts(cropped_resp)
        cropped_seqlen = self._seqlen(cropped_resp)

        assert seqlen == cropped_seqlen
        expected_rows = seqlen - 1 - start_len
        assert (
            cropped_rows.shape[0] == expected_rows
        ), f"expected {expected_rows} rows, got {cropped_rows.shape[0]}"
        assert np.array_equal(
            full_rows[start_len:], cropped_rows
        ), "cropped routed experts must match the tail of the full sequence"

    # ── T-SL-03 ────────────────────────────────────────────────────
    def test_03_start_len_exceeds_prompt_tokens_aborts(self):
        """T-SL-03: start_len > prompt_tokens → abort."""
        baseline = self._send(self._build_payload()).json()
        prompt_tokens = baseline["meta_info"]["prompt_tokens"]

        # start_len == prompt_tokens should pass
        ok = self._send(self._build_payload(routed_experts_start_len=prompt_tokens))
        assert (
            ok.status_code == 200
        ), f"start_len=={prompt_tokens} should pass, got {ok.text}"

        # start_len > prompt_tokens should abort
        too_big = self._send(
            self._build_payload(routed_experts_start_len=prompt_tokens + 1)
        )
        self._assert_aborted(too_big, "is higher than the number of input tokens")

    # ── T-SL-04 ────────────────────────────────────────────────────
    def test_04_start_len_with_cache_hit(self):
        """T-SL-04: start_len handled correctly under radix cache hit.

        NOTE: NPU requires a long prompt (>=256 tokens) to trigger radix cache hit
        because NPU page_size > 1 restricts short sequence caching.
        """
        cache_salt = "cache-hit-rr-test-npu"
        long_text = "What is The capital of France? " * 36
        first = self._send(
            self._build_payload(text=long_text, extra_key=cache_salt)
        ).json()
        assert (
            first["meta_info"].get("cached_tokens", 0) == 0
        ), "first request must be a cold miss"

        prompt_tokens = first["meta_info"]["prompt_tokens"]
        start_len = max(1, prompt_tokens // 2)
        second = self._send(
            self._build_payload(
                text=long_text,
                extra_key=cache_salt,
                routed_experts_start_len=start_len,
            )
        ).json()

        cached = second["meta_info"].get("cached_tokens", 0)
        completion_tokens = second["meta_info"].get("completion_tokens", 0)
        logger.info(
            f"[T-SL-04] prompt_tokens={prompt_tokens} "
            f"start_len={start_len} "
            f"cached_tokens={cached} "
            f"completion_tokens={completion_tokens}"
        )
        assert cached > start_len, (
            f"expected radix prefix past start_len={start_len}, "
            f"got cached_tokens={cached}"
        )

        rows = self._routed_experts(second)
        expected = self._seqlen(second) - 1 - start_len
        logger.info(
            f"[T-SL-04] routed_experts rows={rows.shape[0]} expected={expected}"
        )
        assert (
            rows.shape[0] == expected
        ), f"expected {expected} rows, got {rows.shape[0]}"

    def _assert_aborted(self, resp, expected_substring):
        if resp.status_code == 200:
            body = resp.json()
            meta = body.get("meta_info", {})
            finish_reason = meta.get("finish_reason") or {}
            message = (
                str(finish_reason.get("message", ""))
                + " "
                + str(body.get("text", ""))
                + " "
                + str(body.get("error", ""))
            )
            assert (
                expected_substring in message
            ), f"expected abort with '{expected_substring}', got body={body}"
        else:
            assert resp.status_code >= 400
            assert expected_substring in resp.text


# ══════════════════════════════════════════════════════════════════════
# Test Class 3: Capture W8A8 — INT8 quantized, TP=1
# ══════════════════════════════════════════════════════════════════════
class TestNPURoutingCaptureW8A8(_BaseNPURoutingCaptureTest):
    """
    Verify routing capture on the W8A8 INT8 quantized MoE path.

    [Test Category] RL Routing Capture + W8A8
    [Test Target] POST /v1/completions (return_routed_experts, W8A8)
    """

    model = QWEN3_30B_A3B_W8A8
    tp_size = 1

    def setUp(self):
        self._launch_server()

    # ── T-W8-01 ────────────────────────────────────────────────────
    def test_01_capture_w8a8_shape(self):
        """T-W8-01: Capture shape correct on W8A8 quantized path.

        Verifies: Quantization does not affect routing capture output format.
        """
        result = self._run_decode(max_new_tokens=16)
        routed = result["routed_experts"]
        _assert_experts_valid(routed)
        seqlen = (
            result["meta_info"]["prompt_tokens"]
            + result["meta_info"]["completion_tokens"]
        )
        _assert_experts_shape(routed, seqlen)

    # ── T-W8-02 ────────────────────────────────────────────────────
    def test_02_capture_w8a8_deterministic(self):
        """T-W8-02: Routing deterministic on W8A8 quantized path.

        Verifies: INT8 quantization does not affect routing determinism.
        """
        sig1 = self._run_decode(max_new_tokens=16)
        sig2 = self._run_decode(max_new_tokens=16)

        _assert_logprob_signature_equal(sig1, sig2, msg="T-W8-02: ")
        _assert_experts_equal(
            sig1["routed_experts"], sig2["routed_experts"], msg="T-W8-02: "
        )

    # ── T-W8-03 ────────────────────────────────────────────────────
    def test_03_capture_w8a8_expert_ids_valid(self):
        """T-W8-03: Expert ID validity + uniqueness on W8A8 path.

        Verifies: Quantization path does not affect expert selection validity.
        """
        result = self._run_decode(max_new_tokens=32)
        routed = result["routed_experts"]
        _assert_experts_valid(routed)
        seqlen = (
            result["meta_info"]["prompt_tokens"]
            + result["meta_info"]["completion_tokens"]
        )
        reshaped = _assert_experts_shape(routed, seqlen)

        # Verify top-k experts are unique per token per layer
        for layer in range(_NUM_LAYERS):
            layer_experts = reshaped[:, layer, :]  # [seqlen-1, topk]
            for token_idx in range(layer_experts.shape[0]):
                expert_set = set(layer_experts[token_idx].tolist())
                assert len(expert_set) == _TOPK, (
                    f"Duplicate experts at layer={layer}, token={token_idx}: "
                    f"{layer_experts[token_idx]}"
                )


# ══════════════════════════════════════════════════════════════════════
# Test Class 4: Capture TP=2 — BF16, routing correctness
# ══════════════════════════════════════════════════════════════════════
class TestNPURoutingCaptureTP2(_BaseNPURoutingCaptureTest):
    """
    Verify routing capture correctness under TP=2.

    Verification points:
      1. Capture shape correct under TP=2
      2. Routing deterministic under TP=2
      3. Consecutive decodes consistent under TP=2 (no state pollution)

    [Test Category] RL Routing Capture + TP2
    [Test Target] POST /v1/completions (return_routed_experts, TP=2)
    """

    model = QWEN3_30B_A3B
    tp_size = 2

    def setUp(self):
        self._launch_server()

    # ── T-TP-01 ────────────────────────────────────────────────────
    def test_01_capture_tp2_shape(self):
        """T-TP-01: Capture shape correct under TP=2."""
        result = self._run_decode(max_new_tokens=16)
        routed = result["routed_experts"]
        _assert_experts_valid(routed)
        seqlen = (
            result["meta_info"]["prompt_tokens"]
            + result["meta_info"]["completion_tokens"]
        )
        _assert_experts_shape(routed, seqlen)

    # ── T-TP-02 ────────────────────────────────────────────────────
    def test_02_capture_tp2_deterministic(self):
        """T-TP-02: Routing deterministic under TP=2."""
        sig1 = self._run_decode(max_new_tokens=16)
        sig2 = self._run_decode(max_new_tokens=16)

        _assert_logprob_signature_equal(sig1, sig2, msg="T-TP-02: ")
        _assert_experts_equal(
            sig1["routed_experts"], sig2["routed_experts"], msg="T-TP-02: "
        )

    # ── T-TP-03 ────────────────────────────────────────────────────
    def test_03_capture_tp2_consecutive_consistent(self):
        """T-TP-03: Multiple consecutive decodes consistent under TP=2.

        Verifies no state pollution under TP sharding.
        """
        sig1 = self._run_decode(max_new_tokens=16)
        sig2 = self._run_decode(max_new_tokens=16)
        sig3 = self._run_decode(max_new_tokens=16)

        _assert_logprob_signature_equal(sig1, sig2, msg="T-TP-03: sig1 vs sig2 ")
        _assert_logprob_signature_equal(sig2, sig3, msg="T-TP-03: sig2 vs sig3 ")
        _assert_experts_equal(
            sig1["routed_experts"],
            sig2["routed_experts"],
            msg="T-TP-03: experts sig1 vs sig2 ",
        )
        _assert_experts_equal(
            sig2["routed_experts"],
            sig3["routed_experts"],
            msg="T-TP-03: experts sig2 vs sig3 ",
        )


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main()
