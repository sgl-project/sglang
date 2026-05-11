import asyncio
import json
import logging
import unittest
from typing import List

import aiohttp
import numpy as np
import requests
import torch
from torch.nn.utils.rnn import pad_sequence

from sglang.benchmark.utils import download_and_cache_hf_file
from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=194, suite="stage-c-test-4-gpu-h100")

# FP8 variant of Qwen3-30B-A3B: required because DeepEP normal/LL fast paths in
# ep_moe/layer.py only run for {Fp8Config (via deep_gemm), W4AFp8Config, aiter,
# NPU, modelopt_fp4+cutedsl}. Bf16 hits an `assert False, "deprecated"` today.
MODEL_PATH = "Qwen/Qwen3-30B-A3B-FP8"

SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
logger = logging.getLogger(__name__)

_QWEN3_30B_A3B_NUM_LAYERS = 48
_QWEN3_30B_A3B_TOPK = 8


class TestReturnRoutedExperts(CustomTestCase):
    """End-to-end check that --enable-return-routed-experts stays correct
    under DeepEP a2a + attn_tp_size > 1, across overlap/cuda-graph/radix
    optimisations.

    Both servers run ``--tp 4 --dp 2 --enable-dp-attention --moe-a2a-backend
    deepep`` so attn_tp_size=2 and the all-gather hot path in
    RoutedExpertsCapturer.capture is hit on every step. Baseline disables
    overlap/cuda-graph/radix to give a deterministic ground truth; reference
    leaves them on. If the gather were skipping a rank or racing against the
    forward stream, the captured topk_ids would diverge between the two.
    """

    @classmethod
    def setUpClass(cls):
        common = [
            "--enable-return-routed-experts",
            "--enable-deterministic-inference",
            "--tp",
            4,
            "--dp",
            2,
            "--enable-dp-attention",
            "--moe-a2a-backend",
            "deepep",
            # Force normal-mode dispatch: deepep auto routes decode through
            # low_latency mode whose buffer (num_max_dispatch_tokens_per_rank)
            # is undersized for cuda graph capture at default --cuda-graph-max-bs.
            "--deepep-mode",
            "normal",
        ]
        cls.baseline_args = common + [
            "--disable-overlap-schedule",
            "--disable-cuda-graph",
            "--disable-radix-cache",
        ]
        cls.reference_args = common
        cls.sampling_args = {"temperature": 0}
        # prepare ShareGPT dataset
        dataset_path = download_and_cache_hf_file(SHAREGPT_REPO_ID, SHAREGPT_FILENAME)
        with open(dataset_path) as f:
            data = json.load(f)
        print(f"Dataset size: {len(data)}")
        cls.texts = []
        for s in data:
            if "conversations" in s and len(s["conversations"]) > 0:
                try:
                    text = s["conversations"][0]["value"]
                    if isinstance(text, str) and len(text) <= 2000:
                        cls.texts.append(text)
                except (KeyError, IndexError, TypeError) as e:
                    print(f"Warning: Skipping invalid conversation data: {e}")
                    continue

        if not cls.texts:
            raise ValueError("No valid texts found in the dataset")
        cls.texts = cls.texts[:100]
        cls._endpoints = [
            (
                "/generate",
                cls._build_generate_payload,
                extract_routed_experts_from_meta_info,
            ),
            (
                "/v1/chat/completions",
                cls._build_chat_payload,
                extract_routed_experts_from_openai_response,
            ),
            (
                "/v1/completions",
                cls._build_completion_payload,
                extract_routed_experts_from_openai_response,
            ),
        ]
        cls.baseline_results = cls._collect_results(cls.baseline_args)
        cls.reference_results = cls._collect_results(cls.reference_args)

    @classmethod
    def test_return_routed_experts(cls):
        cls._run_endpoint_test("/generate")

    @classmethod
    def test_return_routed_experts_chat_completions(cls):
        cls._run_endpoint_test("/v1/chat/completions")

    @classmethod
    def test_return_routed_experts_completions(cls):
        cls._run_endpoint_test("/v1/completions")

    @classmethod
    def _run_endpoint_test(cls, endpoint):
        captured_baseline_experts = cls.baseline_results[endpoint]
        captured_reference_experts = cls.reference_results[endpoint]

        check_all_experts_id_valid(captured_baseline_experts)
        check_all_experts_id_valid(captured_reference_experts)

        num_baseline_topks = (
            sum([len(seq) for seq in captured_baseline_experts])
            * len(captured_baseline_experts[0][0])
            * len(captured_baseline_experts[0][0][0])
        )

        num_mismatches = compare_baseline_w_reference(
            captured_baseline_experts, captured_reference_experts
        )
        logger.info(
            f"Total mismatches report: {num_mismatches} out of {num_baseline_topks} ({num_mismatches/num_baseline_topks:.4%})"
        )
        print(
            f"Total mismatches report: {num_mismatches} out of {num_baseline_topks} ({num_mismatches/num_baseline_topks:.4%})"
        )
        assert (
            num_mismatches / num_baseline_topks < 0.10
        ), f"Too many mismatches: {num_mismatches} out of {num_baseline_topks} ({num_mismatches/num_baseline_topks:.4%})"

    @classmethod
    def _collect_results(
        cls,
        other_args,
    ):
        process = popen_launch_server(
            MODEL_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        try:
            return asyncio.run(cls._collect_results_async())
        finally:
            kill_process_tree(process.pid)

    @classmethod
    async def _collect_results_async(cls):
        results = {}
        async with aiohttp.ClientSession() as session:
            for endpoint, payload_builder, response_extractor in cls._endpoints:
                tasks = [
                    asyncio.create_task(
                        make_request(
                            session,
                            f"{DEFAULT_URL_FOR_TEST}{endpoint}",
                            payload_builder(text),
                        )
                    )
                    for text in cls.texts
                ]
                # return value shape: List[[seq_len, num_layers, topk]...]
                http_result = await asyncio.gather(*tasks)
                results[endpoint] = [
                    response_extractor(res).reshape(-1, 48, 8) for res in http_result
                ]
        return results

    @classmethod
    def _build_generate_payload(cls, text):
        return {
            "text": text,
            "sampling_params": cls.sampling_args,
            "return_routed_experts": True,
            "max_new_tokens": 100,
        }

    @classmethod
    def _build_chat_payload(cls, text):
        return {
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "max_tokens": 100,
            "return_routed_experts": True,
        }

    @classmethod
    def _build_completion_payload(cls, text):
        return {
            "prompt": text,
            "temperature": 0,
            "max_tokens": 100,
            "return_routed_experts": True,
        }


async def make_request(session, url, payload):
    """Make a single async HTTP request"""
    async with session.post(url=url, json=payload) as response:
        return await response.json()


def extract_routed_experts_from_openai_response(response):
    if "error" in response:
        raise ValueError(f"OpenAI response error: {response['error']}")
    # sglext is at response level (not in choices) as of PR #17648
    sglext = response.get("sglext", None)
    if sglext is None:
        raise ValueError("OpenAI response missing sglext.")
    routed_experts = sglext.get("routed_experts", None)
    if routed_experts is None:
        raise ValueError("OpenAI response sglext missing routed_experts.")
    return extract_routed_experts_from_meta_info(
        {"meta_info": {"routed_experts": routed_experts}}
    )


def check_all_experts_id_valid(experts: List[List[List[int]]]):
    tensor_list = [torch.tensor(lst) for lst in experts]
    padded_tensor = pad_sequence(tensor_list, batch_first=True, padding_value=0)

    # temporary hardcode as we only use Qwen3 30BA3B
    if not ((padded_tensor >= 0) & (padded_tensor <= 127)).all():
        raise ValueError(
            f"Some expert indices are out of valid range [0, 127], MAX: {padded_tensor.max()} MIN: {padded_tensor.min()}"
        )


def compare_baseline_w_reference(baseline, reference):
    num_total_mismatches = 0
    for baseline_seq, reference_seq in zip(baseline, reference):
        for bsl_token, ref_token in zip(baseline_seq, reference_seq):
            for bsl_topk, ref_topk in zip(bsl_token, ref_token):
                len_bsl, len_ref = len(bsl_topk), len(ref_topk)
                set_bsl, set_ref = set(bsl_topk), set(ref_topk)
                if set_bsl != set_ref:
                    num_total_mismatches += len(set_bsl - set_ref)
                if (len_bsl != len_ref) or (len_bsl != len(set_bsl)):
                    raise ValueError(
                        f"Duplicates experts ids found: Baseline({len_bsl}): {bsl_topk} vs Reference({len_ref}): {ref_topk}"
                    )
    return num_total_mismatches


class TestRoutedExpertsStartLen(CustomTestCase):
    """Verify the `routed_experts_start_len` parameter:

    - default (0) returns the full sequence
    - explicit start_len crops the response and the cropped tail matches
      the corresponding tail of the full response
    """

    MAX_NEW_TOKENS = 8

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-return-routed-experts",
                "--enable-deterministic-inference",
                "--tp",
                2,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _send(self, payload: dict) -> dict:
        resp = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate", json=payload, timeout=120
        )
        return resp

    def _build_payload(self, **extra) -> dict:
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

    def _routed_experts(self, resp_json: dict):
        return extract_routed_experts_from_meta_info(resp_json).reshape(
            -1, _QWEN3_30B_A3B_NUM_LAYERS, _QWEN3_30B_A3B_TOPK
        )

    def _seqlen(self, resp_json: dict) -> int:
        meta = resp_json["meta_info"]
        return meta["prompt_tokens"] + meta["completion_tokens"]

    def test_start_len_zero_is_default(self):
        """Omitting the field must match `routed_experts_start_len=0`,
        which returns the full sequence (start_len=0)."""
        resp_default = self._send(self._build_payload()).json()
        resp_zero = self._send(self._build_payload(routed_experts_start_len=0)).json()

        rows_default = self._routed_experts(resp_default)
        rows_zero = self._routed_experts(resp_zero)

        seqlen_default = self._seqlen(resp_default)
        seqlen_zero = self._seqlen(resp_zero)
        self.assertEqual(seqlen_default, seqlen_zero)
        self.assertEqual(rows_default.shape[0], seqlen_default - 1)
        self.assertEqual(rows_zero.shape[0], seqlen_zero - 1)
        self.assertTrue(
            np.array_equal(rows_default, rows_zero),
            "default and explicit 0 must produce identical routed experts",
        )

    def test_start_len_controls_row_count(self):
        """`routed_experts_start_len=N` must return `seqlen - 1 - N` rows
        and the returned tail must match the corresponding tail of the
        full sequence (start_len omitted)."""
        full_resp = self._send(self._build_payload()).json()
        full_rows = self._routed_experts(full_resp)
        seqlen = self._seqlen(full_resp)
        self.assertEqual(full_rows.shape[0], seqlen - 1)

        start_len = max(1, full_resp["meta_info"]["prompt_tokens"] // 2)

        cropped_resp = self._send(
            self._build_payload(routed_experts_start_len=start_len)
        ).json()
        cropped_rows = self._routed_experts(cropped_resp)
        cropped_seqlen = self._seqlen(cropped_resp)

        self.assertEqual(seqlen, cropped_seqlen)
        expected_rows = seqlen - 1 - start_len
        self.assertEqual(
            cropped_rows.shape[0],
            expected_rows,
            f"expected {expected_rows} rows, got {cropped_rows.shape[0]}",
        )
        self.assertTrue(
            np.array_equal(full_rows[start_len:], cropped_rows),
            "cropped routed experts must match the tail of the full sequence",
        )

    def test_start_len_exceeds_prompt_tokens_aborts(self):
        """`routed_experts_start_len > prompt_tokens` must abort the request:
        the caller cannot meaningfully reference positions that don't exist
        in the prompt yet."""
        baseline = self._send(self._build_payload()).json()
        prompt_tokens = baseline["meta_info"]["prompt_tokens"]

        ok = self._send(self._build_payload(routed_experts_start_len=prompt_tokens))
        self.assertEqual(
            ok.status_code,
            200,
            f"start_len=={prompt_tokens} should pass, got {ok.text}",
        )

        too_big = self._send(
            self._build_payload(routed_experts_start_len=prompt_tokens + 1)
        )
        self._assert_aborted(too_big, "is higher than the number of input tokens")

    def test_start_len_with_cache_hit(self):
        """`start_len` must allow the radix prefix to extend past it. The
        first request seeds the cache; the second sends the same prompt
        with `start_len` somewhere inside the prompt. We verify:

          - meta_info.cached_tokens > start_len (would be impossible if a
            cap forced the prefix match to <= start_len),
          - the response row count still equals `seqlen - 1 - start_len`.
        """
        cache_salt = "cache-hit-test"
        first = self._send(self._build_payload(extra_key=cache_salt)).json()
        self.assertEqual(
            first["meta_info"].get("cached_tokens", 0),
            0,
            "first request must be a cold miss",
        )

        prompt_tokens = first["meta_info"]["prompt_tokens"]
        start_len = max(1, prompt_tokens // 2)
        second = self._send(
            self._build_payload(
                extra_key=cache_salt,
                routed_experts_start_len=start_len,
            )
        ).json()

        cached = second["meta_info"].get("cached_tokens", 0)
        self.assertGreater(
            cached,
            start_len,
            f"expected radix prefix past start_len={start_len}, "
            f"got cached_tokens={cached} (cap not removed?)",
        )

        rows = self._routed_experts(second)
        expected = self._seqlen(second) - 1 - start_len
        self.assertEqual(
            rows.shape[0],
            expected,
            f"expected {expected} rows, got {rows.shape[0]}",
        )

    def _assert_aborted(self, resp, expected_substring: str):
        """Assert a request was aborted with `expected_substring` in the
        error message."""
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
            self.assertIn(
                expected_substring,
                message,
                f"expected abort with '{expected_substring}', got body={body}",
            )
        else:
            self.assertGreaterEqual(resp.status_code, 400)
            self.assertIn(expected_substring, resp.text)


if __name__ == "__main__":
    unittest.main()
