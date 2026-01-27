import asyncio
import logging
import unittest
from typing import List

import aiohttp
import requests
import torch
from torch.nn.utils.rnn import pad_sequence

from sglang.srt.layers.moe.routed_experts_capturer import (
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

register_cuda_ci(est_time=360, suite="stage-c-test-large-4-gpu")

SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
logger = logging.getLogger(__name__)


class TestReturnRoutedExperts(CustomTestCase):
    # modified from test_hicache.py
    @classmethod
    def setUpClass(cls):

        cls.baseline_args = [
            "--enable-return-routed-experts",
            "--enable-deterministic-inference",
            "--disable-overlap-schedule",
            "--disable-cuda-graph",
            "--disable-radix-cache",
            "--tp",
            4,
            "--dp",
            4,
            "--enable-dp-attention",
        ]
        cls.reference_args = [
            "--enable-return-routed-experts",
            "--enable-deterministic-inference",
            "--tp",
            4,
            "--dp",
            4,
            "--enable-dp-attention",
        ]
        cls.sampling_args = {
            "temperature": 0,
        }
        # prepare ShareGPT dataset
        try:
            response = requests.get(SHAREGPT_URL, timeout=60)
            response.raise_for_status()
            data = response.json()
            print(f"Dataset size: {len(data)}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download ShareGPT dataset: {e}") from e
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
            num_mismatches / num_baseline_topks < 0.05
        ), f"Too many mismatches: {num_mismatches} out of {num_baseline_topks} ({num_mismatches/num_baseline_topks:.4%})"

    @classmethod
    def _collect_results(
        cls,
        other_args,
    ):
        process = popen_launch_server(
            DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
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
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("OpenAI response has no choices.")
    sgl_ext = choices[0].get("sgl_ext", None)
    if sgl_ext is None:
        raise ValueError("OpenAI response missing sgl_ext.")
    routed_experts = sgl_ext.get("routed_experts", None)
    if routed_experts is None:
        raise ValueError("OpenAI response sgl_ext missing routed_experts.")
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


if __name__ == "__main__":
    unittest.main()
