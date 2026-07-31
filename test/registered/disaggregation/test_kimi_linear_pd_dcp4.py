import math
import os
import time
import unittest

import requests
import torch
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    popen_launch_server,
)

register_cuda_ci(est_time=1200, suite="nightly-8-gpu-b200", nightly=True)

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
PHYSICAL_PAGE_SIZE = 64
CHUNKED_PREFILL_SIZE = 8192
PARITY_PROMPT_LENGTHS = (63, 64, 65, 255, 256, 257, 8191, 8192, 8193)
LONG_CONTEXT_TOKENS = int(os.environ.get("SGLANG_TEST_PD_LONG_CONTEXT_TOKENS", "32768"))
LONG_CONTEXT_DEPTHS = tuple(
    float(value)
    for value in os.environ.get("SGLANG_TEST_PD_NIAH_DEPTHS", "0.1,0.5,0.9").split(",")
)
LOGPROB_ATOL = 0.20
NIAH_KEY = "739391"
KIMI_LINEAR_MAX_CONTEXT = 1_048_576


def _has_eight_blackwell_gpus() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        return False
    return all(
        torch.cuda.get_device_capability(device_index) >= (10, 0)
        for device_index in range(8)
    )


@unittest.skipUnless(
    _has_eight_blackwell_gpus(),
    "Kimi-Linear PD+DCP acceptance requires eight Blackwell GPUs",
)
class TestKimiLinearPDDCP4(GSM8KMixin, PDDisaggregationServerBase):
    model = KIMI_LINEAR_MODEL
    gsm8k_score_threshold = 0.88
    gsm8k_num_examples = 200
    gsm8k_num_threads = 4
    gsm8k_num_shots = 5

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._collect_monolithic_references()
        cls.launch_all()

    @classmethod
    def _monolithic_reference_args(cls):
        return [
            "--tp-size",
            "4",
            "--ep-size",
            "4",
            "--attention-backend",
            "tokenspeed_mla",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--trust-remote-code",
            "--random-seed",
            "0",
            "--dtype",
            "bfloat16",
            "--page-size",
            str(PHYSICAL_PAGE_SIZE),
            "--chunked-prefill-size",
            str(CHUNKED_PREFILL_SIZE),
            "--cuda-graph-max-bs-decode",
            "64",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--mem-fraction-static",
            "0.80",
        ]

    @classmethod
    def _tokenize(cls, text: str, *, add_special_tokens: bool):
        return cls.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    @classmethod
    def _repeat_to_length(cls, token_ids, length: int):
        if length == 0:
            return []
        assert token_ids, "filler must tokenize to at least one token"
        return (token_ids * math.ceil(length / len(token_ids)))[:length]

    @classmethod
    def _chat_template_parts(cls):
        marker = "SGLANGPDCONTENTSENTINEL739391"
        marker_ids = cls._tokenize(marker, add_special_tokens=False)
        templated = cls.tokenizer.apply_chat_template(
            [{"role": "user", "content": marker}],
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = templated if isinstance(templated, list) else templated["input_ids"]
        marker_starts = [
            index
            for index in range(len(full_ids) - len(marker_ids) + 1)
            if full_ids[index : index + len(marker_ids)] == marker_ids
        ]
        assert len(marker_starts) == 1, (
            "Kimi chat template must contain the user-content marker exactly once, "
            f"found offsets={marker_starts}"
        )
        marker_start = marker_starts[0]
        return (
            full_ids[:marker_start],
            full_ids[marker_start + len(marker_ids) :],
        )

    @classmethod
    def _build_boundary_prompt(cls, target_length: int):
        prefix = cls._tokenize(
            "Read the repeated facts carefully.\n",
            add_special_tokens=True,
        )
        filler = cls._tokenize(
            "The sky is blue and the grass is green. ",
            add_special_tokens=False,
        )
        suffix = cls._tokenize(
            "\nAnswer with one word. The capital of France is",
            add_special_tokens=False,
        )
        middle_length = target_length - len(prefix) - len(suffix)
        assert middle_length >= 0, (
            f"target prompt length {target_length} is shorter than fixed prompt "
            f"content ({len(prefix) + len(suffix)})"
        )
        prompt_ids = prefix + cls._repeat_to_length(filler, middle_length) + suffix
        assert len(prompt_ids) == target_length
        assert target_length < KIMI_LINEAR_MAX_CONTEXT
        return prompt_ids

    @classmethod
    def _build_niah_prompt(cls, target_length: int, depth: float):
        assert 0 <= depth <= 1
        chat_prefix, chat_suffix = cls._chat_template_parts()
        prefix = cls._tokenize(
            (
                "You will read a long collection of mundane records. One record "
                "contains an access code. Remember that code exactly.\n"
            ),
            add_special_tokens=False,
        )
        filler = cls._tokenize(
            (
                "Archive note: the weather was mild, the office lights were on, "
                "and no unusual event was reported.\n"
            ),
            add_special_tokens=False,
        )
        needle = cls._tokenize(
            f"IMPORTANT RECORD: The access code is {NIAH_KEY}.\n",
            add_special_tokens=False,
        )
        query = cls._tokenize(
            "\nWhat is the access code? Reply with the digits only.",
            add_special_tokens=False,
        )
        filler_length = (
            target_length
            - len(chat_prefix)
            - len(prefix)
            - len(needle)
            - len(query)
            - len(chat_suffix)
        )
        assert filler_length >= 0
        before_length = int(filler_length * depth)
        after_length = filler_length - before_length
        prompt_ids = (
            chat_prefix
            + prefix
            + cls._repeat_to_length(filler, before_length)
            + needle
            + cls._repeat_to_length(filler, after_length)
            + query
            + chat_suffix
        )
        assert len(prompt_ids) == target_length
        assert target_length < KIMI_LINEAR_MAX_CONTEXT
        return prompt_ids

    @classmethod
    def _generate(
        cls,
        base_url: str,
        input_ids,
        *,
        max_new_tokens: int,
        ignore_eos: bool = True,
    ):
        response = requests.post(
            base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": ignore_eos,
                },
                "return_logprob": True,
            },
            timeout=900,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _flush_cache(base_url: str):
        response = requests.post(
            base_url + "/flush_cache",
            params={"timeout": 30},
            timeout=120,
        )
        response.raise_for_status()

    @classmethod
    def _flush_pd_caches(cls):
        cls._flush_cache(cls.prefill_url)
        cls._flush_cache(cls.decode_url)

    @classmethod
    def _collect_monolithic_references(cls):
        reference_process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=cls._monolithic_reference_args(),
        )
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(
                cls.model, trust_remote_code=True
            )
            cls.parity_prompts = [
                cls._build_boundary_prompt(target_length)
                for target_length in PARITY_PROMPT_LENGTHS
            ]
            cls._flush_cache(cls.base_url)
            cls.parity_references = [
                cls._generate(cls.base_url, prompt, max_new_tokens=1)
                for prompt in cls.parity_prompts
            ]
            cls._flush_cache(cls.base_url)
            cls.parity_batch_references = cls._generate(
                cls.base_url, cls.parity_prompts, max_new_tokens=1
            )

            cls.niah_prompts = [
                cls._build_niah_prompt(LONG_CONTEXT_TOKENS, needle_depth)
                for needle_depth in LONG_CONTEXT_DEPTHS
            ]
            cls._flush_cache(cls.base_url)
            cls.niah_references = [
                cls._generate(
                    cls.base_url,
                    prompt,
                    max_new_tokens=16,
                    ignore_eos=False,
                )
                for prompt in cls.niah_prompts
            ]
        finally:
            kill_process_tree(reference_process.pid, wait_timeout=60)
            time.sleep(5)

    @classmethod
    def start_prefill(cls):
        args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            "4",
            "--ep-size",
            "4",
            "--attention-backend",
            "tokenspeed_mla",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--dtype",
            "bfloat16",
            "--random-seed",
            "0",
            "--page-size",
            str(PHYSICAL_PAGE_SIZE),
            "--chunked-prefill-size",
            str(CHUNKED_PREFILL_SIZE),
            "--mem-fraction-static",
            "0.80",
        ]
        args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=args,
        )

    @classmethod
    def start_decode(cls):
        args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            "4",
            "--dcp-size",
            "4",
            "--base-gpu-id",
            "4",
            "--attention-backend",
            "tokenspeed_mla",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--dcp-comm-backend",
            "a2a",
            "--dcp-replicate-q-proj",
            "--dtype",
            "bfloat16",
            "--random-seed",
            "0",
            "--page-size",
            str(PHYSICAL_PAGE_SIZE),
            "--cuda-graph-max-bs-decode",
            "64",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--mem-fraction-static",
            "0.80",
        ]
        args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=args,
        )

    def _assert_output_parity(
        self, reference, actual, *, label: str, check_logprobs: bool = True
    ):
        self.assertEqual(
            actual["output_ids"],
            reference["output_ids"],
            f"{label}: generated token IDs diverged",
        )
        if not check_logprobs:
            return

        reference_logprobs = reference["meta_info"]["output_token_logprobs"]
        actual_logprobs = actual["meta_info"]["output_token_logprobs"]
        self.assertEqual(
            len(actual_logprobs),
            len(reference_logprobs),
            f"{label}: output logprob lengths diverged",
        )

        max_delta = 0.0
        for token_index, (expected, observed) in enumerate(
            zip(reference_logprobs, actual_logprobs)
        ):
            self.assertEqual(
                int(observed[1]),
                int(expected[1]),
                f"{label}: logprob token ID diverged at output {token_index}",
            )
            max_delta = max(max_delta, abs(float(observed[0]) - float(expected[0])))
        self.assertLessEqual(
            max_delta,
            LOGPROB_ATOL,
            f"{label}: max output logprob delta {max_delta:.6f}",
        )

    def test_monolithic_pd_token_and_logprob_parity(self):
        self._flush_pd_caches()
        for target_length, prompt, reference in zip(
            PARITY_PROMPT_LENGTHS, self.parity_prompts, self.parity_references
        ):
            with self.subTest(prompt_tokens=target_length):
                actual = self._generate(self.base_url, prompt, max_new_tokens=1)
                self.assertEqual(actual["meta_info"]["prompt_tokens"], target_length)
                self._assert_output_parity(
                    reference,
                    actual,
                    label=f"sequential prompt_tokens={target_length}",
                )

    def test_chunk_and_virtual_page_boundary_batch_parity(self):
        self._flush_pd_caches()
        actual_outputs = self._generate(
            self.base_url, self.parity_prompts, max_new_tokens=1
        )
        self.assertEqual(len(actual_outputs), len(self.parity_batch_references))
        for target_length, reference, actual in zip(
            PARITY_PROMPT_LENGTHS, self.parity_batch_references, actual_outputs
        ):
            with self.subTest(prompt_tokens=target_length):
                self.assertEqual(actual["meta_info"]["prompt_tokens"], target_length)
                self._assert_output_parity(
                    reference,
                    actual,
                    label=f"batched prompt_tokens={target_length}",
                    check_logprobs=False,
                )

    def test_long_context_needle_parity(self):
        self._flush_pd_caches()
        for needle_depth, prompt, reference in zip(
            LONG_CONTEXT_DEPTHS, self.niah_prompts, self.niah_references
        ):
            with self.subTest(
                prompt_tokens=LONG_CONTEXT_TOKENS, needle_depth=needle_depth
            ):
                self.assertIn(NIAH_KEY, reference["text"])
                actual = self._generate(
                    self.base_url,
                    prompt,
                    max_new_tokens=16,
                    ignore_eos=False,
                )
                self.assertIn(NIAH_KEY, actual["text"])
                self._assert_output_parity(
                    reference,
                    actual,
                    label=(
                        f"niah prompt_tokens={LONG_CONTEXT_TOKENS} "
                        f"depth={needle_depth}"
                    ),
                )

    def _assert_batch_completes(self, batch_size: int):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": [
                    f"Reply with one short word for request {index}: the sky is"
                    for index in range(batch_size)
                ],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        outputs = response.json()
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), batch_size)
        self.assertTrue(all(output["text"].strip() for output in outputs))

    def test_decode_cuda_graph_and_eager_batch(self):
        self._assert_batch_completes(2)
        self._assert_batch_completes(2)
        self._assert_batch_completes(65)

    def test_decode_physical_capacity_sanity(self):
        response = requests.get(self.decode_url + "/server_info", timeout=30)
        response.raise_for_status()
        self.assertGreater(response.json()["max_total_num_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
