import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=700, stage="nightly", runner_config="8-gpu-b200")

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
PHYSICAL_PAGE_SIZE = 64
CHUNKED_PREFILL_SIZE = 8192
LONG_CONTEXT_TOKENS = int(os.environ.get("SGLANG_TEST_PD_LONG_CONTEXT_TOKENS", "32768"))
LONG_CONTEXT_DEPTHS = tuple(
    float(value)
    for value in os.environ.get("SGLANG_TEST_PD_NIAH_DEPTHS", "0.1,0.5,0.9").split(",")
)
NIAH_KEY = "739391"

COMMON_ARGS = [
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
    "--cuda-graph-backend-prefill",
    "disabled",
    "--mem-fraction-static",
    "0.80",
]
PREFILL_ARGS = [
    "--ep-size",
    "4",
    "--chunked-prefill-size",
    str(CHUNKED_PREFILL_SIZE),
]
DECODE_ARGS = [
    "--dcp-size",
    "4",
    "--dcp-comm-backend",
    "a2a",
    "--dcp-replicate-q-proj",
    "--cuda-graph-max-bs-decode",
    "64",
]
# Mooncake >=0.3.13 (kvcache-ai/Mooncake#2974) bounds TCP admission per peer
# and hard-fails the overflow instead of applying backpressure. The TP4/EP4 ->
# TP4/DCP4 fan-out blows past the 1024 + 1024 defaults on hosts that fall back
# to TCP, killing every KV transfer.
MOONCAKE_TCP_ENV = {
    "MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER": "65535",
    "MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER": "65535",
}


def _has_eight_blackwell_gpus() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        return False
    return all(
        torch.cuda.get_device_capability(device_index) >= (10, 0)
        for device_index in range(8)
    )


def _write_dummy_qwen3_dspark_draft(root: Path) -> str:
    draft_dir = root / "qwen3-dspark-kimi-proxy"
    draft_dir.mkdir()
    config = {
        "architectures": ["Qwen3DSparkModel"],
        "model_type": "qwen3",
        "dtype": "bfloat16",
        "hidden_size": 2304,
        "intermediate_size": 9216,
        "num_hidden_layers": 5,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "max_position_embeddings": 1048576,
        "rope_parameters": {
            "rope_theta": 10000.0,
            "rope_type": "default",
        },
        "vocab_size": 163840,
        "bos_token_id": 163584,
        "eos_token_id": 163586,
        "mask_token_id": 163839,
        "block_size": 7,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "num_target_layers": 27,
        "target_layer_ids": [1, 7, 13, 19, 26],
        "layer_types": ["full_attention"] * 5,
        "tie_word_embeddings": False,
        "use_cache": True,
    }
    (draft_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(draft_dir)


def _generate(*, base_url: str, payload: dict):
    response = requests.post(base_url + "/generate", json=payload, timeout=900)
    response.raise_for_status()
    return response.json()


def _flush_cache(base_url: str):
    response = requests.post(
        base_url + "/flush_cache", params={"timeout": 30}, timeout=120
    )
    response.raise_for_status()


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
    prefill_tp_size = 4
    decode_tp_size = 4
    decode_base_gpu_id = 4
    extra_prefill_args = COMMON_ARGS + PREFILL_ARGS
    extra_decode_args = COMMON_ARGS + DECODE_ARGS

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Set before launch_all() so the server subprocesses inherit it.
        os.environ.update(MOONCAKE_TCP_ENV)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, trust_remote_code=True)
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        for key in MOONCAKE_TCP_ENV:
            os.environ.pop(key, None)
        super().tearDownClass()

    @classmethod
    def _tokenize(cls, text: str):
        return cls.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _repeat_to_length(*, token_ids, length: int):
        if length == 0:
            return []
        assert token_ids, "filler must tokenize to at least one token"
        return (token_ids * math.ceil(length / len(token_ids)))[:length]

    @classmethod
    def _chat_template_parts(cls):
        marker = "SGLANGPDCONTENTSENTINEL739391"
        marker_ids = cls._tokenize(marker)
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
    def _build_niah_prompt(cls, *, target_length: int, depth: float):
        assert 0 <= depth <= 1
        chat_prefix, chat_suffix = cls._chat_template_parts()
        prefix = cls._tokenize(
            "You will read a long collection of mundane records. One record "
            "contains an access code. Remember that code exactly.\n"
        )
        filler = cls._tokenize(
            "Archive note: the weather was mild, the office lights were on, "
            "and no unusual event was reported.\n"
        )
        needle = cls._tokenize(f"IMPORTANT RECORD: The access code is {NIAH_KEY}.\n")
        query = cls._tokenize("\nWhat is the access code? Reply with the digits only.")
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
        prompt_ids = (
            chat_prefix
            + prefix
            + cls._repeat_to_length(token_ids=filler, length=before_length)
            + needle
            + cls._repeat_to_length(
                token_ids=filler, length=filler_length - before_length
            )
            + query
            + chat_suffix
        )
        assert len(prompt_ids) == target_length
        return prompt_ids

    def _generate_needle(self, prompt):
        return _generate(
            base_url=self.base_url,
            payload={
                "input_ids": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
        )

    def test_long_context_needle(self):
        _flush_cache(self.prefill_url)
        _flush_cache(self.decode_url)
        for needle_depth in LONG_CONTEXT_DEPTHS:
            prompt = self._build_niah_prompt(
                target_length=LONG_CONTEXT_TOKENS, depth=needle_depth
            )
            with self.subTest(
                prompt_tokens=LONG_CONTEXT_TOKENS, needle_depth=needle_depth
            ):
                self.assertIn(NIAH_KEY, self._generate_needle(prompt)["text"])
                # Prefill now holds the long prefix. Decode must receive it
                # again, even though prefill computes almost no new tokens.
                _flush_cache(self.decode_url)
                cached = self._generate_needle(prompt)
                self.assertGreater(
                    cached["meta_info"]["cached_tokens"], CHUNKED_PREFILL_SIZE
                )
                self.assertIn(NIAH_KEY, cached["text"])

    def _assert_batch_completes(self, batch_size: int):
        outputs = _generate(
            base_url=self.base_url,
            payload={
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
        )
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


@unittest.skipUnless(
    _has_eight_blackwell_gpus(),
    "Kimi-Linear PD DCP4 + DSPARK requires eight Blackwell GPUs",
)
class TestKimiLinearPDDCP4DSpark(GSM8KMixin, PDDisaggregationServerBase):
    model = KIMI_LINEAR_MODEL
    gsm8k_score_threshold = 0.88
    gsm8k_num_examples = 400
    gsm8k_num_threads = 64
    gsm8k_num_shots = 5
    prefill_tp_size = 4
    decode_tp_size = 4
    decode_base_gpu_id = 4
    extra_prefill_env = {"SGLANG_RAGGED_VERIFY_MODE": "static"}
    extra_decode_env = {"SGLANG_RAGGED_VERIFY_MODE": "static"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ.update(MOONCAKE_TCP_ENV)
        cls._draft_root = tempfile.mkdtemp(prefix="dspark_pd_dcp_draft_")
        draft_path = _write_dummy_qwen3_dspark_draft(Path(cls._draft_root))
        dspark_args = [
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            draft_path,
            "--speculative-draft-load-format",
            "dummy",
            "--speculative-attention-mode",
            "decode",
            "--speculative-draft-attention-backend",
            "trtllm_mha",
        ]
        cls.extra_prefill_args = COMMON_ARGS + dspark_args + PREFILL_ARGS
        cls.extra_decode_args = COMMON_ARGS + dspark_args + DECODE_ARGS
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        for key in MOONCAKE_TCP_ENV:
            os.environ.pop(key, None)
        shutil.rmtree(cls._draft_root, ignore_errors=True)
        super().tearDownClass()

    def test_spec_verify_runs_on_decode(self):
        output = _generate(
            base_url=self.base_url,
            payload={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "ignore_eos": True,
                },
            },
        )
        meta_info = output["meta_info"]
        self.assertGreater(
            meta_info.get("spec_verify_ct", 0),
            0,
            "DSPARK verify did not run on the decode side",
        )
        self.assertGreater(meta_info["completion_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
