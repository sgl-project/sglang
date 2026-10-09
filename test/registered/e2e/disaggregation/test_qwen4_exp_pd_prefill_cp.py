"""Qwen3.8-Flash-Next (Qwen4-Exp) PD with a CP-TP group sharing prefill node.

Prefill runs TP4 with prefill CP4 on GPUs 0-3, and the CP group is the TP
group. The residual stream, QSA attention and the QSA indexer are CP-sharded;
the MoE and the GDN keep their TP partition. Decode runs plain TP4 on GPUs
4-7 and sees only what the prefill node transferred, so an accuracy gap is the
fault of the prefill CP path or of the transfer.

CP-TP group sharing is derived, not requested: with a wrong recipe the prefill
node boots without it and still serves. ``test_a_cp_tp_group_sharing_is_derived``
pins the resolved flag.
"""

import unittest

import requests
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import try_cached_model

register_cuda_ci(est_time=600, stage="nightly", runner_config="8-gpu-b200")

MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
CP_SIZE = 4

# The recipe of test/registered/e2e/models/test_qwen4_exp_models.py, on both
# sides so the transferred GDN state has the same dtype and layout.
COMMON_ARGS = [
    "--chunked-prefill-size",
    "8192",
    "--linear-attn-prefill-backend",
    "flashinfer",
    "--linear-attn-decode-backend",
    "flashinfer",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--reasoning-parser",
    "qwen3-thinking",
]
PREFILL_ARGS = COMMON_ARGS + [
    "--attn-cp-size",
    str(CP_SIZE),
    "--enable-prefill-cp",
    "--cp-strategy",
    "zigzag",
    # Attention TP is 1: every rank holds all attention heads and the whole
    # sequence's KV.
    "--mem-fraction-static",
    "0.8",
]
DECODE_ARGS = COMMON_ARGS + ["--mem-fraction-static", "0.85"]

# Past the indexer budget (2048 tokens), so QSA selects a sparse subset, and
# past the chunk size, so later chunks run CP over a cached prefix.
NEEDLE_PROMPT_TOKENS = 20000
NEEDLE_DEPTHS = (0.1, 0.5, 0.9)
NEEDLE_KEY = "739391"


class TestQwen4ExpPDPrefillCP(GSM8KMixin, PDDisaggregationServerBase):
    model = try_cached_model(MODEL)
    prefill_tp_size = CP_SIZE
    decode_tp_size = CP_SIZE
    decode_base_gpu_id = CP_SIZE
    extra_prefill_args = PREFILL_ARGS
    extra_decode_args = DECODE_ARGS

    # The gate of the plain-TP4 sibling, test_qwen4_exp_models.py.
    gsm8k_backend = "sgl_eval"
    gsm8k_thinking = True
    gsm8k_num_examples = 200
    gsm8k_num_threads = 32
    gsm8k_max_tokens = 16384
    gsm8k_score_threshold = 0.94

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model)
        cls.launch_all()

    def test_a_cp_tp_group_sharing_is_derived(self):
        info = requests.get(self.prefill_url + "/server_info", timeout=60).json()
        self.assertTrue(
            info["enable_cp_tp_group_sharing"],
            "CP-TP group sharing was not derived on the prefill node "
            f"(attn_cp_size={info.get('attn_cp_size')}, "
            f"enable_prefill_cp={info.get('enable_prefill_cp')})",
        )
        self.assertEqual(info["attn_cp_size"], CP_SIZE)
        self.assertEqual(info["tp_size"], CP_SIZE)

    def _needle_prompt(self, depth: float) -> str:
        filler = (
            "Archive note: the weather was mild, the office lights were on, "
            "and no unusual event was reported.\n"
        )
        repeats = NEEDLE_PROMPT_TOKENS // len(self.tokenizer.encode(filler))
        before = int(repeats * depth)
        document = (
            filler * before
            + f"IMPORTANT RECORD: The access code is {NEEDLE_KEY}.\n"
            + filler * (repeats - before)
        )
        return self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "One of these records contains an access code.\n"
                    + document
                    + "\nWhat is the access code? Reply with the digits only.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def test_long_context_needle(self):
        for depth in NEEDLE_DEPTHS:
            with self.subTest(depth=depth):
                response = requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": self._needle_prompt(depth),
                        "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                    },
                    timeout=600,
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn(NEEDLE_KEY, response.json()["text"])


if __name__ == "__main__":
    unittest.main()
