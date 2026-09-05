"""Routed-expert capture (R3) under speculative decoding.

The capture tape is keyed by KV slot, so a verify step that relocates accepted
KV has to relocate the capture with it, and a verify worker that drops the
capture handle leaves the tape at its zero-initialised contents.

Both arms check one request against itself on one server: its DECODE rows must
agree with the PREFILL rows for the same token sequence. Prefill never runs the
verify path, so it is the no-speculation ground truth, and a fresh cache salt
forces the reference request through a cold extend rather than the tape the
decode steps just wrote.

The NGRAM arm needs no draft model and drafts an irregular tree, so it covers
both the capture-handle and the accepted-path failures. The EAGLE arm runs
`--speculative-eagle-topk 4`, and topk > 1 is the only setting where the
accepted path is not already the front of each per-request block.
"""

import unittest

import numpy as np
import requests
from transformers import AutoConfig

from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=1200, stage="extra-a", runner_config="2-gpu-large")

# Repetitive on purpose: NGRAM drafts from the corpus it has already seen, so a
# prompt that invites repetition keeps the verify path busy.
PROMPT = (
    "Repeat the following list back exactly, then continue it in the same "
    "format: item one, item two, item three, item four, item five, item six, "
    "item seven, item eight, item nine, item ten, item eleven, item twelve,"
)
MAX_NEW_TOKENS = 96

# Prefill and decode take different kernels and batch shapes, so a small number
# of near-tied router scores flip between them. Same budget the DeepEP
# routed-experts parity test uses for the same reason.
MAX_MISMATCH_FRACTION = 0.10


def _count_expert_mismatches(lhs: np.ndarray, rhs: np.ndarray) -> int:
    """Expert ids present on one side but not the other, per (token, layer).

    Compared as sets: the captured top-k order is score order, which can differ
    between two kernels that select the same experts.
    """
    mismatches = 0
    for lhs_token, rhs_token in zip(lhs, rhs):
        for lhs_layer, rhs_layer in zip(lhs_token, rhs_token):
            mismatches += len(set(lhs_layer.tolist()) - set(rhs_layer.tolist()))
    return mismatches


class _SpecRoutedExpertsMixin:
    """Subclasses set `model`, `server_args` and (if needed) `hf_config_kwargs`."""

    model: str
    server_args: list
    hf_config_kwargs: dict = {}

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_layers = AutoConfig.from_pretrained(
            cls.model, **cls.hf_config_kwargs
        ).num_hidden_layers
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.server_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, payload: dict) -> dict:
        resp = requests.post(f"{self.base_url}/generate", json=payload, timeout=600)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def _routing_rows(self, body: dict) -> np.ndarray:
        """`[seqlen - 1, num_layers, topk]`: one row per input position."""
        meta = body["meta_info"]
        rows = meta["prompt_tokens"] + meta["completion_tokens"] - 1
        flat = extract_routed_experts_from_meta_info(body)
        self.assertEqual(flat.size % (rows * self.num_layers), 0, "unexpected shape")
        return flat.reshape(rows, self.num_layers, -1)

    def _assert_drafts_were_accepted(self):
        """Without acceptance the verify path under test never runs and the
        comparison below would pass on any implementation."""
        internal = requests.get(f"{self.base_url}/server_info").json()
        accept_length = internal["internal_states"][0].get("avg_spec_accept_length")
        self.assertIsNotNone(accept_length, "server reported no accept length")
        self.assertGreater(
            accept_length,
            1.0,
            "no draft token was accepted, so this test proves nothing",
        )

    def test_decode_routing_matches_prefill_routing(self):
        spec = self._generate(
            {
                "text": PROMPT,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "ignore_eos": True,
                },
                "return_routed_experts": True,
                "return_logprob": True,
                "logprob_start_len": 0,
                "extra_key": "spec-decode",
            }
        )
        meta = spec["meta_info"]
        prompt_len = meta["prompt_tokens"]
        token_ids = [tok for _, tok, _ in meta["input_token_logprobs"]] + [
            tok for _, tok, _ in meta["output_token_logprobs"]
        ]
        self.assertEqual(len(token_ids), prompt_len + meta["completion_tokens"])
        self._assert_drafts_were_accepted()

        spec_rows = self._routing_rows(spec)
        self.assertTrue(
            spec_rows.any(),
            "routed_experts came back entirely zero: the verify path dropped the "
            "capture handle, so the host cache was never written",
        )

        reference = self._generate(
            {
                "input_ids": token_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
                "return_routed_experts": True,
                "extra_key": "prefill-reference",
            }
        )
        self.assertEqual(
            reference["meta_info"].get("cached_tokens", 0),
            0,
            "reference request hit the radix cache, so its rows are the decode "
            "rows under test rather than a fresh prefill",
        )
        reference_rows = self._routing_rows(reference)

        # Row i is the routing for the token at position i, so decode rows start
        # at prompt_len; the reference covers the same positions from prefill.
        decode = spec_rows[prompt_len:]
        self.assertGreater(decode.shape[0], 0, "no decode rows to compare")
        mismatches = _count_expert_mismatches(
            decode, reference_rows[prompt_len : spec_rows.shape[0]]
        )
        fraction = mismatches / decode.size
        self.assertLess(
            fraction,
            MAX_MISMATCH_FRACTION,
            f"{mismatches} of {decode.size} captured expert ids differ from the "
            f"prefill capture of the same tokens ({fraction:.2%}); the decode "
            "rows are describing different tokens",
        )


class TestNgramRoutedExperts(_SpecRoutedExpertsMixin, CustomTestCase):
    model = DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST
    server_args = [
        "--tp",
        "2",
        "--enable-return-routed-experts",
        "--speculative-algorithm",
        "NGRAM",
        "--speculative-num-draft-tokens",
        "16",
    ]


class TestEagleTopkRoutedExperts(_SpecRoutedExpertsMixin, CustomTestCase):
    model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
    hf_config_kwargs = {"trust_remote_code": True}
    server_args = [
        "--trust-remote-code",
        "--tp",
        "2",
        "--enable-return-routed-experts",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-draft-model-path",
        DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
        # Same tree as the topk > 1 arm of test/manual/attention/test_fa3.py,
        # which holds an accept length above 2.95 on this model pair. topk > 1
        # is the point: at topk == 1 the accepted path already is the front of
        # each block and the compaction under test is an identity.
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "4",
        "--speculative-num-draft-tokens",
        "8",
    ]


if __name__ == "__main__":
    unittest.main()
