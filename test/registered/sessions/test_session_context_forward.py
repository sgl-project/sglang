"""E2E tests for the session context forward primitive (PR-2).

A context forward is a normal-queue request carrying input_ids (prefix +
query span, for radix matching) + input_embeds covering the span tail +
query_attention="bidirectional" + explicit token_positions +
max_new_tokens=0, returning the span's hidden states. With a single query
token, bidirectional attention equals causal attention, so the hidden state
must be bit-equal to a plain extend of the token the embeds encode, and the
scratch span must never enter the prefix cache.
"""

import unittest

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

PROMPT = "The capital of France is Paris. The capital of Germany is"


class TestSessionContextForward(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-return-hidden-states"],
        )
        tokenizer = AutoTokenizer.from_pretrained(cls.model)
        cls.prompt_ids = tokenizer.encode(PROMPT)
        cls.placeholder_id = tokenizer.encode(" Madrid", add_special_tokens=False)[0]
        cls.injected_id = tokenizer.encode(" Berlin", add_special_tokens=False)[0]
        embed_tokens = AutoModelForCausalLM.from_pretrained(
            cls.model, torch_dtype=torch.bfloat16
        ).get_input_embeddings()
        cls.injected_embedding = embed_tokens.weight[cls.injected_id].float().tolist()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _generate(self, payload):
        response = requests.post(f"{self.base_url}/generate", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()

    def test_embeds_inject_over_placeholder_ids(self):
        n = len(self.prompt_ids)
        # seed the radix prefix
        self._generate(
            {
                "input_ids": self.prompt_ids,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            }
        )

        # ids claim " Madrid" but the embeds encode " Berlin": the hidden
        # state must match a plain extend of " Berlin", proving the client
        # embeds actually replace the span's embedding lookup
        forward = self._generate(
            {
                "input_ids": self.prompt_ids + [self.placeholder_id],
                "input_embeds": [self.injected_embedding],
                "query_attention": "bidirectional",
                "token_positions": [n],
                "sampling_params": {"max_new_tokens": 0},
                "return_hidden_states": True,
            }
        )
        self.assertEqual(forward["meta_info"].get("cached_tokens", 0), n)
        hidden = torch.tensor(forward["meta_info"]["hidden_states"][-1][-1])

        reference = self._generate(
            {
                "input_ids": self.prompt_ids + [self.injected_id],
                "sampling_params": {"max_new_tokens": 0},
                "return_hidden_states": True,
            }
        )
        expected = torch.tensor(reference["meta_info"]["hidden_states"][-1][-1])

        cos = torch.nn.functional.cosine_similarity(hidden, expected, dim=0).item()
        self.assertGreater(cos, 0.999)

    def test_scratch_span_never_enters_the_prefix_cache(self):
        n = len(self.prompt_ids)
        payload = {
            "input_ids": self.prompt_ids + [self.placeholder_id],
            "input_embeds": [self.injected_embedding],
            "query_attention": "bidirectional",
            "token_positions": [n],
            "sampling_params": {"max_new_tokens": 0},
            "return_hidden_states": True,
        }
        self._generate(payload)
        repeat = self._generate(payload)
        # a cached span would show n+1 here
        self.assertEqual(repeat["meta_info"].get("cached_tokens", 0), n)

    def test_multi_dim_positions_round_trip(self):
        n = len(self.prompt_ids)
        forward = self._generate(
            {
                "input_ids": self.prompt_ids + [self.placeholder_id] * 2,
                "input_embeds": [self.injected_embedding] * 2,
                "query_attention": "bidirectional",
                "token_positions": [[n, n], [0, 1], [0, 1]],
                "sampling_params": {"max_new_tokens": 0},
                "return_hidden_states": True,
            }
        )
        span = forward["meta_info"]["hidden_states"][-1]
        self.assertGreaterEqual(len(span), 2)
        self.assertTrue(all(abs(v) < 1e4 for v in span[-1]))

    def test_invalid_inputs_are_rejected(self):
        n = len(self.prompt_ids)
        bad_mode = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": self.prompt_ids,
                "input_embeds": [self.injected_embedding],
                "query_attention": "diagonal",
                "sampling_params": {"max_new_tokens": 0},
            },
            timeout=120,
        )
        self.assertNotEqual(bad_mode.status_code, 200)

        oversized = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": [self.placeholder_id],
                "input_embeds": [self.injected_embedding] * 3,
                "query_attention": "bidirectional",
                "token_positions": [n],
                "sampling_params": {"max_new_tokens": 0},
            },
            timeout=120,
        )
        self.assertNotEqual(oversized.status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=3)
