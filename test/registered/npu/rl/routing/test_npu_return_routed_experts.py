"""Phase 2 NPU tests: --enable-return-routed-experts end to end (TP=2 / attn-DP=2).

An RL trainer consumes the routed experts of every generated token, so the
capture pipeline (device buffer, DP-rank slicing, the attention-TP all-gather on
scattered a2a backends, host readback and ``routed_experts_start_len`` cropping)
must return exactly the rows of exactly the tokens of exactly the requesting
request.

The suite reuses the reduced GLM-5.2 payload of the phase-1 weight-transfer
suites (``test_npu_rl_weights.GLM52_CASE``): routed experts are routing
*indices*, so the generated weights only have to load. The reduced model keeps
three dense layers followed by three MoE layers, so expert-id assertions cover
only the MoE columns.
"""

import json
import os
import threading
import unittest

import numpy as np

from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)
from sglang.test.ascend import test_npu_rl_weights as weights
from sglang.test.ascend.test_npu_rl_utils import npu_rl_env, npu_server, post
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=600, suite="extra-b-test-2-npu-a3")
register_npu_ci(est_time=600, suite="nightly-test-2-npu-a3", nightly=True)

CASE = weights.GLM52_CASE
TP_SIZE = 2
MAX_NEW_TOKENS = 8
# TP collectives need their own socket range on shared hosts.
HCCL_PORT_RANGE = os.environ.setdefault(
    "HCCL_NPU_SOCKET_PORT_RANGE",
    f"{30000 + (os.getpid() * 7) % 20000}-{30000 + (os.getpid() * 7) % 20000 + 99}",
)
PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
)
LONG_PROMPT = "Solve: 17 * 24 = "
SAMPLING = {
    "temperature": 0,
    "max_new_tokens": MAX_NEW_TOKENS,
    "ignore_eos": True,
}


def _server_args():
    return (
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--tp-size",
        str(TP_SIZE),
        "--attn-dp-size",
        str(TP_SIZE),
        "--enable-dp-lm-head",
        "--enable-return-routed-experts",
        "--context-length",
        "2048",
        "--max-running-requests",
        "8",
        "--mem-fraction-static",
        "0.5",
    )


def _model_config(model_path):
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _moe_layer_start(config, num_layers):
    """Index of the first MoE layer of the reduced model.

    GLM-5.2 marks its first layers dense via ``mlp_layer_types`` (matching
    ``first_k_dense_replace``). Dense layers never route, so their columns in
    the returned tensor are never written by the capturer; expert-id checks
    start at the first non-dense layer.
    """
    types = config.get("mlp_layer_types")
    if not types:
        return int(config.get("first_k_dense_replace", 0))
    return sum(1 for layer_type in types[:num_layers] if layer_type == "dense")


def _generate(base_url, text, **extra):
    payload = {
        "text": text,
        "sampling_params": dict(SAMPLING),
        "return_routed_experts": True,
    }
    payload.update(extra)
    body = post(base_url, "/generate", payload, timeout=300).json()
    if isinstance(body, dict) and "error" in body:
        raise AssertionError(f"/generate returned error: {body['error']}")
    return body


def _responses(body):
    """Per-request responses of a /generate reply, single or batched."""
    return body if isinstance(body, list) else [body]


def _seqlen(body):
    meta = body["meta_info"]
    return meta["prompt_tokens"] + meta["completion_tokens"]


@unittest.skipUnless(weights.case_available(CASE), f"model not found: {CASE.model}")
class TestNpuReturnRoutedExperts(CustomTestCase):
    """Routed-experts capture under TP=2 / attn-DP=2."""

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = weights.ensure_checkpoint(CASE)
        config = _model_config(cls.checkpoint)
        cls.num_layers = int(config["num_hidden_layers"])
        cls.num_experts = int(config["n_routed_experts"])
        cls.topk = int(config["num_experts_per_tok"])
        cls.moe_start = _moe_layer_start(config, cls.num_layers)
        env = {**npu_rl_env(), "HCCL_NPU_SOCKET_PORT_RANGE": HCCL_PORT_RANGE}
        cls._server = npu_server(cls.checkpoint, extra_args=_server_args(), env=env)
        cls.base_url = cls._server.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)

    def _experts(self, body):
        experts = extract_routed_experts_from_meta_info(body)
        return np.asarray(experts).reshape(-1, self.num_layers, self.topk)

    def _check_shape_and_ids(self, experts):
        self.assertEqual(experts.ndim, 3)
        _, num_layers, topk = experts.shape
        self.assertEqual(num_layers, self.num_layers, "unexpected layer count")
        self.assertEqual(topk, self.topk, "unexpected top-k width")
        moe = experts[:, self.moe_start :, :]
        self.assertTrue(
            ((moe >= 0) & (moe < self.num_experts)).all(),
            f"expert ids out of [0, {self.num_experts}): "
            f"min={moe.min()} max={moe.max()}",
        )
        for row in moe:
            for layer_ids in row:
                self.assertEqual(
                    len(layer_ids),
                    len(set(layer_ids.tolist())),
                    f"duplicate expert ids in one token: {layer_ids}",
                )

    def test_01_shape_and_rows(self):
        """Rows = seqlen-1; shape [rows, num_layers, topk]; ids valid and unique."""
        for prompt in PROMPTS:
            body = _generate(self.base_url, prompt)
            experts = self._experts(body)
            self._check_shape_and_ids(experts)
            self.assertEqual(
                experts.shape[0],
                _seqlen(body) - 1,
                f"{prompt!r}: row count != seqlen-1",
            )

    def test_02_start_len_semantics(self):
        """start_len=N crops rows to [N, seqlen-1) and matches the full tail."""
        body_full = _generate(self.base_url, LONG_PROMPT)
        experts_full = self._experts(body_full)
        seqlen = _seqlen(body_full)
        prompt_tokens = body_full["meta_info"]["prompt_tokens"]

        # Omitted start_len must behave exactly like an explicit 0.
        body_zero = _generate(self.base_url, LONG_PROMPT, routed_experts_start_len=0)
        self.assertTrue(
            np.array_equal(experts_full, self._experts(body_zero)),
            "default and explicit start_len=0 must be identical",
        )

        # start_len = k crops the leading rows; the rest is the full tail.
        k = prompt_tokens // 2
        body_k = _generate(self.base_url, LONG_PROMPT, routed_experts_start_len=k)
        experts_k = self._experts(body_k)
        self.assertEqual(
            experts_k.shape[0], seqlen - 1 - k, "start_len did not crop rows"
        )
        self.assertTrue(
            np.array_equal(experts_full[k:], experts_k),
            "cropped response is not the tail of the full response",
        )

        # start_len = prompt_tokens is the inclusive upper bound.
        body_prompt = _generate(
            self.base_url, LONG_PROMPT, routed_experts_start_len=prompt_tokens
        )
        self.assertEqual(
            self._experts(body_prompt).shape[0],
            seqlen - 1 - prompt_tokens,
            "start_len=prompt_tokens must crop to the generated tokens",
        )

    def test_03_batch_alignment_and_disabled_requests(self):
        """Every request of a batch gets its own rows; disabled requests get none.

        Also the negative path: a request that does not ask for routed experts
        must not receive the field, even when it races one that does.
        """
        batch = list(PROMPTS) + [LONG_PROMPT * 3]
        bodies = _responses(_generate(self.base_url, batch))
        self.assertEqual(len(bodies), len(batch), "batch reply lost a request")
        for prompt, body in zip(batch, bodies):
            experts = self._experts(body)
            self._check_shape_and_ids(experts)
            self.assertEqual(
                experts.shape[0],
                _seqlen(body) - 1,
                f"rows of one request leaked across the batch for {prompt!r}",
            )

        # The same prompt twice in one greedy batch must route identically.
        twins = _responses(_generate(self.base_url, [PROMPTS[0], PROMPTS[0]]))
        self.assertTrue(
            np.array_equal(self._experts(twins[0]), self._experts(twins[1])),
            "identical prompts in one batch returned different routings",
        )

        results = {}

        def _worker(key, return_flag):
            results[key] = _generate(
                self.base_url, PROMPTS[key], return_routed_experts=return_flag
            )

        threads = [
            threading.Thread(target=_worker, args=(0, False)),
            threading.Thread(target=_worker, args=(1, True)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertNotIn(
            "routed_experts",
            results[0].get("meta_info", {}),
            "return_routed_experts=False must not return the field",
        )
        self.assertIn(
            "routed_experts",
            results[1].get("meta_info", {}),
            "return_routed_experts=True must return the field",
        )


if __name__ == "__main__":
    unittest.main()
