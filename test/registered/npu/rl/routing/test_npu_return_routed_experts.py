"""Phase 2 NPU tests for DP-local routed-expert slicing (attention TP=1)."""

import json
import os
import threading
import time
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
ATTN_DP_SIZE = 2
MAX_NEW_TOKENS = 8
REQUEST_TIMEOUT = 300
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


def _server_args(tp_size, moe_a2a_backend):
    return (
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--tp-size",
        str(tp_size),
        "--attn-dp-size",
        str(ATTN_DP_SIZE),
        "--moe-a2a-backend",
        moe_a2a_backend,
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


def generate(base_url, text, **extra):
    payload = {
        "text": text,
        "sampling_params": dict(SAMPLING),
        "return_routed_experts": True,
    }
    payload.update(extra)
    body = post(base_url, "/generate", payload, timeout=REQUEST_TIMEOUT).json()
    if isinstance(body, dict) and "error" in body:
        raise AssertionError(f"/generate returned error: {body['error']}")
    return body


def responses(body):
    """Per-request responses of a /generate reply, single or batched."""
    return body if isinstance(body, list) else [body]


def seqlen(body):
    meta = body["meta_info"]
    return meta["prompt_tokens"] + meta["completion_tokens"]


class RoutedExpertsTestMixin:
    extra_server_args = ()

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = weights.ensure_checkpoint(CASE)
        config = _model_config(cls.checkpoint)
        cls.num_layers = int(config["num_hidden_layers"])
        cls.num_experts = int(config["n_routed_experts"])
        cls.topk = int(config["num_experts_per_tok"])
        cls.moe_start = _moe_layer_start(config, cls.num_layers)
        # TP collectives need their own socket range on shared hosts.
        port = 30000 + (os.getpid() * 7) % 20000
        env = {
            **npu_rl_env(),
            "HCCL_NPU_SOCKET_PORT_RANGE": os.environ.get(
                "HCCL_NPU_SOCKET_PORT_RANGE", f"{port}-{port + 99}"
            ),
        }
        cls.base_url = cls.enterClassContext(
            npu_server(
                cls.checkpoint,
                extra_args=(
                    *_server_args(cls.tp_size, cls.moe_a2a_backend),
                    *cls.extra_server_args,
                ),
                env=env,
            )
        )

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


@unittest.skipUnless(weights.case_available(CASE), f"model not found: {CASE.model}")
class TestNpuReturnRoutedExperts(RoutedExpertsTestMixin, CustomTestCase):
    tp_size = 2
    moe_a2a_backend = "none"

    def test_01_shape_and_rows(self):
        """Rows = seqlen-1; shape [rows, num_layers, topk]; ids valid and unique."""
        for prompt in PROMPTS:
            body = generate(self.base_url, prompt)
            experts = self._experts(body)
            self._check_shape_and_ids(experts)
            self.assertEqual(
                experts.shape[0],
                seqlen(body) - 1,
                f"{prompt!r}: row count != seqlen-1",
            )

    def test_02_start_len_semantics(self):
        """start_len=N crops rows to [N, seqlen-1) and matches the full tail."""
        body_full = generate(self.base_url, LONG_PROMPT)
        experts_full = self._experts(body_full)
        full_length = seqlen(body_full)
        prompt_tokens = body_full["meta_info"]["prompt_tokens"]

        # Omitted start_len must behave exactly like an explicit 0.
        body_zero = generate(self.base_url, LONG_PROMPT, routed_experts_start_len=0)
        self.assertTrue(
            np.array_equal(experts_full, self._experts(body_zero)),
            "default and explicit start_len=0 must be identical",
        )

        # start_len = k crops the leading rows; the rest is the full tail.
        k = prompt_tokens // 2
        body_k = generate(self.base_url, LONG_PROMPT, routed_experts_start_len=k)
        experts_k = self._experts(body_k)
        self.assertEqual(
            experts_k.shape[0], full_length - 1 - k, "start_len did not crop rows"
        )
        self.assertTrue(
            np.array_equal(experts_full[k:], experts_k),
            "cropped response is not the tail of the full response",
        )

        # start_len = prompt_tokens is the inclusive upper bound.
        body_prompt = generate(
            self.base_url, LONG_PROMPT, routed_experts_start_len=prompt_tokens
        )
        self.assertEqual(
            self._experts(body_prompt).shape[0],
            full_length - 1 - prompt_tokens,
            "start_len=prompt_tokens must crop to the generated tokens",
        )

    def test_03_batch_alignment_and_disabled_requests(self):
        """Every request of a batch gets its own rows; disabled requests get none.

        Also the negative path: a request that does not ask for routed experts
        must not receive the field, even when it races one that does.
        """
        batch = list(PROMPTS) + [LONG_PROMPT * 3]
        bodies = responses(generate(self.base_url, batch))
        self.assertEqual(len(bodies), len(batch), "batch reply lost a request")
        for prompt, body in zip(batch, bodies):
            experts = self._experts(body)
            self._check_shape_and_ids(experts)
            self.assertEqual(
                experts.shape[0],
                seqlen(body) - 1,
                f"rows of one request leaked across the batch for {prompt!r}",
            )

        # The same prompt twice in one greedy batch must route identically.
        twins = responses(generate(self.base_url, [PROMPTS[0], PROMPTS[0]]))
        self.assertTrue(
            np.array_equal(self._experts(twins[0]), self._experts(twins[1])),
            "identical prompts in one batch returned different routings",
        )

        results = {}
        errors = {}

        def _worker(key, return_flag):
            try:
                results[key] = generate(
                    self.base_url, PROMPTS[key], return_routed_experts=return_flag
                )
            except Exception as exc:
                errors[key] = exc

        threads = [
            threading.Thread(target=_worker, args=(0, False), daemon=True),
            threading.Thread(target=_worker, args=(1, True), daemon=True),
        ]
        deadline = time.monotonic() + REQUEST_TIMEOUT + 10
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        pending = [index for index, thread in enumerate(threads) if thread.is_alive()]
        errors = errors.copy()
        if errors:
            raise AssertionError(
                f"concurrent requests failed: {errors!r}; still pending: {pending}"
            ) from next(iter(errors.values()))
        self.assertFalse(pending, f"concurrent requests timed out: {pending}")

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
