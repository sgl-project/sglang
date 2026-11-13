"""NPU tests: --enable-return-routed-experts end-to-end on GLM-5.2 (MoE, TP=2/DP=2).

Phase 2 of https://github.com/sgl-project/sglang/issues/48822. An RL trainer
consumes the routed experts of every generated token, so the capture pipeline
(device buffer, DP-rank slicing, the attention-TP all-gather on scattered
a2a backends, host readback and start_len cropping) must return exactly the
rows of exactly the tokens of exactly the requesting request.

The suite runs a layer-reduced GLM-5.2 with dummy weights (structure-only
model, same recipe as the NPU indexer-topk suite): routed experts are routing
*indices*, so real weights are not required for shape/range/alignment checks.
The reduced model keeps 3 dense + 3 MoE layers, so assertions on expert ids
cover only the MoE layers' columns.

[Test Category] RL routing capture
[Test Target] --enable-return-routed-experts, return_routed_experts,
               routed_experts_start_len
"""

import json
import os
import threading
import unittest

import numpy as np
import requests

from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import GLM_5_2_BF16_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=420, suite="nightly-2-npu-a3", nightly=True)
register_npu_ci(est_time=420, suite="base-b-test-2-npu-a3")

MODEL_PATH = GLM_5_2_BF16_MODEL_PATH
MODEL_OVERRIDES = {"num_hidden_layers": 6, "quantization_config": None}
NUM_LAYERS = MODEL_OVERRIDES["num_hidden_layers"]
MAX_NEW_TOKENS = 8

PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
)
LONG_PROMPT = "Solve: 17 * 24 = "


def _model_config():
    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        return json.load(f)


def _moe_layer_start(config):
    """Index of the first MoE layer of the reduced model.

    GLM-5.2 marks its first layers dense via ``mlp_layer_types`` (matching
    ``first_k_dense_replace``). Dense layers never route, so their columns in
    the returned tensor are never written by the capturer; expert-id checks
    start at the first non-dense layer.
    """
    types = config.get("mlp_layer_types")
    if not types:
        return int(config.get("first_k_dense_replace", 0))
    dense = sum(1 for t in types[:NUM_LAYERS] if t == "dense")
    return dense


@unittest.skipUnless(
    os.path.isfile(os.path.join(MODEL_PATH, "config.json")),
    f"GLM-5.2 model config not found at {MODEL_PATH}",
)
class TestNpuReturnRoutedExperts(CustomTestCase):
    """Routed-experts capture under TP=2 / attn-DP=2 on NPU."""

    @classmethod
    def setUpClass(cls):
        cls.config = _model_config()
        cls.num_experts = int(cls.config["n_routed_experts"])
        cls.topk = int(cls.config["num_experts_per_tok"])
        cls.moe_start = _moe_layer_start(cls.config)
        cls.sampling_args = {
            "temperature": 0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "ignore_eos": True,
        }
        cls.process = popen_launch_server(
            MODEL_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={
                **os.environ,
                "PYTORCH_NPU_ALLOC_CONF": (
                    "expandable_segments:True,pin_memory_expandable_segments:True"
                ),
            },
            other_args=[
                "--device",
                "npu",
                "--attention-backend",
                "ascend",
                "--load-format",
                "dummy",
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.56",
                "--tp-size",
                "2",
                "--attn-dp-size",
                "2",
                "--enable-dp-lm-head",
                "--disable-cuda-graph",
                "--enable-return-routed-experts",
                "--json-model-override-args",
                json.dumps(MODEL_OVERRIDES),
            ],
        )
        cls.base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _generate(self, text, **extra):
        payload = {
            "text": text,
            "sampling_params": self.sampling_args,
            "return_routed_experts": True,
        }
        payload.update(extra)
        response = requests.post(
            self.base_url + "/generate", json=payload, timeout=120
        )
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            raise AssertionError(f"generate returned error: {body['error']}")
        return body

    def _experts(self, body):
        experts = extract_routed_experts_from_meta_info(body)
        return experts.reshape(-1, NUM_LAYERS, self.topk)

    def _seqlen(self, body):
        meta = body["meta_info"]
        return meta["prompt_tokens"] + meta["completion_tokens"]

    def _check_shape_and_ids(self, experts):
        self.assertEqual(experts.ndim, 3)
        rows, num_layers, topk = experts.shape
        self.assertEqual(num_layers, NUM_LAYERS)
        self.assertEqual(topk, self.topk)
        moe = experts[:, self.moe_start :, :]
        self.assertTrue(
            ((moe >= 0) & (moe < self.num_experts)).all(),
            f"expert ids out of [0, {self.num_experts}): "
            f"min={moe.min()} max={moe.max()}",
        )
        for row in moe:
            for layer_ids in row:
                self.assertEqual(
                    len(layer_ids), len(set(layer_ids.tolist())),
                    f"duplicate expert ids in one token: {layer_ids}",
                )

    # ------------------------------------------------------------------
    # cases
    # ------------------------------------------------------------------

    def test_01_shape_and_rows(self):
        """Rows = seqlen-1; shape [rows, num_layers, topk]; ids valid + unique."""
        for prompt in PROMPTS:
            body = self._generate(prompt)
            experts = self._experts(body)
            self._check_shape_and_ids(experts)
            self.assertEqual(
                experts.shape[0],
                self._seqlen(body) - 1,
                f"{prompt!r}: row count != seqlen-1",
            )

    def test_02_start_len_semantics(self):
        """start_len=N crops rows to [N, seqlen-1) and matches the full tail."""
        body_full = self._generate(LONG_PROMPT)
        experts_full = self._experts(body_full)
        seqlen = self._seqlen(body_full)
        prompt_tokens = body_full["meta_info"]["prompt_tokens"]

        # default (omitted) == explicit 0
        body_zero = self._generate(LONG_PROMPT, routed_experts_start_len=0)
        self.assertTrue(
            np.array_equal(experts_full, self._experts(body_zero)),
            "default and explicit start_len=0 must be identical",
        )

        # start_len = k: rows = seqlen-1-k and tail matches the full response
        k = prompt_tokens // 2
        body_k = self._generate(LONG_PROMPT, routed_experts_start_len=k)
        experts_k = self._experts(body_k)
        self.assertEqual(
            experts_k.shape[0], seqlen - 1 - k, "start_len did not crop rows"
        )
        self.assertTrue(
            np.array_equal(experts_full[k:], experts_k),
            "cropped response is not the tail of the full response",
        )

        # start_len = prompt_tokens is the inclusive upper bound
        body_pt = self._generate(LONG_PROMPT, routed_experts_start_len=prompt_tokens)
        self.assertEqual(
            self._experts(body_pt).shape[0],
            seqlen - 1 - prompt_tokens,
            "start_len=prompt_tokens must crop to the generated tokens",
        )

    def test_03_batch_alignment_and_disabled_requests(self):
        """Each request of a batch gets its own rows; return_routed_experts=False
        requests must not carry the field at all."""
        # Mixed batch: different-length prompts, every row belongs to its own
        # request (DP slicing must not shift rows across requests).
        batch = list(PROMPTS) + [LONG_PROMPT * 3]
        body = self._generate(batch)
        per_request = body if isinstance(body, list) else body["data"] if "data" in body else [body]
        bodies = per_request if isinstance(per_request, list) else [per_request]
        self.assertEqual(len(bodies), len(batch))
        for prompt, item in zip(batch, bodies):
            item = item[0] if isinstance(item, list) else item
            experts = self._experts(item)
            self._check_shape_and_ids(experts)
            self.assertEqual(
                experts.shape[0], self._seqlen(item) - 1,
                f"rows of one request leaked across the batch for {prompt!r}",
            )

        # Same prompt twice in one batch: greedy decoding of a fixed payload
        # must give identical routings for both copies.
        body_twice = self._generate([PROMPTS[0], PROMPTS[0]])
        twins = body_twice if isinstance(body_twice, list) else [body_twice]
        t0 = twins[0][0] if isinstance(twins[0], list) else twins[0]
        t1 = twins[1][0] if isinstance(twins[1], list) else twins[1]
        self.assertTrue(
            np.array_equal(self._experts(t0), self._experts(t1)),
            "identical prompts in one batch returned different routings",
        )

        # Negative path: requests that do not ask for routed experts must not
        # receive the field, even when sent concurrently with one that does.
        results = {}

        def _worker(key, return_flag):
            payload = {
                "text": PROMPTS[key],
                "sampling_params": self.sampling_args,
                "return_routed_experts": return_flag,
            }
            response = requests.post(
                self.base_url + "/generate", json=payload, timeout=120
            )
            response.raise_for_status()
            results[key] = response.json()

        threads = [
            threading.Thread(target=_worker, args=(0, False)),
            threading.Thread(target=_worker, args=(1, True)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertNotIn(
            "routed_experts", results[0].get("meta_info", {}),
            "return_routed_experts=False must not return the field",
        )
        self.assertIn(
            "routed_experts", results[1].get("meta_info", {}),
            "return_routed_experts=True must return the field",
        )


if __name__ == "__main__":
    unittest.main()
