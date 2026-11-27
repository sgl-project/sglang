"""Shared fixture for Ascend routed-expert readback tests."""

import json
import os

import numpy as np

from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)
from sglang.test.ascend import test_npu_rl_weights as weights
from sglang.test.ascend.test_npu_rl_utils import npu_rl_env, npu_server, post

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
