"""Verify routed-expert gathering across attention TP ranks on Ascend."""

import json
import os
import unittest

import numpy as np
from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)
from sglang.test.ascend import test_npu_rl_weights as weights
from sglang.test.ascend.test_npu_rl_utils import npu_rl_env, npu_server, post
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=600, suite="extra-b-test-4-npu-a3")
register_npu_ci(est_time=600, suite="nightly-test-4-npu-a3", nightly=True)

CASE = weights.GLM52_CASE
ATTN_DP_SIZE = 2


@unittest.skipUnless(weights.case_available(CASE), f"model not found: {CASE.model}")
class TestNpuRoutedExpertsAttentionTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = weights.ensure_checkpoint(CASE)
        with open(os.path.join(checkpoint, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        cls.num_layers = int(config["num_hidden_layers"])
        cls.num_experts = int(config["n_routed_experts"])
        cls.topk = int(config["num_experts_per_tok"])
        layer_types = config.get("mlp_layer_types")
        cls.moe_start = (
            layer_types[: cls.num_layers].count("dense")
            if layer_types
            else int(config.get("first_k_dense_replace", 0))
        )
        port = 30000 + (os.getpid() * 7) % 20000
        env = {
            **npu_rl_env(),
            "HCCL_NPU_SOCKET_PORT_RANGE": os.environ.get(
                "HCCL_NPU_SOCKET_PORT_RANGE", f"{port}-{port + 99}"
            ),
        }
        cls.base_url = cls.enterClassContext(
            npu_server(
                checkpoint,
                extra_args=(
                    "--trust-remote-code",
                    "--attention-backend",
                    "ascend",
                    "--tp-size",
                    "4",
                    "--attn-dp-size",
                    str(ATTN_DP_SIZE),
                    # DeepEP scatters tokens across attention TP ranks.
                    "--moe-a2a-backend",
                    "deepep",
                    "--deepep-mode",
                    "low_latency",
                    "--disable-radix-cache",
                    "--enable-dp-lm-head",
                    "--enable-return-routed-experts",
                    "--context-length",
                    "2048",
                    "--max-running-requests",
                    "8",
                    "--mem-fraction-static",
                    "0.5",
                ),
                env=env,
            )
        )

    def _generate(self, prompt, dp_rank, **extra):
        body = post(
            self.base_url,
            "/generate",
            {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
                "return_routed_experts": True,
                "routed_dp_rank": dp_rank,
                **extra,
            },
            timeout=300,
        ).json()
        if "error" in body:
            self.fail(f"/generate returned error: {body['error']}")
        return body

    def _moe_experts(self, body):
        experts = extract_routed_experts_from_meta_info(body)
        return experts.reshape(-1, self.num_layers, self.topk)[:, self.moe_start :, :]

    def test_gathered_experts_preserve_request_rows(self):
        start_len = 2
        for dp_rank in range(ATTN_DP_SIZE):
            with self.subTest(dp_rank=dp_rank):
                prompt = "Solve: 17 * 24 = " * (dp_rank + 3)
                full = self._generate(prompt, dp_rank)
                cropped = self._generate(
                    prompt, dp_rank, routed_experts_start_len=start_len
                )
                experts = self._moe_experts(full)
                meta = full["meta_info"]
                self.assertEqual(
                    experts.shape[0],
                    meta["prompt_tokens"] + meta["completion_tokens"] - 1,
                )
                self.assertGreater(experts.size, 0)
                self.assertTrue(((experts >= 0) & (experts < self.num_experts)).all())
                self.assertTrue((np.diff(np.sort(experts, axis=-1), axis=-1) > 0).all())
                self.assertEqual(full["output_ids"], cropped["output_ids"])
                np.testing.assert_array_equal(
                    self._moe_experts(cropped),
                    experts[start_len:],
                    err_msg=f"incorrect routed-expert crop on DP rank {dp_rank}",
                )


if __name__ == "__main__":
    unittest.main()
