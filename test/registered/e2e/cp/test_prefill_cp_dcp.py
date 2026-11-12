"""Exercise FA3 (SM90) or FA4 (SM100) prefill with FlashInfer MLA decode.

By default the test builds a tiny local model and scores a prompt on one GPU
before starting the distributed server. Set SGLANG_TEST_PCP_DCP_MODEL to "dummy"
for config-only smoke testing, or to a DeepSeek dense MLA checkpoint path or
model ID to test pretrained weights. These modes skip the single-GPU reference.
With eight GPUs the test uses TP8/PCP4/DCP4 and the tiny model includes a MoE layer.
Four-GPU runs default to interleave, other runs to zigzag. Set
SGLANG_TEST_PCP_DCP_STRATEGY to override the strategy.
"""

import math
import os
import tempfile
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(est_time=240, stage="extra-a", runner_config="2-gpu-large")
register_cuda_ci(est_time=240, stage="extra-b", runner_config="4-gpu-h100")
register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=360, stage="extra-b", runner_config="8-gpu-h200")
register_cuda_ci(est_time=360, stage="extra-b", runner_config="8-gpu-b300")

LONG_PROMPT = [3 + position % 127 for position in range(257)]


def _write_tiny_mla_model(model_dir, *, dummy, with_moe):
    from transformers import DeepseekV3Config

    config = DeepseekV3Config(
        architectures=["DeepseekV3ForCausalLM"],
        dtype="bfloat16",
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        first_k_dense_replace=1 if with_moe else 3,
        moe_layer_freq=1,
        scoring_func="sigmoid",
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=128,
        num_attention_heads=16,
        num_key_value_heads=16,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        max_position_embeddings=1024,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    if dummy:
        config.save_pretrained(model_dir)
        return

    from transformers import DeepseekV3ForCausalLM

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        model = DeepseekV3ForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(model_dir)


class TestPrefillCpDcp(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.parallel_size = torch.cuda.device_count()
        if cls.parallel_size not in (2, 4, 8):
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly 2, 4 or 8 GPUs")
        cls.cp_size = 4 if cls.parallel_size == 8 else cls.parallel_size
        cls.cp_strategy = os.environ.get(
            "SGLANG_TEST_PCP_DCP_STRATEGY",
            "interleave" if cls.parallel_size == 4 else "zigzag",
        )
        with_moe = cls.parallel_size == 8
        majors = {
            torch.cuda.get_device_capability(i)[0] for i in range(cls.parallel_size)
        }
        if torch.version.cuda is None or majors not in ({9}, {10}):
            raise RuntimeError("Use GPUs that are all SM90 or all SM100")
        cls.prefill_backend = "fa4" if majors == {10} else "fa3"
        model = os.environ.get("SGLANG_TEST_PCP_DCP_MODEL")
        cls.dummy = model == "dummy"
        if model and not cls.dummy:
            cls.model = try_cached_model(model)
        else:
            model_dir = tempfile.TemporaryDirectory(prefix="pcp-dcp-mla-")
            cls.addClassCleanup(model_dir.cleanup)
            _write_tiny_mla_model(model_dir.name, dummy=cls.dummy, with_moe=with_moe)
            cls.model = model_dir.name
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.reference_logprobs = None
        if not model:
            reference = cls._launch_server(
                [
                    "--attention-backend",
                    "flashinfer",
                    "--chunked-prefill-size",
                    "-1",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    "--cuda-graph-backend-decode",
                    "disabled",
                ]
            )
            try:
                cls.reference_logprobs = cls._generate(
                    LONG_PROMPT, max_new_tokens=0, logprob_start_len=0
                )["input_token_logprobs"][1:]
            finally:
                terminate_and_kill_process_tree(reference)

        cls.process = cls._launch_server(
            [
                "--tp-size",
                str(cls.parallel_size),
                "--attn-cp-size",
                str(cls.cp_size),
                "--dcp-size",
                str(cls.cp_size),
                "--enable-prefill-cp",
                "--cp-strategy",
                cls.cp_strategy,
                "--prefill-attention-backend",
                cls.prefill_backend,
                "--decode-attention-backend",
                "flashinfer",
                "--chunked-prefill-size",
                # The second interleaved chunk starts on an odd position.
                "127" if cls.cp_strategy == "interleave" else "128",
                # Exercise dispatch/combine for the tiny MoE's hidden size.
                *(
                    ["--deepep-mode", "normal"]
                    if with_moe and (not model or cls.dummy)
                    else []
                ),
            ]
        )

    @classmethod
    def _launch_server(cls, other_args):
        return popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--skip-tokenizer-init",
                "--load-format",
                "dummy" if cls.dummy else "auto",
                "--kv-cache-dtype",
                "bf16",
                "--context-length",
                "1024",
                "--max-total-tokens",
                "2048",
                "--max-running-requests",
                "8",
                "--enable-cache-report",
                *other_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)
        super().tearDownClass()

    @classmethod
    def _generate(cls, ids, *, max_new_tokens=8, logprob_start_len=-1):
        response = requests.post(
            cls.base_url + "/generate",
            json={
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": logprob_start_len,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["meta_info"]

    def _assert_logprobs_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for (actual_lp, actual_id, *_), (expected_lp, expected_id, *_) in zip(
            actual, expected, strict=True
        ):
            self.assertEqual(actual_id, expected_id)
            self.assertTrue(math.isfinite(actual_lp) and math.isfinite(expected_lp))
            self.assertAlmostEqual(actual_lp, expected_lp, delta=0.10)

    def _assert_decode_matches_prefill(self, prompt, output):
        self.assertEqual(len(output), 8)
        output_ids = [row[1] for row in output]
        scores = self._generate(
            prompt + output_ids, max_new_tokens=0, logprob_start_len=0
        )["input_token_logprobs"]
        self._assert_logprobs_close(output, scores[len(prompt) :])
        return scores

    def test_short_prefill_fallback(self):
        prompt = [17]
        # The one-token prefill is too short for CP. Every decode step
        # still has to use the sharded layout established by that fallback.
        output = self._generate(prompt)["output_token_logprobs"]
        self._assert_decode_matches_prefill(prompt, output)

    def test_server_uses_requested_topology(self):
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        info = response.json()
        self.assertEqual(info["tp_size"], self.parallel_size)
        for field in ("attn_cp_size", "dcp_size"):
            self.assertEqual(info[field], self.cp_size, field)
        self.assertTrue(info["enable_prefill_cp"])
        self.assertEqual(info["cp_strategy"], self.cp_strategy)
        self.assertEqual(info["dcp_comm_backend"], "ag_ar")
        self.assertEqual(info["prefill_attention_backend"], self.prefill_backend)
        self.assertEqual(info["decode_attention_backend"], "flashinfer")

    def test_chunked_prefill_and_radix_hit(self):
        # Two full chunks and a short tail. Under interleave, the second
        # chunk's Q and persistent KV are assigned to different ranks.
        cold = self._generate(LONG_PROMPT)
        warm = self._generate(LONG_PROMPT)
        self.assertGreaterEqual(warm["cached_tokens"], 128)
        self._assert_logprobs_close(
            cold["output_token_logprobs"], warm["output_token_logprobs"]
        )
        scores = self._assert_decode_matches_prefill(
            LONG_PROMPT, cold["output_token_logprobs"]
        )
        if self.reference_logprobs is not None:
            self._assert_logprobs_close(
                scores[1 : len(LONG_PROMPT)], self.reference_logprobs
            )


if __name__ == "__main__":
    unittest.main()
