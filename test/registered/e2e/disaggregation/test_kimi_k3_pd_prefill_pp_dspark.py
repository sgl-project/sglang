import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=300, stage="nightly", runner_config="4-gpu-b200")

MODEL_CONFIGS = {
    "target": {
        "architectures": ["KimiK3LinearForCausalLM"],
        "model_type": "kimi_linear",
        "dtype": "bfloat16",
        "vocab_size": 256,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-06,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": False,
        "attn_res_block_size": 2,
        "q_lora_rank": 256,
        "kv_lora_rank": 128,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_output_gate": True,
        "linear_attn_config": {
            "kda_layers": [1, 2, 3, 5, 6, 7],
            "full_attn_layers": [4, 8],
            "num_heads": 4,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        },
    },
    "draft": {
        "architectures": ["Qwen3DSparkModel"],
        "model_type": "qwen3",
        "dtype": "bfloat16",
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-06,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "max_position_embeddings": 2048,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "vocab_size": 256,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "mask_token_id": 255,
        "block_size": 3,
        "markov_rank": 16,
        "markov_head_type": "vanilla",
        "enable_confidence_head": False,
        "num_target_layers": 8,
        "target_layer_ids": [1, 3, 6],
        "layer_types": ["full_attention", "full_attention"],
        "tie_word_embeddings": False,
        "use_cache": True,
    },
}


@unittest.skipUnless(torch.cuda.device_count() >= 3, "Requires three CUDA GPUs")
class TestKimiK3PDPrefillPPDSpark(PDDisaggregationServerBase):
    @classmethod
    def rdma_devices_for(cls, gpu_indices):
        return []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._model_dir = tempfile.TemporaryDirectory(prefix="kimi_k3_dspark_pp_")
        root = Path(cls._model_dir.name)
        for name, config in MODEL_CONFIGS.items():
            (root / name).mkdir()
            (root / name / "config.json").write_text(json.dumps(config))
        cls.model = str(root / "target")
        cls.prefill_tp_size = 1
        cls.decode_tp_size = 1
        cls.decode_base_gpu_id = 2
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
        common_args = [
            "--load-format",
            "dummy",
            "--skip-tokenizer-init",
            "--dtype",
            "bfloat16",
            "--attention-backend",
            "triton",
            "--linear-attn-decode-backend",
            "triton",
            "--linear-attn-verify-backend",
            "triton",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            str(root / "draft"),
            "--speculative-draft-load-format",
            "dummy",
            "--speculative-draft-attention-backend",
            "triton",
            "--speculative-dspark-block-size",
            "3",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--max-total-tokens",
            "1024",
            "--max-running-requests",
            "4",
            "--max-mamba-cache-size",
            "16",
            "--context-length",
            "512",
            "--chunked-prefill-size",
            "64",
            "--mem-fraction-static",
            "0.80",
            "--disable-radix-cache",
            "--random-seed",
            "42",
        ]
        cls.extra_prefill_args = common_args + ["--pp-size", "2"]
        cls.extra_decode_args = common_args + ["--enable-linear-replayssm-spec"]
        env = {
            "SGLANG_RAGGED_VERIFY_MODE": "static",
            "MOONCAKE_PROTOCOL": "tcp",
            "MC_FORCE_TCP": "1",
        }
        cls.extra_prefill_env = dict(env)
        cls.extra_decode_env = dict(env)
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            if hasattr(cls, "_model_dir"):
                cls._model_dir.cleanup()

    def _generate(self, length):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": [10 + i % 40 for i in range(length)],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                    "ignore_eos": True,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
        print(json.dumps(result), flush=True)
        self.assertEqual(result["meta_info"]["completion_tokens"], 16)
        self.assertGreater(result["meta_info"].get("spec_verify_ct", 0), 0)
        return result

    def test_chunked_prefill_and_request_reuse(self):
        for length in [24, 160, 24]:
            with self.subTest(length=length):
                self._generate(length)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(self._generate, [80, 120]))


if __name__ == "__main__":
    unittest.main()
