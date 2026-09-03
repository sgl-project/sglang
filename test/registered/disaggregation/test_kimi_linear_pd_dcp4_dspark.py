import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import requests
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=500, stage="nightly", runner_config="8-gpu-b200")

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
PHYSICAL_PAGE_SIZE = 64
CHUNKED_PREFILL_SIZE = 8192


def _has_eight_blackwell_gpus() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        return False
    return all(
        torch.cuda.get_device_capability(device_index) >= (10, 0)
        for device_index in range(8)
    )


def _write_dummy_qwen3_dspark_draft(root: Path) -> str:
    draft_dir = root / "qwen3-dspark-kimi-proxy"
    draft_dir.mkdir()
    config = {
        "architectures": ["Qwen3DSparkModel"],
        "model_type": "qwen3",
        "dtype": "bfloat16",
        "hidden_size": 2304,
        "intermediate_size": 9216,
        "num_hidden_layers": 5,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "max_position_embeddings": 1048576,
        "rope_parameters": {
            "rope_theta": 10000.0,
            "rope_type": "default",
        },
        "vocab_size": 163840,
        "bos_token_id": 163584,
        "eos_token_id": 163586,
        "mask_token_id": 163839,
        "block_size": 7,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "num_target_layers": 27,
        "target_layer_ids": [1, 7, 13, 19, 26],
        "layer_types": ["full_attention"] * 5,
        "tie_word_embeddings": False,
        "use_cache": True,
    }
    (draft_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(draft_dir)


@unittest.skipUnless(
    _has_eight_blackwell_gpus(),
    "Kimi-Linear PD DCP4 + DSPARK requires eight Blackwell GPUs",
)
class TestKimiLinearPDDCP4DSpark(GSM8KMixin, PDDisaggregationServerBase):
    model = KIMI_LINEAR_MODEL
    gsm8k_score_threshold = 0.88
    gsm8k_num_examples = 400
    gsm8k_num_threads = 64
    gsm8k_num_shots = 5

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ["MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER"] = "65535"
        os.environ["MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER"] = "65535"

        cls._draft_root = tempfile.mkdtemp(prefix="dspark_pd_dcp_draft_")
        draft_path = _write_dummy_qwen3_dspark_draft(Path(cls._draft_root))
        dspark_args = [
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            draft_path,
            "--speculative-draft-load-format",
            "dummy",
            "--speculative-attention-mode",
            "decode",
            "--speculative-draft-attention-backend",
            "trtllm_mha",
        ]
        common_args = [
            "--attention-backend",
            "tokenspeed_mla",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--dtype",
            "bfloat16",
            "--random-seed",
            "0",
            "--page-size",
            str(PHYSICAL_PAGE_SIZE),
            "--cuda-graph-backend-prefill",
            "disabled",
            "--mem-fraction-static",
            "0.80",
        ] + dspark_args

        cls.prefill_tp_size = 4
        cls.decode_tp_size = 4
        cls.decode_base_gpu_id = 4
        cls.extra_prefill_args = common_args + [
            "--ep-size",
            "4",
            "--chunked-prefill-size",
            str(CHUNKED_PREFILL_SIZE),
        ]
        cls.extra_decode_args = common_args + [
            "--dcp-size",
            "4",
            "--dcp-comm-backend",
            "a2a",
            "--dcp-replicate-q-proj",
            "--cuda-graph-max-bs-decode",
            "64",
        ]
        cls.extra_prefill_env = {"SGLANG_RAGGED_VERIFY_MODE": "static"}
        cls.extra_decode_env = {"SGLANG_RAGGED_VERIFY_MODE": "static"}
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER", None)
        os.environ.pop("MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER", None)
        shutil.rmtree(cls._draft_root, ignore_errors=True)
        super().tearDownClass()

    def test_spec_verify_runs_on_decode(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "ignore_eos": True,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        meta_info = response.json()["meta_info"]
        self.assertGreater(
            meta_info.get("spec_verify_ct", 0),
            0,
            "DSPARK verify did not run on the decode side",
        )
        self.assertGreater(meta_info["completion_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
