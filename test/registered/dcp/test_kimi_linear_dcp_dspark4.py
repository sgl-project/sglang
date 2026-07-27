"""Four-Blackwell Kimi Linear TokenSpeed DCP + DSpark static acceptance test."""

import json
import socket
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=1800, stage="base-c", runner_config="4-gpu-b200")

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
GSM8K_SCORE_THRESHOLD = 0.90


def _has_four_blackwell_gpus() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        return False
    return all(
        torch.cuda.get_device_capability(device_index) >= (10, 0)
        for device_index in range(4)
    )


def _write_dummy_qwen3_dspark_draft(root: Path) -> str:
    """Write a dummy Qwen3 DSpark config with Kimi Linear dimensions."""
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


def _wait_for_port_release(base_url: str, timeout: float = 30.0) -> None:
    parsed = urlparse(base_url)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((parsed.hostname, parsed.port)) != 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"Server port was not released after {timeout}s: {base_url}")


@unittest.skipUnless(
    _has_four_blackwell_gpus(),
    "Kimi Linear TokenSpeed DCP + DSpark requires four Blackwell GPUs",
)
class TestKimiLinearDCPDSpark4(CustomTestCase):
    base_url = DEFAULT_URL_FOR_TEST

    def _generate(self, prompts: list[str], *, max_new_tokens: int):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        outputs = response.json()
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), len(prompts))
        for output in outputs:
            self.assertTrue(output["text"].strip())
            self.assertGreater(output["meta_info"]["completion_tokens"], 0)
        return outputs

    def _run_static(self, *, draft_path: str, qrep: bool):
        other_args = [
            "--tp-size",
            "4",
            "--dcp-size",
            "4",
            "--attention-backend",
            "tokenspeed_mla",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--dcp-comm-backend",
            "a2a",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            draft_path,
            "--speculative-draft-load-format",
            "dummy",
            "--speculative-draft-attention-backend",
            "trtllm_mha",
            "--cuda-graph-max-bs-decode",
            "16",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.80",
        ]
        if qrep:
            other_args.append("--dcp-replicate-q-proj")

        process = popen_launch_server(
            KIMI_LINEAR_MODEL,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 8,
            other_args=other_args,
            env={
                "SGLANG_PREP_IN_CUDA_GRAPH": "1",
                "SGLANG_RAGGED_VERIFY_MODE": "static",
            },
        )
        try:
            captured_outputs = self._generate(
                [
                    f"Reply with one short word for captured request {index}: the sky is"
                    for index in range(2)
                ],
                max_new_tokens=8,
            )
            max_graph_outputs = self._generate(
                [
                    f"Reply with one short word for graph request {index}: ice is"
                    for index in range(16)
                ],
                max_new_tokens=8,
            )
            requests.get(self.base_url + "/flush_cache", timeout=30).raise_for_status()
            metrics = run_eval(
                SimpleNamespace(
                    base_url=self.base_url,
                    model=KIMI_LINEAR_MODEL,
                    eval_name="gsm8k",
                    api="completion",
                    max_tokens=512,
                    num_examples=200,
                    num_threads=4,
                    num_shots=5,
                )
            )
            return captured_outputs + max_graph_outputs, float(metrics["score"])
        finally:
            kill_process_tree(process.pid, wait_timeout=60)
            _wait_for_port_release(self.base_url)

    def test_static_verify_cuda_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_path = _write_dummy_qwen3_dspark_draft(root)
            for qrep in (True, False):
                with self.subTest(qrep=qrep):
                    outputs, score = self._run_static(draft_path=draft_path, qrep=qrep)
                    self.assertGreaterEqual(score, GSM8K_SCORE_THRESHOLD)
                    self.assertTrue(
                        all(
                            not output["meta_info"].get("spec_cap_lens_histogram")
                            for output in outputs
                        )
                    )


if __name__ == "__main__":
    unittest.main()
