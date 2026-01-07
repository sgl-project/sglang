import os
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    is_in_ci,
    popen_launch_server,
)


def _send_generate(base_url: str, prompt: str, *, max_new_tokens: int) -> dict:
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


class TestQwen3DFlashCorrectness(CustomTestCase):
    def test_qwen3_dflash_matches_target_only_greedy(self):
        if is_in_ci():
            self.skipTest("Manual test; skipped in CI.")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for this manual DFlash integration test.")

        target_model = os.getenv("SGLANG_DFLASH_TARGET_MODEL", "Qwen/Qwen3-8B")
        draft_model_path = os.getenv(
            "SGLANG_DFLASH_DRAFT_MODEL_PATH", "/tmp/Qwen3-8B-DFlash-bf16"
        )
        if not os.path.isdir(draft_model_path):
            self.skipTest(
                f"Draft model folder not found: {draft_model_path}. "
                "Set SGLANG_DFLASH_DRAFT_MODEL_PATH to run this test."
            )

        max_new_tokens = int(os.getenv("SGLANG_DFLASH_MAX_NEW_TOKENS", "128"))
        prompt = os.getenv(
            "SGLANG_DFLASH_PROMPT",
            "How many positive whole-number divisors does 196 have?",
        )
        attention_backend = os.getenv("SGLANG_DFLASH_ATTENTION_BACKEND", "flashinfer")

        baseline_port = find_available_port(20000)
        dflash_port = find_available_port(baseline_port + 1)
        baseline_url = f"http://127.0.0.1:{baseline_port}"
        dflash_url = f"http://127.0.0.1:{dflash_port}"

        # 1) Target-only baseline.
        baseline_proc = popen_launch_server(
            target_model,
            baseline_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-radix-cache",
                "--attention-backend",
                attention_backend,
            ],
        )
        try:
            baseline = _send_generate(
                baseline_url, prompt, max_new_tokens=max_new_tokens
            )
        finally:
            kill_process_tree(baseline_proc.pid)

        # 2) DFLASH speculative decoding.
        dflash_proc = popen_launch_server(
            target_model,
            dflash_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-radix-cache",
                "--attention-backend",
                attention_backend,
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                draft_model_path,
            ],
        )
        try:
            dflash = _send_generate(dflash_url, prompt, max_new_tokens=max_new_tokens)
        finally:
            kill_process_tree(dflash_proc.pid)

        self.assertEqual(
            baseline["output_ids"],
            dflash["output_ids"],
            f"Token IDs mismatch.\nbaseline={baseline['output_ids']}\ndflash={dflash['output_ids']}",
        )
        self.assertEqual(
            baseline["text"],
            dflash["text"],
            "Decoded text mismatch for greedy decoding.",
        )

        meta = dflash.get("meta_info", {})
        self.assertIn("spec_verify_ct", meta, f"Missing spec metrics: {meta.keys()}")
        self.assertGreater(meta["spec_verify_ct"], 0, "DFLASH did not run verify steps.")
        self.assertIn("spec_accept_length", meta, f"Missing spec_accept_length: {meta.keys()}")
        self.assertGreaterEqual(
            float(meta["spec_accept_length"]),
            1.0,
            "Spec accept length should be >= 1.0 (bonus token).",
        )
        print(
            "DFLASH metrics:",
            {
                "spec_verify_ct": meta.get("spec_verify_ct"),
                "spec_accept_length": meta.get("spec_accept_length"),
                "spec_accept_rate": meta.get("spec_accept_rate"),
                "spec_accept_token_num": meta.get("spec_accept_token_num"),
                "spec_draft_token_num": meta.get("spec_draft_token_num"),
                "completion_tokens": meta.get("completion_tokens"),
            },
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
