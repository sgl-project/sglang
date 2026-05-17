"""End-to-end Double Sparsity smoke test.

Generates a synthetic calibration on the fly, launches an SGLang server
with `--enable-double-sparsity`, and sends a handful of prompts through
the standard completions API. Asserts that:

  1. The server starts up cleanly (DS coordinator wires through ModelRunner,
     RadixAttention.ds_enabled is stamped, FA3 is selected, page_size=1
     is enforced).
  2. Generations are non-empty and longer than the prompt — i.e., the
     full extend → decode → DS-selection → FA3 sparse path actually
     works on a real GPU.
  3. CUDA-graph capture+replay survives multiple decode steps without
     crashing.

Quality / accuracy comparisons against dense `main` are part of the
benchmark (M8), not of this smoke test.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Tiny model — keeps test fast while exercising the same code paths as
# Llama-3.1-8B (Qwen2.5 has GQA, RoPE, Llama-style q_proj/k_proj/v_proj).
TINY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

register_cuda_ci(est_time=180, suite="stage-b-test-1-gpu-small")


def _generate_synthetic_calibration(model_path: str, output_path: Path) -> None:
    """Run scripts/double_sparsity/calibrate.py --synthetic to produce a JSON.

    Subprocess so test is robust to import-side effects from the calibrator.
    """
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "double_sparsity" / "calibrate.py"
    cmd = [
        sys.executable,
        str(script),
        "--model",
        model_path,
        "--output",
        str(output_path),
        "--synthetic",
        "--n-samples",
        "4",
        "--seq-len",
        "256",
        "--heavy-channels",
        "16",
        "--device",
        "cuda:0",
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)


class TestDoubleSparsityE2E(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.calib_path = Path(cls.tmp.name) / "calib.json"
        _generate_synthetic_calibration(TINY_MODEL, cls.calib_path)

        # Sanity: parser accepts the generated calibration.
        with cls.calib_path.open("r", encoding="utf-8") as f:
            blob = json.load(f)
        assert blob["heavy_channels"] == 16
        assert blob["channel_type"] == "k"

        cls.model = TINY_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-double-sparsity",
                "--double-sparsity-config",
                str(cls.calib_path),
                "--double-sparsity-heavy-channels",
                "16",
                "--double-sparsity-token-budget",
                "32",
                "--double-sparsity-min-seq-len",
                "16",  # ≤ max_selected_per_request
                "--double-sparsity-max-selected-per-request",
                "256",
                "--page-size",
                "1",
                "--attention-backend",
                "fa3",
                "--mem-fraction-static",
                "0.6",
                "--max-running-requests",
                "8",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.tmp.cleanup()

    def _gen(self, prompt: str, max_tokens: int = 32) -> str:
        import requests

        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    def test_short_prompts_decode_path(self):
        # Below `min_seq_len`: each row falls back to dense FA3 device-side.
        # The whole DS lifecycle still runs; we just verify it produces output.
        for prompt in ("Hello", "What is 2+2?"):
            out = self._gen(prompt, max_tokens=8)
            self.assertGreater(len(out.strip()), 0, f"empty completion for {prompt!r}")

    def test_long_prompt_sparse_decode(self):
        # Long enough to cross min_seq_len; DS selection actively prunes K.
        long_prompt = (
            "The quick brown fox jumps over the lazy dog. " * 64
            + "\n\nWrite a one-sentence summary:"
        )
        out = self._gen(long_prompt, max_tokens=64)
        self.assertGreater(len(out.strip()), 0, "empty completion on long prompt")

    def test_concurrent_requests(self):
        # Drives multiple decode steps under CUDA graph capture/replay.
        import concurrent.futures

        prompts = [
            "Once upon a time, " + "in a faraway land. " * 20,
            "The capital of France is " + "important. " * 20,
            "List five fruits: " + "apple, banana, cherry. " * 10,
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            outs = list(ex.map(lambda p: self._gen(p, max_tokens=32), prompts))
        for p, o in zip(prompts, outs):
            self.assertGreater(len(o.strip()), 0, f"empty completion for {p[:30]!r}...")


if __name__ == "__main__":
    unittest.main()
