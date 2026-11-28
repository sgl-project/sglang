"""Beam search parity acceptance test (executable API spec).

- Trigger: sampling_params.beam_width = k (> 1); no server-level beam flag.
- Response: one response per rid; meta_info.beam_results holds the top-n
  sequences (n <= beam_width, default 1 as in HF/OpenAI), best score first.
- Acceptance: sequence-set overlap vs HF transformers >= 0.8 for k in {2, 10}.
- TestBeamParityDecodeGraph: the same acceptance on a server that replays
  all-beam decode batches as CUDA graphs.

Manual test (GPU host): python3 test_beam_parity.py
"""

import io
import os
import re
import unittest
from typing import List

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

PROMPT = "Hello SGLang"
MAX_NEW_TOKENS = 10
OVERLAP_THRESHOLD = 0.8
GRAPH_BEAM_WIDTH = 10


def get_transformers_beam_sequences(
    model_path: str, prompt: str, beam_width: int, max_new_tokens: int
) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=beam_width,
            num_return_sequences=beam_width,
            do_sample=False,
        )
    sequences = [
        tokenizer.decode(seq[input_length:].cpu().tolist(), skip_special_tokens=True)
        for seq in generated
    ]

    del model
    torch.cuda.empty_cache()
    return sequences


def _decode_passes(base_url: str) -> dict:
    """Decode forwards so far, by whether they replayed a CUDA graph."""
    metrics = requests.get(f"{base_url}/metrics", timeout=30).text
    return {
        mode: sum(
            float(value)
            for value in re.findall(
                r'^sglang:cuda_graph_passes_total\{[^}]*mode="'
                + mode
                + r'"[^}]*\}\s+([0-9.eE+-]+)$',
                metrics,
                re.MULTILINE,
            )
        )
        for mode in ("decode_cuda_graph", "decode_none")
    }


class TestBeamParity(CustomTestCase):
    server_args: tuple = ()
    server_env: dict = {}
    parity_beam_widths: tuple = (2, 10)

    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(
            "SGLANG_TEST_BEAM_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.server_log = io.StringIO()
        # Overlap is pinned off so a parity mismatch can only come from the
        # search, not from scheduling.
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--disable-overlap-schedule", *cls.server_args],
            env=cls.server_env,
            return_stdout_stderr=(cls.server_log, cls.server_log),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            terminate_and_kill_process_tree(cls.process)

    def _generate_beams(self, beam_width, n=None):
        sampling_params = {"beam_width": beam_width, "max_new_tokens": MAX_NEW_TOKENS}
        if n is not None:
            sampling_params["n"] = n
        resp = requests.post(
            f"{self.base_url}/generate",
            json={"text": PROMPT, "sampling_params": sampling_params},
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        beam_results = resp.json().get("meta_info", {}).get("beam_results")
        self.assertIsNotNone(beam_results, "response carries no beam_results")
        return beam_results

    def test_parity_vs_transformers(self):
        for beam_width in self.parity_beam_widths:
            # n=beam_width to compare the whole beam set; the API default is 1.
            beam_results = self._generate_beams(beam_width, n=beam_width)
            self.assertEqual(len(beam_results), beam_width)

            scores = [r["meta_info"]["sequence_score"] for r in beam_results]
            self.assertEqual(scores, sorted(scores, reverse=True))

            sglang_sequences = {r["text"] for r in beam_results}
            hf_sequences = set(
                get_transformers_beam_sequences(
                    self.model, PROMPT, beam_width, MAX_NEW_TOKENS
                )
            )
            overlap = len(sglang_sequences & hf_sequences) / beam_width
            print(f"beam_width={beam_width} overlap={overlap:.2%}")
            self.assertGreaterEqual(overlap, OVERLAP_THRESHOLD)

    def test_return_top_n(self):
        beam_results = self._generate_beams(beam_width=10, n=3)
        self.assertEqual(len(beam_results), 3)

    def test_default_returns_one_sequence(self):
        self.assertEqual(len(self._generate_beams(beam_width=10)), 1)


class TestBeamParityDecodeGraph(TestBeamParity):
    # At 2 beams eager shared-context attention returns none of HF's beams for
    # this prompt, graphs or not, so parity covers the graph's width only.
    parity_beam_widths = (GRAPH_BEAM_WIDTH,)
    server_env = {
        "SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION": "1",
        "SGLANG_ENABLE_BEAM_DECODE_GRAPH": "1",
        "SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH": str(GRAPH_BEAM_WIDTH),
        "SGLANG_BEAM_DECODE_GRAPH_NEW_TOKEN_CAPACITY": str(MAX_NEW_TOKENS),
    }
    # Shared-context attention needs FlashInfer decode attention.
    server_args = (
        "--decode-attention-backend",
        "flashinfer",
        "--enable-metrics",
        "--cuda-graph-bs-decode",
        str(GRAPH_BEAM_WIDTH),
    )

    def test_beam_request_replays_a_beam_graph(self):
        # Parity alone also passes when every beam batch decodes eagerly.
        self.assertIn(
            f"Beam decode graphs captured for [{GRAPH_BEAM_WIDTH}] rows "
            "(compact trie head off, shared-context attention on)",
            self.server_log.getvalue(),
        )
        before = _decode_passes(self.base_url)
        self._generate_beams(GRAPH_BEAM_WIDTH)
        after = _decode_passes(self.base_url)
        self.assertGreater(after["decode_cuda_graph"], before["decode_cuda_graph"])
        self.assertEqual(after["decode_none"], before["decode_none"])


if __name__ == "__main__":
    unittest.main()
