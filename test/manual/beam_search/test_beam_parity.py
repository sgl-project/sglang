"""Beam search parity acceptance test (executable API spec).

- Trigger: sampling_params.beam_width = k (> 1); no server-level beam flag.
  During the sync transition the server must run --disable-overlap-schedule.
- Response: one response per rid; meta_info.beam_results holds the top-n
  sequences (n <= beam_width, default = beam_width), best score first.
- Acceptance: sequence-set overlap vs HF transformers >= 0.8 for k in {2, 10}.

Manual test (GPU host): python3 test_beam_parity.py
"""

import os
import unittest
from typing import List

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

BEAM_V1_READY = True  # v1 beam path (S4) wired

PROMPT = "Hello SGLang"
MAX_NEW_TOKENS = 10
OVERLAP_THRESHOLD = 0.8


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


@unittest.skipUnless(BEAM_V1_READY, "v1 beam search path not wired yet (S4)")
class TestBeamParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(
            "SGLANG_TEST_BEAM_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        # No beam-specific server flag; overlap off is the sync-transition
        # requirement (lifted for EOS-only requests once S5.5 lands).
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--disable-overlap-schedule"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

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
        for beam_width in [2, 10]:
            beam_results = self._generate_beams(beam_width)
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


if __name__ == "__main__":
    unittest.main()
