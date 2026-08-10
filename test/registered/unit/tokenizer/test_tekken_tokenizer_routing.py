"""Unit tests for bare-tekken checkpoint tokenizer routing — no server, no model loading."""

import os
import shutil
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=16, suite="base-a-test-cpu")

from sglang.srt.utils.hf_transformers.mistral_utils import is_bare_tekken_checkpoint
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

TEKKEN_REPO = "mistralai/Leanstral-1.5-119B-A6B"
PROMPT = "The capital of France is"
# Reference ids from mistral-common, which owns the tekken id space
# (1000 special-token slots precede the BPE vocab).
EXPECTED_IDS = [1784, 8961, 1307, 5498, 1395]


class TestBareTekkenDetection(CustomTestCase):
    def test_detects_bare_tekken_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(is_bare_tekken_checkpoint(d))
            with open(os.path.join(d, "tekken.json"), "w") as f:
                f.write("{}")
            self.assertTrue(is_bare_tekken_checkpoint(d))
            with open(os.path.join(d, "tokenizer.json"), "w") as f:
                f.write("{}")
            self.assertFalse(is_bare_tekken_checkpoint(d))


class TestTekkenRouting(CustomTestCase):
    def test_get_tokenizer_matches_mistral_common(self):
        from huggingface_hub import hf_hub_download

        tekken = hf_hub_download(TEKKEN_REPO, "tekken.json")
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(tekken, os.path.join(d, "tekken.json"))
            tokenizer = get_tokenizer(d)
            ids = tokenizer.encode(PROMPT, add_special_tokens=False)
            self.assertEqual(ids, EXPECTED_IDS)


if __name__ == "__main__":
    unittest.main()
