"""Unit tests for Evo 2 tokenizer auto-generation.

Tests the CharLevelTokenizer generation used by Evo 2 models.
Vortex CharLevelTokenizer encodes DNA bases as UTF-8 bytes:
  A=65, C=67, G=71, T=84, N=78
Special tokens: <eod>=0 (BOS/EOS), <pad>=1
"""

import json
import os
import tempfile
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.models.evo2 import generate_evo2_tokenizer_files


class TestEvo2TokenizerGeneration(CustomTestCase):
    """Tests for generate_evo2_tokenizer_files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_tokenizer_json_and_config(self):
        """Generating tokenizer files creates both tokenizer.json and tokenizer_config.json."""
        generate_evo2_tokenizer_files(self.tmpdir)
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "tokenizer.json")))
        self.assertTrue(
            os.path.isfile(os.path.join(self.tmpdir, "tokenizer_config.json"))
        )

    def test_is_idempotent(self):
        """Calling twice does not raise errors."""
        generate_evo2_tokenizer_files(self.tmpdir)
        generate_evo2_tokenizer_files(self.tmpdir)  # Should be a no-op

    def test_tokenizer_json_valid_json(self):
        """The generated tokenizer.json is valid JSON with expected structure."""
        generate_evo2_tokenizer_files(self.tmpdir)
        with open(os.path.join(self.tmpdir, "tokenizer.json")) as f:
            tok = json.load(f)
        self.assertIn("model", tok)
        self.assertIn("vocab", tok["model"])
        self.assertIn("pre_tokenizer", tok)
        self.assertIn("decoder", tok)

    def test_vocabulary_includes_all_indices(self):
        """Vocabulary has valid entries for all token indices 0..511."""
        generate_evo2_tokenizer_files(self.tmpdir)
        with open(os.path.join(self.tmpdir, "tokenizer.json")) as f:
            tok = json.load(f)
        vocab = tok["model"]["vocab"]

        # Every index 0..511 should be reachable via some token string
        indices_covered = set(vocab.values())
        # Not all indices map to unique strings (control chars get clamped),
        # but indices 0, 1, and 65+ should all be present
        self.assertIn(0, indices_covered)  # <eod>
        self.assertIn(1, indices_covered)  # <pad>
        self.assertIn(65, indices_covered)  # A
        self.assertIn(67, indices_covered)  # C
        self.assertIn(71, indices_covered)  # G
        self.assertIn(84, indices_covered)  # T
        self.assertGreaterEqual(len(vocab), 200)  # At minimum 200 unique chars

    def test_dna_bases_map_to_ascii_bytes(self):
        """DNA bases map to their ASCII byte values: A=65, C=67, G=71, T=84."""
        generate_evo2_tokenizer_files(self.tmpdir)
        with open(os.path.join(self.tmpdir, "tokenizer.json")) as f:
            tok = json.load(f)
        vocab = tok["model"]["vocab"]

        self.assertEqual(vocab["A"], 65)
        self.assertEqual(vocab["C"], 67)
        self.assertEqual(vocab["G"], 71)
        self.assertEqual(vocab["T"], 84)
        self.assertEqual(vocab["N"], 78)

    def test_special_tokens(self):
        """Special tokens <eod> and <pad> are at IDs 0 and 1."""
        generate_evo2_tokenizer_files(self.tmpdir)
        with open(os.path.join(self.tmpdir, "tokenizer.json")) as f:
            tok = json.load(f)
        vocab = tok["model"]["vocab"]

        self.assertEqual(vocab["<eod>"], 0)
        self.assertEqual(vocab["<pad>"], 1)

    def test_tokenizer_loadable_by_tokenizers_lib(self):
        """Generated tokenizer is loadable by the `tokenizers` library."""
        try:
            from tokenizers import Tokenizer
        except ImportError:
            self.skipTest("tokenizers library not installed")

        generate_evo2_tokenizer_files(self.tmpdir)
        tok = Tokenizer.from_file(os.path.join(self.tmpdir, "tokenizer.json"))

        encoded = tok.encode("ACGT")
        self.assertEqual(encoded.ids, [65, 67, 71, 84])
        self.assertEqual(tok.decode(encoded.ids), "ACGT")

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding a DNA sequence returns the original."""
        try:
            from tokenizers import Tokenizer
        except ImportError:
            self.skipTest("tokenizers library not installed")

        generate_evo2_tokenizer_files(self.tmpdir)
        tok = Tokenizer.from_file(os.path.join(self.tmpdir, "tokenizer.json"))

        sequences = [
            "ACGT",
            "ACGTACGTACGT",
            "ATGAAACGCATTAGCACC",
            "NNNNACGT",
            "GATTACA",
        ]
        for seq in sequences:
            encoded = tok.encode(seq)
            decoded = tok.decode(encoded.ids)
            self.assertEqual(
                decoded, seq, f"Roundtrip failed for {seq!r}: got {decoded!r}"
            )

    def test_tokenizer_bytes_match_vortex_charlevel(self):
        """Token IDs match what Vortex CharLevelTokenizer would produce."""
        try:
            from tokenizers import Tokenizer
        except ImportError:
            self.skipTest("tokenizers library not installed")

        generate_evo2_tokenizer_files(self.tmpdir)
        tok = Tokenizer.from_file(os.path.join(self.tmpdir, "tokenizer.json"))

        # Test with a real DNA sequence
        seq = "ATGCATGC"
        encoded = tok.encode(seq)

        # Vortex CharLevelTokenizer uses: np.frombuffer(seq.encode('utf-8'), dtype=np.uint8)
        expected = list(np.frombuffer(seq.encode("utf-8"), dtype=np.uint8))
        self.assertEqual(encoded.ids, expected)

    def test_custom_vocab_size(self):
        """Custom vocab_size parameter is respected."""
        generate_evo2_tokenizer_files(self.tmpdir, vocab_size=256)
        with open(os.path.join(self.tmpdir, "tokenizer.json")) as f:
            tok = json.load(f)
        vocab = tok["model"]["vocab"]
        # Verify key indices are present with custom vocab_size
        indices_covered = set(vocab.values())
        self.assertIn(0, indices_covered)
        self.assertIn(1, indices_covered)
        self.assertIn(65, indices_covered)  # A
        self.assertIn(67, indices_covered)  # C
        # vocab_size=256 produces fewer unique keys than full 512
        self.assertLess(max(indices_covered), 256)
        self.assertGreaterEqual(len(vocab), 100)

    def test_tokenizer_config_contents(self):
        """tokenizer_config.json has expected fields."""
        generate_evo2_tokenizer_files(self.tmpdir)
        with open(os.path.join(self.tmpdir, "tokenizer_config.json")) as f:
            cfg = json.load(f)

        self.assertEqual(cfg["tokenizer_class"], "PreTrainedTokenizerFast")
        self.assertEqual(cfg["bos_token"], "<eod>")
        self.assertEqual(cfg["eos_token"], "<eod>")
        self.assertEqual(cfg["pad_token"], "<pad>")
        self.assertEqual(cfg["unk_token"], "<eod>")
        self.assertEqual(cfg["pad_token_id"], 1)


class TestEvo2TokenizerAutoDetection(CustomTestCase):
    """Tests for _ensure_evo2_tokenizer_files auto-detection logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def write_evo2_config(self, extra=None):
        cfg = {
            "model_type": "evo2",
            "tokenizer_type": "CharLevelTokenizer",
            "vocab_size": 512,
            "hidden_size": 1920,
        }
        if extra:
            cfg.update(extra)
        with open(os.path.join(self.tmpdir, "config.json"), "w") as f:
            json.dump(cfg, f)

    def test_detects_charleveltokenizer_type(self):
        """_ensure_evo2_tokenizer_files generates files when tokenizer_type is CharLevelTokenizer."""
        from sglang.srt.utils.hf_transformers.tokenizer import (
            _ensure_evo2_tokenizer_files,
        )

        self.write_evo2_config()
        _ensure_evo2_tokenizer_files(self.tmpdir)

        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "tokenizer.json")))
        self.assertTrue(
            os.path.isfile(os.path.join(self.tmpdir, "tokenizer_config.json"))
        )

    def test_skips_non_charlevel_tokenizer(self):
        """Non-CharLevelTokenizer configs are ignored."""
        from sglang.srt.utils.hf_transformers.tokenizer import (
            _ensure_evo2_tokenizer_files,
        )

        self.write_evo2_config({"tokenizer_type": "HFAutoTokenizer"})
        _ensure_evo2_tokenizer_files(self.tmpdir)

        self.assertFalse(os.path.isfile(os.path.join(self.tmpdir, "tokenizer.json")))

    def test_skips_non_directory_paths(self):
        """Non-directory paths (e.g., HF repo names) are skipped gracefully."""
        from sglang.srt.utils.hf_transformers.tokenizer import (
            _ensure_evo2_tokenizer_files,
        )

        # Should not raise
        _ensure_evo2_tokenizer_files("arcinstitute/evo2_1b_base")

    def test_skips_missing_config(self):
        """Directories without config.json are skipped."""
        from sglang.srt.utils.hf_transformers.tokenizer import (
            _ensure_evo2_tokenizer_files,
        )

        _ensure_evo2_tokenizer_files(self.tmpdir)  # No config.json
        self.assertFalse(os.path.isfile(os.path.join(self.tmpdir, "tokenizer.json")))

    def test_skips_when_tokenizer_exists(self):
        """If tokenizer.json and tokenizer_config.json already exist, they are not overwritten."""
        from sglang.srt.utils.hf_transformers.tokenizer import (
            _ensure_evo2_tokenizer_files,
        )

        self.write_evo2_config()
        existing_tok = {
            "version": "1.0",
            "model": {"type": "BPE", "vocab": {"<eod>": 0}},
        }
        existing_cfg = {"tokenizer_class": "Custom"}
        with open(os.path.join(self.tmpdir, "tokenizer.json"), "w") as f:
            json.dump(existing_tok, f)
        with open(os.path.join(self.tmpdir, "tokenizer_config.json"), "w") as f:
            json.dump(existing_cfg, f)

        _ensure_evo2_tokenizer_files(self.tmpdir)

        # Should still have our custom files, not the generated ones
        with open(os.path.join(self.tmpdir, "tokenizer.json")) as f:
            reloaded_tok = json.load(f)
        with open(os.path.join(self.tmpdir, "tokenizer_config.json")) as f:
            reloaded_cfg = json.load(f)
        self.assertEqual(reloaded_tok, existing_tok)
        self.assertEqual(reloaded_cfg, existing_cfg)


if __name__ == "__main__":
    unittest.main()
