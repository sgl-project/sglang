"""Unit tests for DFLASH Domino projector config parsing in srt/speculative/dflash_utils.

These cover the Domino-specific fields added to ``DFlashDraftConfig`` and the
validation in ``parse_dflash_draft_config``. No server / GPU is required: the
parser operates on plain HF-style config dicts.
"""

import unittest

from sglang.srt.speculative.dflash_utils import (
    is_dflash_domino_projector,
    parse_dflash_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _ordinary_dflash_config() -> dict:
    """A plain (non-Domino) DFLASH draft config dict."""
    return {
        "vocab_size": 151936,
        "hidden_size": 4096,
        "num_hidden_layers": 1,
        "block_size": 16,
        "dflash_config": {
            "num_target_layers": 36,
            "target_layer_ids": [1, 9, 17, 25, 33],
        },
    }


def _domino_dflash_config() -> dict:
    """A Domino DFLASH draft config dict matching the public b16 checkpoint."""
    cfg = _ordinary_dflash_config()
    cfg["emb_dim"] = 256
    cfg["dflash_config"].update(
        {
            "projector_type": "domino",
            "gru_hidden_dim": 1024,
            "pure_draft_prefix_len": 1,
            "shift_label": True,
        }
    )
    return cfg


class TestDFlashDominoConfigParsing(CustomTestCase):
    def test_ordinary_config_still_parses(self):
        parsed = parse_dflash_draft_config(draft_hf_config=_ordinary_dflash_config())
        self.assertIsNone(parsed.projector_type)
        self.assertIsNone(parsed.gru_hidden_dim)
        self.assertIsNone(parsed.emb_dim)
        self.assertEqual(parsed.pure_draft_prefix_len, 0)
        self.assertFalse(parsed.shift_label)
        self.assertFalse(is_dflash_domino_projector(parsed.projector_type))

    def test_domino_config_parses(self):
        parsed = parse_dflash_draft_config(draft_hf_config=_domino_dflash_config())
        self.assertEqual(parsed.projector_type, "domino")
        self.assertEqual(parsed.gru_hidden_dim, 1024)
        self.assertEqual(parsed.emb_dim, 256)
        self.assertEqual(parsed.pure_draft_prefix_len, 1)
        self.assertTrue(parsed.shift_label)
        self.assertTrue(is_dflash_domino_projector(parsed.projector_type))

    def test_causal_v5_alias_recognized(self):
        cfg = _domino_dflash_config()
        cfg["dflash_config"]["projector_type"] = "causal_v5"
        parsed = parse_dflash_draft_config(draft_hf_config=cfg)
        self.assertEqual(parsed.projector_type, "causal_v5")
        self.assertTrue(is_dflash_domino_projector(parsed.projector_type))

    def test_unsupported_projector_type_fails(self):
        cfg = _domino_dflash_config()
        cfg["dflash_config"]["projector_type"] = "totally_unsupported"
        with self.assertRaises(ValueError):
            parse_dflash_draft_config(draft_hf_config=cfg)

    def test_missing_gru_hidden_dim_fails(self):
        cfg = _domino_dflash_config()
        del cfg["dflash_config"]["gru_hidden_dim"]
        with self.assertRaises(ValueError):
            parse_dflash_draft_config(draft_hf_config=cfg)

    def test_missing_emb_dim_fails(self):
        cfg = _domino_dflash_config()
        del cfg["emb_dim"]
        with self.assertRaises(ValueError):
            parse_dflash_draft_config(draft_hf_config=cfg)

    def test_unsupported_pure_draft_prefix_len_fails(self):
        cfg = _domino_dflash_config()
        cfg["dflash_config"]["pure_draft_prefix_len"] = 2
        with self.assertRaises(NotImplementedError):
            parse_dflash_draft_config(draft_hf_config=cfg)

    def test_non_bool_shift_label_fails(self):
        cfg = _domino_dflash_config()
        cfg["dflash_config"]["shift_label"] = 1
        with self.assertRaises(ValueError):
            parse_dflash_draft_config(draft_hf_config=cfg)


if __name__ == "__main__":
    unittest.main()
