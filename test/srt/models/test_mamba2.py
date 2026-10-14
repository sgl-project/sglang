"""CPU unit test for Mamba2 (Mamba-Codestral) checkpoint weight-name remapping.

Exercises ``Mamba2ForCausalLM.load_weights`` name translation from the
HuggingFace ``backbone.*`` checkpoint layout to SGLang module names, without
constructing the full model or requiring a GPU:

  - ``backbone.``       -> ``model.``
  - ``embeddings.``     -> ``embed_tokens.``
  - ``norm_f.``         -> ``norm.``
  - ``...mixer.A_log``  -> ``...mixer.A``
  - ``lm_head.weight``  kept as-is
  - ``*inv_freq``       entries skipped

Run: python3 test/srt/models/test_mamba2.py
"""

import unittest

import torch

from sglang.srt.models.mamba2 import Mamba2ForCausalLM


def _param(like: torch.Tensor) -> torch.nn.Parameter:
    """A parameter whose weight_loader copies in place (SGLang loader contract)."""
    p = torch.nn.Parameter(torch.zeros_like(like), requires_grad=False)
    p.weight_loader = lambda param, loaded: param.data.copy_(loaded)
    return p


class _FakeMamba2:
    """Minimal stand-in exposing named_parameters() with SGLang-side names."""

    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return list(self._params.items())

    # Exercise the real method as an unbound function (no full model build).
    load_weights = Mamba2ForCausalLM.load_weights


class TestMamba2WeightRemap(unittest.TestCase):
    def test_backbone_names_are_remapped_and_loaded(self):
        # SGLang-side parameters (load targets).
        sgl = {
            "model.embed_tokens.weight": _param(torch.empty(4, 3)),
            "model.layers.0.norm.weight": _param(torch.empty(3)),
            "model.layers.0.mixer.A": _param(torch.empty(2)),
            "model.norm.weight": _param(torch.empty(3)),
            "lm_head.weight": _param(torch.empty(4, 3)),
        }
        model = _FakeMamba2(sgl)

        # HuggingFace checkpoint names (sources) with distinct values.
        hf = {
            "backbone.embeddings.weight": torch.arange(12, dtype=torch.float32).reshape(
                4, 3
            ),
            "backbone.layers.0.norm.weight": torch.tensor([1.0, 2.0, 3.0]),
            "backbone.layers.0.mixer.A_log": torch.tensor([5.0, 6.0]),
            "backbone.norm_f.weight": torch.tensor([7.0, 8.0, 9.0]),
            "lm_head.weight": torch.full((4, 3), 4.0),
        }

        loaded = model.load_weights(list(hf.items()))

        # Every source mapped onto exactly its SGLang target.
        self.assertEqual(loaded, set(sgl))
        torch.testing.assert_close(
            sgl["model.embed_tokens.weight"].data, hf["backbone.embeddings.weight"]
        )
        torch.testing.assert_close(
            sgl["model.layers.0.norm.weight"].data,
            hf["backbone.layers.0.norm.weight"],
        )
        # A_log -> A: the raw checkpoint tensor lands in the A parameter.
        torch.testing.assert_close(
            sgl["model.layers.0.mixer.A"].data, hf["backbone.layers.0.mixer.A_log"]
        )
        # norm_f -> norm (final norm), distinct from the per-layer norm above.
        torch.testing.assert_close(
            sgl["model.norm.weight"].data, hf["backbone.norm_f.weight"]
        )
        torch.testing.assert_close(sgl["lm_head.weight"].data, hf["lm_head.weight"])

    def test_inv_freq_entries_are_skipped(self):
        sgl = {"model.layers.0.mixer.A": _param(torch.empty(2))}
        model = _FakeMamba2(sgl)

        hf = [
            ("backbone.layers.0.mixer.A_log", torch.tensor([1.0, 2.0])),
            ("backbone.layers.0.mixer.inv_freq", torch.tensor([0.0, 0.0])),
            ("rotary_emb.inv_freq", torch.tensor([0.0])),
        ]

        loaded = model.load_weights(hf)

        # Only the A parameter is loaded; inv_freq sources are ignored.
        self.assertEqual(loaded, {"model.layers.0.mixer.A"})

    def test_unmatched_source_is_ignored_not_fatal(self):
        sgl = {"model.norm.weight": _param(torch.empty(2))}
        model = _FakeMamba2(sgl)

        hf = [
            ("backbone.norm_f.weight", torch.tensor([1.0, 2.0])),
            ("backbone.this.does.not.exist", torch.tensor([9.0])),
        ]

        loaded = model.load_weights(hf)
        self.assertEqual(loaded, {"model.norm.weight"})


if __name__ == "__main__":
    unittest.main()
