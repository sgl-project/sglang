"""
Unit tests for LongcatFlashForCausalLM.load_weights.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.longcat_flash import LongcatFlashForCausalLM


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def load_weight(self, param, weight_name, loaded_weight, *args, **kwargs):
        self.loaded = (param, weight_name, loaded_weight, args, kwargs)


class TestLongcatFlashWeightLoading(unittest.TestCase):
    def _make_minimal_model(self, named_parameters=()):
        model = object.__new__(LongcatFlashForCausalLM)
        model.config = SimpleNamespace(n_routed_experts=0)
        model.quant_config = None
        model.named_parameters = lambda: iter(named_parameters)
        return model

    def test_flattened_ngram_names_remapped(self):
        """Regression test for the flattened N-gram embedding checkpoint key
        scheme (``oe_embed_tokens{i}.weight`` / ``oe_embed_proj{i}.weight``)
        used by some public checkpoints, which must be remapped to the
        nested ``model.ngram_embeddings.{embedders,post_projs}.{i}.weight``
        layout that ``NgramEmbedding.load_weight`` expects."""
        for src_prefix, dst_prefix in (
            ("oe_embed_tokens", "model.ngram_embeddings.embedders"),
            ("oe_embed_proj", "model.ngram_embeddings.post_projs"),
        ):
            for index in (0, 1, 11):
                with self.subTest(src_prefix=src_prefix, index=index):
                    model = self._make_minimal_model()
                    model.use_ngram_embedding = True
                    model.model = SimpleNamespace(embed_tokens=_FakeParam())
                    loaded_weight = torch.ones(1)

                    model.load_weights([(f"{src_prefix}{index}.weight", loaded_weight)])

                    self.assertEqual(
                        model.model.embed_tokens.loaded,
                        (None, f"{dst_prefix}.{index}.weight", loaded_weight, (), {}),
                    )


if __name__ == "__main__":
    unittest.main()
