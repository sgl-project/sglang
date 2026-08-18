"""Regression: SGLang must not double-apply Laguna's YaRN mscale.

attention_factor is the final mscale, but SGLang multiplies attn_factor onto its
own default, so a naive copy squares it (factor 128: 1.4852 -> 2.2058). These
tests build the actual rotary embedding via ``get_rope`` and assert its served
``mscale`` matches HF's attention_factor, so they track yarn.py's real
composition rule rather than a hand-copy of it.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.configs.laguna import _to_sglang_rope_scaling
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# (factor, attention_factor). S/XS ship attention_factor == the yarn default for
# their factor; _NON_DEFAULT differs from both 1.0 and the default, so a naive
# attn_factor=1.0 or a straight copy (which squares it) is caught.
_LAGUNA_S = (128.0, 1.4852030263919618)
_LAGUNA_XS = (32.0, 1.3465735902799727)
_NON_DEFAULT = (32.0, 1.1)


class TestLagunaRopeScaling(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # base.RotaryEmbedding.__init__ reads get_server_args().
        publish(ServerArgs(model_path="dummy"), role="test")

    def _composed_mscale(self, factor, attention_factor):
        """The mscale the served YaRN embedding actually applies."""
        rope_scaling = _to_sglang_rope_scaling(
            {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": 8192,
                "attention_factor": attention_factor,
            }
        )
        # Pose as the CPU engine: RotaryEmbedding.__init__ otherwise hard-imports
        # vllm on the GPU-less CI runners. No kernel or forward is used here.
        with mock.patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True):
            emb = get_rope(
                128,
                rotary_dim=128,
                max_position=262144,
                base=500000,
                rope_scaling=rope_scaling,
                partial_rotary_factor=0.5,
                dtype=torch.float32,
            )
        return float(emb.mscale)

    def test_s_mscale_matches_hf(self):
        factor, af = _LAGUNA_S
        self.assertAlmostEqual(self._composed_mscale(factor, af), af, places=6)

    def test_xs_mscale_matches_hf(self):
        factor, af = _LAGUNA_XS
        self.assertAlmostEqual(self._composed_mscale(factor, af), af, places=6)

    def test_non_default_attention_factor(self):
        # attention_factor != the yarn default (~1.346): attn_factor=1.0 would land
        # on the default, and copying attention_factor straight in would square it.
        factor, af = _NON_DEFAULT
        self.assertAlmostEqual(self._composed_mscale(factor, af), af, places=6)

    def test_default_and_empty_rope_stay_plain(self):
        self.assertIsNone(_to_sglang_rope_scaling({"rope_type": "default"}))
        self.assertIsNone(_to_sglang_rope_scaling({}))


if __name__ == "__main__":
    unittest.main()
