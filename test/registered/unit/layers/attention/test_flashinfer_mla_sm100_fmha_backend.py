"""Unit tests for SM100 ragged FMHA backend selection in FlashInfer MLA."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.flashinfer_backend import (
    SM100_CUTLASS_FMHA_HEAD_DIMS,
)
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    _select_ragged_fmha_backend,
)
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cfg(qk_nope_head_dim: int, qk_rope_head_dim: int, v_head_dim: int):
    return SimpleNamespace(
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
    )


class TestFlashInferMlaSm100FmhaBackend(CustomTestCase):
    def test_sm100_uses_cutlass_only_for_instantiated_head_dims(self):
        cases = [
            ((128, 64, 128), "cutlass"),  # DeepSeek-V2/V3: (192, 128)
            ((128, 0, 128), "cutlass"),
            ((64, 0, 64), "cutlass"),
            ((192, 64, 256), "auto"),  # GLM-4.7-Flash: (256, 256)
            ((192, 64, 128), "auto"),
        ]
        self.assertEqual(
            SM100_CUTLASS_FMHA_HEAD_DIMS,
            frozenset({(64, 64), (128, 128), (192, 128)}),
        )
        with override_platform(is_sm100=True):
            for dims, expected in cases:
                with self.subTest(dims=dims):
                    self.assertEqual(_select_ragged_fmha_backend(_cfg(*dims)), expected)

    def test_non_sm100_always_uses_auto(self):
        with override_platform(is_sm100=False):
            self.assertEqual(_select_ragged_fmha_backend(_cfg(128, 64, 128)), "auto")


if __name__ == "__main__":
    unittest.main()
