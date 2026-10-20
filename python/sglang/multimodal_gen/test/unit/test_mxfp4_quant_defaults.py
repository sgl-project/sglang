# SPDX-License-Identifier: Apache-2.0
"""MindIE C7 activation-quantization defaults across single-level MXFP4 paths.

CPU-checkable contract: every ``npu_dynamic_mx_quant`` call with an FP4
destination uses ``round_mode="rint"`` plus ``scale_alg=2`` /
``dst_type_max=7.25`` (checkpoint ``mxfp4_*`` descriptors override the two
kwargs). Numerical accuracy and performance require target-NPU validation.
"""

import inspect
import unittest

from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa import (
    _attention_quant_kwargs,
)
from sglang.multimodal_gen.runtime.layers.quantization.mxfp4_utils import (
    mxfp4_quant_kwargs,
)
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    MXFP4_C7_ACTIVATION_QUANT_DEFAULTS,
    NPUSingleLevelMXFP4LinearMethod,
    NPUSingleLevelMXFP4OfflineLinearMethod,
)
from sglang.test.test_utils import CustomTestCase


def _quant_op(*, round_mode=True):
    """Stand-in for torch_npu.npu_dynamic_mx_quant with a chosen signature."""

    def op(x, dst_type, **kwargs):
        return x, None

    names = ("x", "dst_type", "round_mode") if round_mode else ("x", "dst_type")
    op.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD
                if i < 2
                else inspect.Parameter.KEYWORD_ONLY,
                default=None if i >= 2 else inspect.Parameter.empty,
            )
            for i, name in enumerate(names)
        ]
    )
    return op


class TestLinearC7Defaults(CustomTestCase):
    def test_constant_matches_mindie_c7(self):
        self.assertEqual(
            MXFP4_C7_ACTIVATION_QUANT_DEFAULTS,
            {"scale_alg": 2, "dst_type_max": 7.25},
        )

    def test_single_level_kernel_defaults_to_rint(self):
        self.assertEqual(NPUSingleLevelMXFP4LinearMethod.activation_round_mode, "rint")
        self.assertEqual(
            NPUSingleLevelMXFP4OfflineLinearMethod.activation_round_mode,
            "rint",
            "the offline W4A4 kernel must inherit the rint default",
        )

    def test_checkpoint_descriptors_override_c7_kwargs(self):
        self.assertEqual(
            mxfp4_quant_kwargs({}),
            {"scale_alg": 2, "dst_type_max": 7.25},
        )
        self.assertEqual(
            mxfp4_quant_kwargs({"mxfp4_scale_alg": 1, "mxfp4_dst_type_max": 6.0}),
            {"scale_alg": 1, "dst_type_max": 6.0},
        )


class TestAttentionQuantKwargs(CustomTestCase):
    def test_rint_added_when_supported(self):
        kwargs = _attention_quant_kwargs({}, _quant_op())
        self.assertEqual(
            kwargs, {"scale_alg": 2, "dst_type_max": 7.25, "round_mode": "rint"}
        )

    def test_checkpoint_overrides_keep_rint(self):
        kwargs = _attention_quant_kwargs(
            {"mxfp4_scale_alg": 1, "mxfp4_dst_type_max": 6.0}, _quant_op()
        )
        self.assertEqual(
            kwargs, {"scale_alg": 1, "dst_type_max": 6.0, "round_mode": "rint"}
        )

    def test_missing_round_mode_falls_back_silently(self):
        kwargs = _attention_quant_kwargs({}, _quant_op(round_mode=False))
        self.assertEqual(kwargs, {"scale_alg": 2, "dst_type_max": 7.25})

    def test_non_introspectable_op_keeps_rint(self):
        from unittest import mock

        with mock.patch(
            "sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa"
            ".inspect.signature",
            side_effect=ValueError("no signature found for builtin"),
        ):
            kwargs = _attention_quant_kwargs({}, _quant_op())
        self.assertEqual(kwargs["round_mode"], "rint")


if __name__ == "__main__":
    unittest.main()
