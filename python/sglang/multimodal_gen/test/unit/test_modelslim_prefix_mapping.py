# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ModelSlimConfig's leaf-only prefix renaming.

``str.replace(proj_name, new_name)`` rewrites every occurrence of
``proj_name`` in the prefix, not just the trailing path component, so a
prefix where an earlier segment happens to equal the leaf name got
corrupted. Both call sites now rebuild the prefix from its last
``.``-segment instead.
"""

import unittest
from unittest import mock

from sglang.multimodal_gen.runtime.layers.linear import LinearBase
from sglang.multimodal_gen.runtime.layers.quantization.modelslim import (
    ModelSlimConfig,
)


class TestIsLayerSkippedLeafRename(unittest.TestCase):
    def test_only_renames_the_leaf_segment(self):
        # "q" also appears as an earlier path segment ("model.q."). A global
        # replace would turn that into "model.to_q." too, missing the real
        # checkpoint key below, which only renames the trailing component.
        config = ModelSlimConfig(
            quant_config={"model.q.transformer.blocks.0.to_q.weight": "FLOAT"}
        )
        is_skipped = config.is_layer_skipped(
            "model.q.transformer.blocks.0.q", {"q": ["to_q"]}
        )
        self.assertTrue(is_skipped)


class TestGetQuantMethodLeafRename(unittest.TestCase):
    def test_scheme_lookup_uses_leaf_only_rename(self):
        config = ModelSlimConfig(
            quant_config={},
            reverse_param_names_mapping=None,
        )
        config.packed_modules_mapping = {"model": {"q": ["to_q"]}}
        layer = mock.Mock(spec=LinearBase)

        with mock.patch.object(config, "get_scheme") as get_scheme:
            config.get_quant_method(layer, "model.q.transformer.blocks.0.q")

        get_scheme.assert_called_once_with(
            layer=layer, layer_name="model.q.transformer.blocks.0.to_q"
        )


if __name__ == "__main__":
    unittest.main()
