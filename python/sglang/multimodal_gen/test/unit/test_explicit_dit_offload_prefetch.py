# SPDX-License-Identifier: Apache-2.0
"""An explicit DiT layerwise request still receives the model prefetch size."""

import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.server_args.auto_tune import ServerArgsAutoTuner


class _Args:
    def __init__(self, *, explicit_prefetch=False, performance_mode="auto"):
        self.dit_layerwise_offload = True
        self.is_dit_layerwise_offload_selected = False
        self.performance_mode = performance_mode
        self.dit_offload_prefetch_size = 0.0
        self.pipeline_config = SimpleNamespace(
            task_type=SimpleNamespace(is_action_gen=lambda: False),
            get_model_deployment_config=lambda: SimpleNamespace(
                auto_dit_offload_prefetch_size=2
            ),
        )
        self._explicit_prefetch = explicit_prefetch

    def is_arg_explicitly_set(self, name):
        return self._explicit_prefetch and name == "dit_offload_prefetch_size"


class TestExplicitDitOffloadPrefetch(unittest.TestCase):
    def _tuner(self, args):
        tuner = ServerArgsAutoTuner.__new__(ServerArgsAutoTuner)
        tuner.server_args = args
        return tuner

    def test_explicit_layerwise_applies_model_prefetch_default(self):
        args = _Args()
        components = self._tuner(
            args
        )._default_layerwise_components_for_unset_placement()
        self.assertEqual(components, [])
        self.assertEqual(args.dit_offload_prefetch_size, 2)

    def test_explicit_prefetch_size_is_left_alone(self):
        args = _Args(explicit_prefetch=True)
        self._tuner(args)._default_layerwise_components_for_unset_placement()
        self.assertEqual(args.dit_offload_prefetch_size, 0.0)


if __name__ == "__main__":
    unittest.main()
