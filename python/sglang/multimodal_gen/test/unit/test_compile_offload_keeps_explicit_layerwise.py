# SPDX-License-Identifier: Apache-2.0
"""Compile warmup must not clear a layerwise DiT the user asked to keep."""

import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)


class _OffloadModule:
    def __init__(self):
        self.disable_calls = 0

    def disable_offload(self):
        self.disable_calls += 1


class TestCompileWarmupKeepsExplicitLayerwise(unittest.TestCase):
    def _stage(self, **server_args):
        stage = DenoisingStage.__new__(DenoisingStage)
        stage.server_args = SimpleNamespace(**server_args)
        module = _OffloadModule()
        stage._offloaded_dit_modules_for_compile = [module]
        return stage, module

    def _finish_compile_window(self, stage):
        batch = SimpleNamespace(is_warmup=False)
        with stage._offload_for_torch_compile_warmup(batch):
            pass

    def test_explicit_dit_layerwise_offload_stays_enabled(self):
        stage, module = self._stage(
            dit_layerwise_offload=True,
            is_dit_layerwise_offload_selected=False,
        )
        self._finish_compile_window(stage)
        self.assertEqual(module.disable_calls, 0)
        self.assertEqual(stage._offloaded_dit_modules_for_compile, [])

    def test_compile_only_offload_is_disabled_for_the_real_request(self):
        stage, module = self._stage(
            dit_layerwise_offload=False,
            is_dit_layerwise_offload_selected=False,
        )
        self._finish_compile_window(stage)
        self.assertEqual(module.disable_calls, 1)
        self.assertEqual(stage._offloaded_dit_modules_for_compile, [])


if __name__ == "__main__":
    unittest.main()
