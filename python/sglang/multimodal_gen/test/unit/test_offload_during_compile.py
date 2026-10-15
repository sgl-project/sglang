"""Leaving the offload-during-compile warmup must not keep its graph breaks."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch._dynamo.testing import CompileCounter

from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.test.test_utils import CustomTestCase


class _Block(nn.Module):
    def forward(self, x):
        return x * 2 + 1


class _DiT(nn.Module):
    """Blocks in a loop, each wrapped by an offload hook Dynamo cannot trace."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(3))
        self.hooks = [b.register_forward_pre_hook(_offload_hook) for b in self.blocks]

    def disable_offload(self):
        for hook in self.hooks:
            hook.remove()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


@torch.compiler.disable
def _offload_hook(module, args):
    return None


class TestOffloadDuringCompile(CustomTestCase):
    def setUp(self):
        torch._dynamo.reset()
        self.dit = _DiT()
        self.counter = CompileCounter()
        self.dit.compile(backend=self.counter)
        self.stage = SimpleNamespace(
            _offloaded_dit_modules_for_compile=[self.dit],
            _move_resident_components_for_warmup=lambda: [],
        )

    def _forward(self, *, is_warmup):
        with DenoisingStage._offload_for_torch_compile_warmup(
            self.stage, SimpleNamespace(is_warmup=is_warmup)
        ):
            self.dit(torch.ones(4))

    def test_warmup_keeps_the_dit_offloaded(self):
        self._forward(is_warmup=True)
        self._forward(is_warmup=True)

        self.assertEqual(self.stage._offloaded_dit_modules_for_compile, [self.dit])
        self.assertTrue(all(b._forward_pre_hooks for b in self.dit.blocks))

    def test_serving_compiles_the_whole_dit_after_the_warmup(self):
        self._forward(is_warmup=True)
        self.counter.frame_count = self.counter.op_count = 0

        self._forward(is_warmup=False)
        self._forward(is_warmup=False)

        # One graph holding all three blocks; without the reset Dynamo keeps
        # skipping the frame it gave up on during the warmup and compiles none.
        self.assertEqual(self.counter.frame_count, 1)
        self.assertEqual(self.counter.op_count, 6)
        self.assertEqual(self.stage._offloaded_dit_modules_for_compile, [])


if __name__ == "__main__":
    unittest.main()
