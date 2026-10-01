"""Denoising graph buckets bound the cached prefix without reading GPU lengths."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.graph_variants import (
    DllmBmmGraphVariants,
    create_dllm_bmm_graph_variants,
    get_dllm_bmm_capture_prefix_capacity,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_utils.capture_mode import (
    _set_capture_attention_variant,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDllmBmmGraphVariants(CustomTestCase):
    def test_prefix_boundaries_include_full_canvas(self):
        variants = DllmBmmGraphVariants(256, tuple(range(512, 16385, 512)))
        batch = SimpleNamespace(
            batch_size=1, input_ids=torch.empty(256), seq_lens_cpu=None
        )
        for prefix, expected in [
            (0, 512),
            (512, 512),
            (513, 1024),
            (2048, 2048),
            (2049, 2560),
            (8000, 8192),
            (8256, 8704),
            (8512, 8704),
            (8768, 9216),
            (16384, 16384),
            (-1, 16384),
        ]:
            with self.subTest(prefix=prefix):
                batch.seq_lens_cpu = torch.tensor([prefix + 256])
                label = variants.select(batch)
                self.assertEqual(label, f"dllm_bmm_prefix_{expected}")
                try:
                    _set_capture_attention_variant(label)
                    self.assertEqual(get_dllm_bmm_capture_prefix_capacity(), expected)
                finally:
                    _set_capture_attention_variant(None)
        self.assertEqual(variants.capture_labels[0], "dllm_bmm_prefix_512")
        self.assertEqual(variants.capture_labels[-1], "dllm_bmm_prefix_16384")
        self.assertIsNone(get_dllm_bmm_capture_prefix_capacity())

    def test_missing_host_mirror_or_wrong_width_uses_full_prefix(self):
        variants = DllmBmmGraphVariants(256, (2048, 16384))
        # A CUDA-like object without scalar conversion makes accidental D2H
        # length reads fail rather than silently synchronizing the device.
        gpu_lengths = SimpleNamespace(device=SimpleNamespace(type="cuda"))
        for lengths, batch_size, tokens in [
            (None, 1, 256),
            (gpu_lengths, 1, 256),
            (torch.tensor([512, 512]), 2, 512),
            (torch.tensor([512]), 1, 33),
            (torch.tensor([]), 1, 256),
        ]:
            batch = SimpleNamespace(
                batch_size=batch_size,
                input_ids=torch.empty(tokens),
                seq_lens_cpu=lengths,
            )
            self.assertEqual(variants.select(batch), "dllm_bmm_prefix_16384")

    def test_factory_does_not_expand_other_capture_modes(self):
        config = SimpleNamespace(
            dtype=torch.bfloat16,
            context_len=16384,
            hf_config=SimpleNamespace(
                architectures=["DiffusionGemmaForBlockDiffusion"]
            ),
        )
        runner = SimpleNamespace(model_config=config, device="cpu")
        for mode, width, sizes in [
            (ForwardMode.DECODE, 256, [1]),
            (ForwardMode.DLLM_EXTEND, 256, [1, 2]),
            (ForwardMode.DLLM_EXTEND, 512, [1]),
            (ForwardMode.DLLM_EXTEND, 256, [1]),
        ]:
            self.assertIsNone(
                create_dllm_bmm_graph_variants(runner, None, mode, width, sizes)
            )


if __name__ == "__main__":
    unittest.main()
