"""CPU checks of the megakernel binding: the view is the running forward's attention
backend and KV pool, and a source replaced after capture is caught."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.megakernels import binding as mb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

BACKEND = "sglang.srt.model_executor.forward_context.get_attn_backend"


class Pool:
    uniform_fp8 = False
    request_window = None

    def __init__(self):
        self.keys = [torch.zeros(4, 8) for _ in range(2)]

    def get_swa_key_buffer_radix(self, layer_id):
        return self.keys[layer_id]

    def get_extra_key_page_size(self, layer_id):
        return 64 + layer_id

    def set_swa_key_buffer_radix_fused_norm_rope(self, **kw):
        return kw


def backend(pool):
    return SimpleNamespace(
        token_to_kv_pool=pool,
        forward_metadata=SimpleNamespace(core_attn_metadata="core"),
        trtllm_attn=False,
        forward=lambda *a, **k: "attention",
        store_cache=lambda **k: None,
        get_swa_out_cache_loc=lambda fb: "swa_loc",
        forward_low_ratio_sources=lambda **k: "sources",
        forward_core_compressor=lambda **k: "compressed",
    )


class TestMegakernelBinding(CustomTestCase):
    def test_the_view_is_the_running_forwards_backend_and_pool(self):
        pool = Pool()
        with patch(BACKEND, return_value=backend(pool)):
            view = mb.MegakernelBinding(model=torch.nn.Module()).forward_view()
        self.assertEqual(view.metadata.core_attn_metadata, "core")
        self.assertIs(view.kv_pool.pool, pool)
        self.assertIs(view.kv_pool.get_swa_key_buffer_radix(1), pool.keys[1])

    def test_a_replaced_source_fails_assert_live_by_name(self):
        model = torch.nn.Module()
        model.engram_hasher = SimpleNamespace(history=torch.zeros(5, 3))
        binding = mb.MegakernelBinding(model=model)
        attn_backend = backend(Pool())
        with patch(BACKEND, return_value=attn_backend):
            binding.forward_view()
            self.assertIs(binding.engram_history(), model.engram_hasher.history)
            binding.assert_live()
            model.engram_hasher.history = torch.zeros(9, 3)
            with self.assertRaisesRegex(RuntimeError, "engram_history"):
                binding.assert_live()
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        with patch(BACKEND, return_value=backend(Pool())):
            binding.forward_view()
        with patch(BACKEND, return_value=backend(Pool())):
            with self.assertRaisesRegex(RuntimeError, "kv_pool"):
                binding.assert_live()


if __name__ == "__main__":
    unittest.main()
