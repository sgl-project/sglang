"""CPU contract of the megakernel binding: the running forward's attention data and
KV pool under SGLang's own names, read from the forward context; sources sealed at
their first read and checked after capture; the tolerated lists name real members."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.megakernels import binding as mb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class Pool:
    uniform_fp8 = False
    request_window = None

    def __init__(self):
        self.keys = [torch.zeros(4, 8) for _ in range(2)]

    def get_swa_key_buffer_radix(self, layer_id):
        return self.keys[layer_id]

    def get_swa_key_page_size(self):
        return 128

    def get_swa_key_bytes_per_token(self):
        return 576

    def get_extra_key_buffer(self, layer_id):
        return None

    def get_extra_key_page_size(self, layer_id):
        return 64 + layer_id

    def get_extra_key_bytes_per_token(self, layer_id):
        return 32 + layer_id

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
    def _view(self, binding, attn_backend):
        with patch(
            "sglang.srt.model_executor.forward_context.get_attn_backend",
            return_value=attn_backend,
        ):
            return binding.forward_view()

    def test_the_view_is_the_running_forward_under_sglangs_names(self):
        pool = Pool()
        view = self._view(mb.MegakernelBinding(model=torch.nn.Module()), backend(pool))
        self.assertEqual(view.metadata.core_attn_metadata, "core")
        self.assertIs(view.kv_pool.pool, pool)
        self.assertIs(view.kv_pool.get_swa_key_buffer_radix(1), pool.keys[1])
        self.assertEqual(view.kv_pool.get_extra_key_page_size(1), 65)
        self.assertEqual(view.swa_out_cache_loc(None), "swa_loc")
        self.assertEqual(view.attention(), "attention")

    def test_a_replaced_pool_or_history_fails_assert_live_by_name(self):
        model = torch.nn.Module()
        model.engram_hasher = SimpleNamespace(history=torch.zeros(5, 3))
        binding = mb.MegakernelBinding(model=model)
        attn_backend = backend(Pool())
        self._view(binding, attn_backend)
        self.assertIs(binding.engram_history(), model.engram_hasher.history)
        with patch(
            "sglang.srt.model_executor.forward_context.get_attn_backend",
            return_value=attn_backend,
        ):
            binding.assert_live()
            model.engram_hasher.history = torch.zeros(9, 3)
            with self.assertRaisesRegex(RuntimeError, "engram_history was reallocated"):
                binding.assert_live()
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        self._view(binding, backend(Pool()))
        with patch(
            "sglang.srt.model_executor.forward_context.get_attn_backend",
            return_value=backend(Pool()),
        ):
            with self.assertRaisesRegex(RuntimeError, "kv_pool was reallocated"):
                binding.assert_live()

    def test_a_model_without_engram_reads_none_and_seals_nothing(self):
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        self.assertIsNone(binding.engram_history())
        self.assertEqual(binding._sealed, {})

    def test_every_tolerated_member_exists(self):
        """A plugin pins these names; renaming one here must fail first."""
        view = self._view(
            mb.MegakernelBinding(model=torch.nn.Module()), backend(Pool())
        )
        owners = {"ForwardView": view, "KvPoolView": view.kv_pool}
        for name in mb.TOLERATED_CALLABLES:
            owner, _, attr = name.partition(".")
            if owner in owners:
                self.assertTrue(hasattr(owners[owner], attr), name)


if __name__ == "__main__":
    unittest.main()
