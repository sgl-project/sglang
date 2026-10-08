import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention import mla_kc_cache_gluon_hip as adapter


class _Storage:
    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class _FakeCache:
    def __init__(self, shape, pointer=17):
        self.dtype = torch.float8_e4m3fn
        self.shape = shape
        self.device = torch.device("cpu")
        self._pointer = pointer

    def stride(self):
        return (576, 576, 1)

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        return _Storage(self._pointer)

    def view(self, *shape):
        return self


def make_inputs(rows=1):
    query = torch.empty((rows, 12, 192), dtype=torch.bfloat16)
    latent = torch.empty((rows, 1, 512), dtype=torch.bfloat16)
    key_tail = torch.empty((rows, 1, 64), dtype=torch.bfloat16)
    batch = SimpleNamespace(
        out_cache_loc=torch.zeros(rows, dtype=torch.int64),
    )
    cache = _FakeCache((655361, 1, 576))
    value = _FakeCache((655361, 1, 512))
    pool = SimpleNamespace(
        page_size=1,
        get_key_buffer=mock.Mock(return_value=cache),
        get_value_buffer=mock.Mock(return_value=value),
    )
    attn = SimpleNamespace(
        w_kc=torch.empty((12, 512, 128), dtype=torch.bfloat16).transpose(1, 2),
        attn_mqa=SimpleNamespace(layer_id=3),
        current_attention_backend="triton",
    )
    return attn, query, latent, key_tail, batch, pool


class TestMlaKcCache(unittest.TestCase):
    def test_entrypoint_selection_is_exact(self):
        expected = {
            1: "mla_kc_cache_m1_4_8",
            2: "mla_kc_cache_m2_16",
            4: "mla_kc_cache_m1_4_8",
            8: "mla_kc_cache_m1_4_8",
            16: "mla_kc_cache_m2_16",
            32: "mla_kc_cache_m32",
            64: "mla_kc_cache_m64",
            128: "mla_kc_cache_m128",
            256: "mla_kc_cache_m256_1024_8192",
            1024: "mla_kc_cache_m256_1024_8192",
            8192: "mla_kc_cache_m256_1024_8192",
        }
        for rows, name in expected.items():
            self.assertEqual(adapter.entrypoint_name(rows), name)
        for rows in (0, 3, 12, 24, 257, 1023, 8193):
            self.assertIsNone(adapter.entrypoint_name(rows))

    def test_runtime_contract_rejects_layout_and_cache_changes(self):
        attn, query, latent, key_tail, batch, pool = make_inputs()
        with mock.patch.object(adapter, "_physical_triton_pool", return_value=pool):
            self.assertTrue(adapter.covered(attn, query, latent, key_tail, batch))
            attn.current_attention_backend = "aiter"
            self.assertFalse(adapter.covered(attn, query, latent, key_tail, batch))
            attn.current_attention_backend = "triton"
            self.assertFalse(
                adapter.covered(attn, query.float(), latent, key_tail, batch)
            )
            batch.out_cache_loc = batch.out_cache_loc.to(torch.int32)
            self.assertFalse(adapter.covered(attn, query, latent, key_tail, batch))
            batch.out_cache_loc = torch.zeros(1, dtype=torch.int64)
            pool.page_size = 16
            self.assertFalse(adapter.covered(attn, query, latent, key_tail, batch))

    def test_apply_propagates_launch_failure_and_validates_outputs(self):
        attn, query, latent, key_tail, batch, pool = make_inputs()
        context_path = "sglang.srt.model_executor.forward_context.get_token_to_kv_pool"
        with (
            mock.patch(context_path, return_value=pool),
            mock.patch.object(adapter, "run", side_effect=RuntimeError("launch")),
        ):
            with self.assertRaisesRegex(RuntimeError, "launch"):
                adapter.apply(attn, query, latent, key_tail, batch)

        qcat = torch.empty((1, 12, 576), dtype=torch.bfloat16)
        fresh = torch.empty((1, 576), dtype=torch.bfloat16)
        with (
            mock.patch(context_path, return_value=pool),
            mock.patch.object(adapter, "run", return_value=(qcat, fresh)),
        ):
            result = adapter.apply(attn, query, latent, key_tail, batch)
        self.assertIs(result[0], qcat)
        self.assertIs(result[1], fresh)
        self.assertIs(result[2], latent)

        with (
            mock.patch(context_path, return_value=pool),
            mock.patch.object(adapter, "run", return_value=(query, fresh)),
        ):
            with self.assertRaisesRegex(RuntimeError, "ABI"):
                adapter.apply(attn, query, latent, key_tail, batch)

    def test_model_hook_preserves_native_path(self):
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
            DeepseekMLARocmForwardMixin,
        )
        from sglang.srt.models.kimi_k3 import KimiK3MLAAttention

        attn = KimiK3MLAAttention.__new__(KimiK3MLAAttention)
        torch.nn.Module.__init__(attn)
        query = torch.empty((1, 12, 192), dtype=torch.bfloat16)
        latent = torch.empty((1, 1, 512), dtype=torch.bfloat16)
        key_tail = torch.empty((1, 1, 64), dtype=torch.bfloat16)
        batch = SimpleNamespace(out_cache_loc=torch.zeros(1, dtype=torch.int64))
        native_out, fused_out = object(), object()
        with (
            mock.patch.object(
                DeepseekMLARocmForwardMixin,
                "_prepare_kc_cache",
                return_value=native_out,
            ) as native,
            mock.patch.object(adapter, "covered", return_value=True),
            mock.patch.object(adapter, "apply", return_value=fused_out) as fused,
        ):
            attn._mla_gluon_kc_active = False
            self.assertIs(
                attn._prepare_kc_cache(query, latent, key_tail, batch), native_out
            )
            fused.assert_not_called()
            attn._mla_gluon_kc_active = True
            self.assertIs(
                attn._prepare_kc_cache(query, latent, key_tail, batch), fused_out
            )
            native.assert_called_once()
            fused.assert_called_once_with(attn, query, latent, key_tail, batch)

    def test_core_disables_second_cache_write_and_reuses_native_tail(self):
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
            DeepseekMLARocmForwardMixin,
        )

        qcat = torch.empty((2, 12, 576), dtype=torch.bfloat16)
        fresh = torch.empty((2, 576), dtype=torch.bfloat16)
        latent = torch.empty((2, 1, 512), dtype=torch.bfloat16)
        attended, expected = object(), object()
        attn_mqa = mock.Mock(return_value=attended)
        instance = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            attn_mqa=attn_mqa,
            _finish_absorb_rocm=mock.Mock(return_value=expected),
        )
        batch = object()
        result = DeepseekMLARocmForwardMixin.forward_absorb_rocm_core(
            instance,
            None,
            None,
            None,
            latent,
            batch,
            None,
            None,
            None,
            None,
            None,
            (qcat, fresh, latent),
        )
        self.assertIs(result, expected)
        attn_mqa.assert_called_once()
        args, kwargs = attn_mqa.call_args
        self.assertIs(args[0], qcat)
        self.assertEqual(tuple(args[1].shape), (2, 1, 576))
        self.assertEqual(args[1].data_ptr(), fresh.data_ptr())
        self.assertEqual(tuple(args[2].shape), (2, 1, 512))
        self.assertEqual(args[2].data_ptr(), latent.data_ptr())
        self.assertIs(args[3], batch)
        self.assertEqual(kwargs, {"save_kv_cache": False})
        instance._finish_absorb_rocm.assert_called_once_with(attended, batch, None)


if __name__ == "__main__":
    unittest.main()
