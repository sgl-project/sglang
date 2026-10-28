import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

with patch.dict(
    sys.modules,
    {
        module: MagicMock()
        for module in (
            "sgl_kernel",
            "sgl_kernel.quantization",
            "sgl_kernel.scalar_type",
        )
    },
):
    from sglang.srt.configs.model_config import AttentionArch
    from sglang.srt.layers.attention import flashattention_backend
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionBackend,
        FlashAttentionMetadata,
    )
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
        UnquantizedKVCacheMethod,
    )
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _backend():
    backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
    backend.page_size = 1
    backend.kv_cache_dtype = torch.float16
    backend.kv_cache_dtype_str = "float8_e4m3fn"
    backend.kv_cache_is_mxfp8 = False
    backend.fa_impl_ver = 3
    backend.num_splits = 4
    return backend


def _mha_pool(k_buffer, v_buffer, *, page_size):
    pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.page_size = page_size
    pool.start_layer = 0
    pool.layer_transfer_counter = None
    pool.quant_method = UnquantizedKVCacheMethod()
    pool.dtype = pool.store_dtype = k_buffer.dtype
    pool.k_buffer = [k_buffer]
    pool.v_buffer = [v_buffer]
    return pool


def _layer():
    return SimpleNamespace(
        layer_id=0,
        tp_k_head_num=2,
        tp_v_head_num=2,
        head_dim=16,
        v_head_dim=16,
    )


class TestFlashAttentionPagedMHA(unittest.TestCase):
    def test_get_paged_mha_kv_cache_supports_head_groups(self):
        backend = _backend()
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=Mock(
                return_value=(
                    torch.empty(8, 2, 16),
                    torch.empty(8, 2, 16),
                )
            )
        )
        layer = SimpleNamespace(
            layer_id=3,
            tp_k_head_num=2,
            tp_v_head_num=2,
            head_dim=16,
            v_head_dim=16,
        )

        key_cache, value_cache = backend.get_paged_mha_kv_cache(
            layer,
            head_group_num=2,
        )

        self.assertEqual(key_cache.shape, (16, 1, 1, 16))
        self.assertEqual(value_cache.shape, (16, 1, 1, 16))

    def test_get_paged_mha_kv_cache_refuses_head_groups_on_a_strided_pool(self):
        # A unified pool's slots are a whole ENTRY apart, and no view of the
        # head dim can regroup pages over that stride: refuse it loudly.
        layer = _layer()
        H, D, slots = layer.tp_k_head_num, layer.head_dim, 8
        entry = 64 * H * D  # every layer's K and V share the slot
        backing = torch.zeros(slots * entry)
        strided = backing.as_strided((slots, H, D), (entry, D, 1))
        self.assertFalse(strided.is_contiguous())

        backend = _backend()
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=Mock(return_value=(strided, strided))
        )

        with self.assertRaises(AssertionError) as caught:
            backend.get_paged_mha_kv_cache(layer, head_group_num=2)
        self.assertIn("contiguous paged KV", str(caught.exception))

        key_cache, _ = backend.get_paged_mha_kv_cache(layer)
        self.assertEqual(key_cache.shape, (slots, 1, H, D))

    def test_get_paged_mha_kv_cache_pages_by_the_backend_page_size(self):
        # Pages follow the backend's page size even when the pool pages at a
        # multiple of it, as a draft pool does under decode context parallelism.
        backend = _backend()
        backend.page_size = 2
        k = torch.arange(8 * 2 * 16, dtype=torch.float16).view(8, 2, 16)
        backend.token_to_kv_pool = _mha_pool(k, k + 1, page_size=4)

        key_cache, value_cache = backend.get_paged_mha_kv_cache(_layer())

        self.assertEqual(key_cache.shape, (4, 2, 2, 16))
        self.assertEqual(value_cache.shape, (4, 2, 2, 16))
        self.assertTrue(torch.equal(key_cache[1, 0], k[2]))

    def test_get_paged_mha_kv_cache_reads_the_hnd_pool_by_head_and_token(self):
        # An HND pool hands over [num_pages, heads, page_size, head_dim]
        # buffers; the paged cache must index them as [page, token, head].
        backend = _backend()
        backend.page_size = 4
        k = torch.arange(3 * 2 * 4 * 16, dtype=torch.float16).view(3, 2, 4, 16)
        backend.token_to_kv_pool = _mha_pool(k, k + 1, page_size=4)

        key_cache, _ = backend.get_paged_mha_kv_cache(_layer())

        self.assertEqual(key_cache.shape, (3, 4, 2, 16))
        self.assertEqual(key_cache.data_ptr(), k.data_ptr())
        for page in range(3):
            for head in range(2):
                for token in range(4):
                    self.assertTrue(
                        torch.equal(key_cache[page, token, head], k[page, head, token])
                    )

    def test_prepare_paged_mha_query_reuses_fa_scaling_policy(self):
        backend = _backend()
        layer = SimpleNamespace(
            head_dim=16,
            k_scale=torch.tensor(2.0),
            v_scale=torch.tensor(4.0),
        )
        q = torch.ones(2, 16, dtype=torch.bfloat16)

        q, _, _, k_descale, v_descale = backend.prepare_paged_mha_query(
            q,
            None,
            None,
            layer,
            logical_batch_size=2,
            kv_head_num=1,
            is_prefill=True,
            ver=3,
        )

        self.assertEqual(q.dtype, torch.float16)
        self.assertEqual(k_descale.shape, (2, 1))
        self.assertEqual(v_descale.shape, (2, 1))


def _fa3_backend():
    backend = _backend()
    backend.fa_impl_ver = 3
    backend.use_mla = False
    return backend


def _gemma4_layer(*, head_dim, sliding_window_size):
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=4,
        tp_k_head_num=2,
        tp_v_head_num=2,
        head_dim=head_dim,
        v_head_dim=head_dim,
        sliding_window_size=sliding_window_size,
        is_cross_attention=False,
        attn_type=None,
        scaling=1.0,
        logit_cap=0.0,
        k_scale=None,
        v_scale=None,
    )


class TestFlashAttentionFa3UpgradesToFa4(unittest.TestCase):
    def test_ver_for_upgrades_fa3_above_the_fa3_head_dim_limit(self):
        # FA3 kernels stop at head_dim 256; the wider of K and V decides.
        backend = _fa3_backend()
        cases = [((256, 256), 3), ((257, 257), 4), ((512, 512), 4), ((128, 512), 4)]
        for (head_dim, v_head_dim), expected in cases:
            layer = SimpleNamespace(head_dim=head_dim, v_head_dim=v_head_dim)
            self.assertEqual(backend._ver_for(layer), expected, (head_dim, v_head_dim))

    def test_ver_for_sends_image_masks_to_fa4(self):
        # Image masks exist only in FA4, so a layer FA3 could run still goes
        # to FA4 when the batch carries them.
        layer = SimpleNamespace(head_dim=256, v_head_dim=256)
        self.assertEqual(_fa3_backend()._ver_for(layer, needs_fa4=True), 4)

    def test_ver_for_keeps_absorbed_mla_on_fa3(self):
        # Absorbed MLA reports head_dim 576 / v_head_dim 512, but FA3 runs it
        # natively as hdim 64 + qv 512; FA4's qv path is SM100-only.
        backend = _fa3_backend()
        backend.use_mla = True
        layer = SimpleNamespace(head_dim=576, v_head_dim=512)
        self.assertEqual(backend._ver_for(layer), 3)

    def test_ver_for_never_downgrades_fa4(self):
        backend = _backend()
        backend.fa_impl_ver = 4
        layer = SimpleNamespace(head_dim=128, v_head_dim=128)
        self.assertEqual(backend._ver_for(layer), 4)

    def test_forward_decode_dispatches_each_layer_to_its_version(self):
        # Gemma-4 on fa3: 256-dim sliding-window layers stay on FA3, 512-dim
        # full layers run FA4.
        backend = _fa3_backend()
        backend.kv_cache_dtype = torch.bfloat16
        backend.kv_cache_dtype_str = "auto"
        backend.use_mla = False
        backend.topk = 0
        backend.local_attn_builder = None
        backend.use_sliding_window_kv_pool = False
        backend.is_prefill_aware_swa = False
        backend._decode_uses_static_max_seqlen_k = False
        backend.decode_num_splits = 0
        backend.num_splits = 0
        backend.full_cg_prefill_metadata = None
        backend.forward_metadata = FlashAttentionMetadata(
            cache_seqlens_int32=torch.ones(1, dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            page_table=torch.zeros(1, 1, dtype=torch.int32),
        )
        forward_batch = SimpleNamespace(batch_size=1, spec_info=None)

        for head_dim, window, expected_ver in ((256, 1023, 3), (512, -1, 4)):
            layer = _gemma4_layer(head_dim=head_dim, sliding_window_size=window)
            backend.token_to_kv_pool = SimpleNamespace(
                get_kv_buffer=Mock(
                    return_value=(
                        torch.empty(1, 2, head_dim, dtype=torch.bfloat16),
                        torch.empty(1, 2, head_dim, dtype=torch.bfloat16),
                    )
                )
            )
            kernels = {
                ver: Mock(return_value=torch.zeros(1, 4, head_dim)) for ver in (3, 4)
            }
            backend._fa_kernels = {ver: (Mock(), k) for ver, k in kernels.items()}
            backend.forward_decode(
                torch.zeros(1, 4 * head_dim, dtype=torch.bfloat16),
                None,
                None,
                layer,
                forward_batch,
                save_kv_cache=False,
            )
            self.assertTrue(kernels[expected_ver].called, head_dim)
            self.assertFalse(kernels[4 if expected_ver == 3 else 3].called, head_dim)


def _gemma4_runner():
    # Enough of a ModelRunner for FlashAttentionBackend.__init__ on CPU, with
    # Gemma-4's head_dim 512 full-attention layers.
    req_to_token = torch.zeros(9, 64, dtype=torch.int32)
    return SimpleNamespace(
        sliding_window_size=None,
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            context_len=64,
            attention_arch=AttentionArch.MHA,
            is_local_attention_model=False,
            head_dim=512,
            hf_text_config=SimpleNamespace(
                num_attention_heads=2, attn_logit_softcapping=None
            ),
            get_num_kv_heads=lambda tp_size: 2,
        ),
        device="cpu",
        req_to_token_pool=SimpleNamespace(size=8, req_to_token=req_to_token),
        token_to_kv_pool=object(),
        token_to_kv_pool_allocator=object(),
        kv_index_translator=SimpleNamespace(),
        kv_cache_dtype=torch.bfloat16,
        kv_cache_dtype_str="auto",
        page_size=1,
        is_draft_worker=False,
        attention_chunk_size=None,
        prefill_aware_swa=False,
    )


class TestFa3UpgradesToFa4Init(unittest.TestCase):
    def test_fa4_layers_enable_batch_invariance(self):
        # fa3 with head_dim 512 layers runs them on FA4, so deterministic
        # inference must switch FA4 to batch-invariant mode at init, before
        # its first compile, as the fa4 backend does.
        fa4, batch_invariance = MagicMock(), MagicMock()
        modules = {
            "sgl_kernel.flash_attn": MagicMock(),
            "sglang.kernels.ops.attention.flash_attention_v3": MagicMock(),
            "sglang.kernels.ops.attention.flash_attention_v4": fa4,
            "sglang.kernels.ops.attention.flash_attn": MagicMock(),
            "sglang.kernels.ops.attention.flash_attn.cute": MagicMock(),
            "sglang.kernels.ops.attention.flash_attn.cute.batch_invariance": (
                batch_invariance
            ),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(
                flashattention_backend, "get_device_capability", return_value=(9, 0)
            ),
            get_context().override_server_args(enable_deterministic_inference=True),
        ):
            backend = FlashAttentionBackend(_gemma4_runner(), fa_impl_ver=3)

        batch_invariance.set_batch_invariant.assert_called_once_with(True)
        self.assertIs(backend._fa_kernels[4][1], fa4.flash_attn_with_kvcache)


if __name__ == "__main__":
    unittest.main()
