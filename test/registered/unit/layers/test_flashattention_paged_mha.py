import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

for _module in ("sgl_kernel", "sgl_kernel.quantization", "sgl_kernel.scalar_type"):
    sys.modules.setdefault(_module, MagicMock())

from sglang.srt.layers.attention import flashattention_backend  # noqa: E402
from sglang.srt.layers.attention.flashattention_backend import (  # noqa: E402
    FlashAttentionBackend,
)
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (  # noqa: E402
    UnquantizedKVCacheMethod,
)
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind  # noqa: E402
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds  # noqa: E402
from sglang.srt.mem_cache.kv.router import PlainPoolRouter  # noqa: E402
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool  # noqa: E402
from sglang.srt.runtime_context import get_context  # noqa: E402

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


def _pool_reading(k_buffer, v_buffer):
    """A pool whose every layer reads the given K and V through its router."""
    kv = SimpleNamespace(kv=Mock(return_value=(k_buffer, v_buffer)))
    return SimpleNamespace(layer_router=SimpleNamespace(read=Mock(return_value=kv)))


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
        backend.token_to_kv_pool = _pool_reading(
            torch.empty(8, 2, 16), torch.empty(8, 2, 16)
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
        backend.token_to_kv_pool = _pool_reading(strided, strided)

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


def _cpu_runner(*, model_head_dim):
    """The least ModelRunner FlashAttentionBackend.__init__ reads up to its
    model-level FA4 decisions, on the CPU device."""
    req_to_token_pool = SimpleNamespace(
        size=4, req_to_token=torch.zeros(5, 16, dtype=torch.int32)
    )
    token_to_kv_pool = SimpleNamespace(
        id_spaces=lambda: {IdSpaceKind.FULL: IdSpace(key=(IdSpaceKind.FULL, None))}
    )
    token_to_kv_pool.layer_router = PlainPoolRouter(token_to_kv_pool, IdSpaceKind.FULL)
    allocator = SimpleNamespace(hands_out_virtual_ids=False)
    return SimpleNamespace(
        sliding_window_size=None,
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            context_len=16,
            attention_arch=flashattention_backend.AttentionArch.MHA,
            is_local_attention_model=False,
            head_dim=model_head_dim,
            hf_text_config=SimpleNamespace(
                num_attention_heads=2, attn_logit_softcapping=None
            ),
            get_num_kv_heads=lambda tp_size: 2,
        ),
        device="cpu",
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
        runner_ids=RunnerIds(
            req_to_token=req_to_token_pool.req_to_token,
            token_to_kv_pool_allocator=allocator,
            token_to_kv_pool=token_to_kv_pool,
            page_size=1,
            device="cpu",
        ),
        kv_cache_dtype=torch.bfloat16,
        kv_cache_dtype_str="auto",
        page_size=1,
        is_draft_worker=False,
        attention_chunk_size=None,
        prefill_aware_swa=False,
    )


class TestFa3UpgradeHardwareGate(unittest.TestCase):
    # FA3 itself runs on SM8x, but the FA4 kernels its wide layers need do not
    # (FA4 asserts "paged KV not supported on SM 8.0" at the first call).
    # Reject at init, with the triton pointer, instead of at the first prefill.
    def test_wide_heads_on_sm8x_are_rejected_at_init(self):
        with (
            get_context().override_server_args(),
            patch.object(
                flashattention_backend, "get_device_capability", return_value=(8, 0)
            ),
        ):
            with self.assertRaisesRegex(ValueError, "SM90.*triton") as raised:
                FlashAttentionBackend(_cpu_runner(model_head_dim=512))
        self.assertIn("head_dim 512", str(raised.exception))


def _load_fa4_kernels(*, deterministic):
    batch_invariance = MagicMock()
    modules = {
        "sglang.kernels.ops.attention.flash_attention_v4": MagicMock(),
        "sglang.kernels.ops.attention.flash_attn": MagicMock(),
        "sglang.kernels.ops.attention.flash_attn.cute": MagicMock(),
        "sglang.kernels.ops.attention.flash_attn.cute.batch_invariance": (
            batch_invariance
        ),
    }
    exec_config = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
    )
    with (
        patch.dict(sys.modules, modules),
        patch.object(flashattention_backend, "get_exec", return_value=exec_config),
    ):
        flashattention_backend._load_fa4_kernels((9, 0))
    return batch_invariance.set_batch_invariant


class TestLoadFa4Kernels(unittest.TestCase):
    # Both the fa4 backend and the fa3 backend's head_dim > 256 layers load FA4
    # through this helper, so batch invariance is set before FA4's first compile
    # on either path.
    def test_deterministic_inference_enables_batch_invariance(self):
        _load_fa4_kernels(deterministic=True).assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main()
