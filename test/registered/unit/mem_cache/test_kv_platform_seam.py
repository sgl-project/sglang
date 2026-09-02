"""CPU-only unit tests for the platform KV seams.

A platform hands core a pool *class* (``SRTPlatform.get_kv_pool_cls``) and a
paged-allocator class (``get_paged_allocator_cls``); core constructs both at
the same sites, with the same keyword arguments, as the in-tree defaults.
These pin that every in-tree construction site honors the resolved class,
that a platform with no opinion gets the in-tree default rather than an
error (the in-tree pool allocates on ``device`` and is correct on any torch
device), that block-scaled and sparse pools without a seam reject an
out-of-tree platform loudly, and that the torch-native allocator path a
Triton-less platform falls back to computes the same page-aligned slots.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache import allocation
from sglang.srt.mem_cache.allocator.paged import (
    PagedTokenToKVPoolAllocator,
    alloc_decode_naive,
)
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MHATokenToKVPoolMXFP8,
    MLATokenToKVPool,
)
from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_CONFIGURATOR_MODULE = "sglang.srt.mem_cache.kv_cache_configurator"
_PAGED_MODULE = "sglang.srt.mem_cache.allocator.paged"


def _fake_platform(*, out_of_tree=False, pool_cls=None, allocator_cls=None):
    platform = SimpleNamespace()
    platform.device_name = "oot" if out_of_tree else "cuda"
    platform.is_out_of_tree = MagicMock(return_value=out_of_tree)
    platform.get_kv_pool_cls = MagicMock(return_value=pool_cls)
    platform.get_paged_allocator_cls = MagicMock(return_value=allocator_cls)
    platform.capabilities = PlatformCapabilities()
    return platform


def _fake_configurator(
    *,
    use_mla_backend=False,
    is_hybrid_swa=False,
    kv_cache_dtype_str=None,
    page_size=1,
):
    fake = SimpleNamespace()
    fake.use_mla_backend = use_mla_backend
    fake.mambaish_config = None
    fake.is_hybrid_swa = is_hybrid_swa
    fake.is_hybrid_swa_compress = False
    fake.is_hybrid_swa_mtp_draft = False
    fake.draft_swa_full_capacity = False
    fake.is_draft_worker = False
    fake.layer_info = SimpleNamespace(
        num_effective_layers=2, start_layer=0, end_layer=2
    )
    fake.pool_page_size = page_size
    fake.page_size = page_size
    fake.kv_cache_dtype = torch.float16
    fake.kv_cache_dtype_str = kv_cache_dtype_str
    fake.model_dtype = torch.float16
    fake.device = "cpu"
    fake.post_capture_kv_active = False
    fake.server_args = object()
    fake.hybrid_gdn_config = None
    fake.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(),
        get_num_kv_heads=lambda tp, dcp=1: 4,
        head_dim=64,
        v_head_dim=64,
        kv_lora_rank=8,
        qk_rope_head_dim=16,
        swa_kv_lora_rank=8,
        swa_qk_rope_head_dim=16,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
    )
    for name in (
        "_page_major_enabled",
        "_resolve_kv_pool_class",
        "_hybrid_full_attention_pool_class",
    ):
        setattr(fake, name, getattr(KVCacheConfigurator, name).__get__(fake))
    return fake


@contextlib.contextmanager
def _runtime_context(*, attention_backend="fa3", page_size=1, dcp_enabled=False):
    with (
        patch(
            f"{_CONFIGURATOR_MODULE}.get_exec",
            return_value=SimpleNamespace(
                features=SimpleNamespace(enable_memory_saver=False),
                kernel=SimpleNamespace(attention_backend=attention_backend),
            ),
        ),
        patch(
            f"{_CONFIGURATOR_MODULE}.get_spec",
            return_value=SimpleNamespace(speculative_algorithm=None),
        ),
        patch(
            f"{_CONFIGURATOR_MODULE}.get_parallel",
            return_value=SimpleNamespace(
                attn_tp_size=1, attn_dcp_size=1, dcp_enabled=dcp_enabled
            ),
        ),
        patch(
            f"{_CONFIGURATOR_MODULE}.get_schedule",
            return_value=SimpleNamespace(
                page_size=page_size, prefill_only_disable_kv_cache=False
            ),
        ),
        patch(
            f"{_CONFIGURATOR_MODULE}.get_disagg",
            return_value=SimpleNamespace(
                disaggregation_mode="null", enable_pdmux=False
            ),
        ),
        patch(
            f"{_CONFIGURATOR_MODULE}.get_memory",
            return_value=SimpleNamespace(
                enable_page_major_kv_layout=False,
                enable_unified_memory=False,
                enable_hisparse=False,
            ),
        ),
        patch(f"{_CONFIGURATOR_MODULE}._is_npu", False),
    ):
        yield


class TestResolveKVPoolClass(CustomTestCase):
    def _resolve(self, platform, *, kind="mha", default=MHATokenToKVPool):
        with patch(f"{_CONFIGURATOR_MODULE}.current_platform", platform):
            return KVCacheConfigurator._resolve_kv_pool_class(
                SimpleNamespace(), kind=kind, default=default
            )

    def test_platform_subclass_wins(self):
        class VendorPool(MHATokenToKVPool):
            pass

        platform = _fake_platform(out_of_tree=True, pool_cls=VendorPool)
        self.assertIs(self._resolve(platform), VendorPool)
        platform.get_kv_pool_cls.assert_called_once_with(kind="mha")

    def test_class_of_the_wrong_kind_is_rejected(self):
        class VendorPool(MHATokenToKVPool):
            pass

        platform = _fake_platform(out_of_tree=True, pool_cls=VendorPool)
        with self.assertRaises(TypeError) as ctx:
            self._resolve(platform, kind="mla", default=MLATokenToKVPool)
        self.assertIn("MLATokenToKVPool", str(ctx.exception))

    def test_no_opinion_means_in_tree_default_even_out_of_tree(self):
        """The in-tree pool allocates on ``device`` and is correct on any torch
        device; an out-of-tree platform without a pool class must not be told
        to write one."""
        platform = _fake_platform(out_of_tree=True, pool_cls=None)
        self.assertIs(self._resolve(platform), MHATokenToKVPool)


class TestBuildersHonorResolvedClass(CustomTestCase):
    def test_mla_builder(self):
        Spy = MagicMock(name="MLASpy")
        fake = _fake_configurator(use_mla_backend=True)
        with _runtime_context():
            KVCacheConfigurator._build_mla_kv_pool(
                fake, max_total_num_tokens=128, mla_pool_class=Spy
            )
        Spy.assert_called_once()
        self.assertEqual(Spy.call_args.args, (128,))
        self.assertEqual(Spy.call_args.kwargs["kv_lora_rank"], 8)
        self.assertEqual(Spy.call_args.kwargs["layer_num"], 2)

    def test_dsa_builder(self):
        Spy = MagicMock(name="DSASpy")
        fake = _fake_configurator(use_mla_backend=True)
        with (
            _runtime_context(),
            patch(
                "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
                return_value=(None, None),
            ),
            patch(
                f"{_CONFIGURATOR_MODULE}._should_elide_dsa_index_k", return_value=False
            ),
            patch(
                f"{_CONFIGURATOR_MODULE}.calculate_mla_kv_cache_dim", return_value=576
            ),
            patch(f"{_CONFIGURATOR_MODULE}.get_dsa_index_head_dim", return_value=128),
            patch(f"{_CONFIGURATOR_MODULE}.current_platform", _fake_platform()),
        ):
            KVCacheConfigurator._build_dsa_kv_pool(
                fake, max_total_num_tokens=128, dsa_pool_class=Spy
            )
        Spy.assert_called_once()
        self.assertEqual(Spy.call_args.kwargs["kv_cache_dim"], 576)
        self.assertEqual(Spy.call_args.kwargs["index_head_dim"], 128)

    def test_mha_builder(self):
        Spy = MagicMock(name="MHASpy")
        fake = _fake_configurator()
        with _runtime_context():
            KVCacheConfigurator._build_mha_kv_pool(
                fake, max_total_num_tokens=128, mha_pool_class=Spy, quant_method=None
            )
        Spy.assert_called_once()
        self.assertEqual(Spy.call_args.kwargs["head_num"], 4)
        self.assertFalse(Spy.call_args.kwargs["post_capture_active"])

    def test_hybrid_mla_swa_builder_threads_both_classes(self):
        MlaSpy, DsaSpy = MagicMock(name="MLA"), MagicMock(name="DSA")
        fake = _fake_configurator(use_mla_backend=True, is_hybrid_swa=True)
        with (
            _runtime_context(),
            patch(f"{_CONFIGURATOR_MODULE}.SWAKVPool") as swa_pool,
            patch(
                f"{_CONFIGURATOR_MODULE}.calculate_mla_kv_cache_dim", return_value=576
            ),
            patch(f"{_CONFIGURATOR_MODULE}.get_dsa_index_head_dim", return_value=128),
        ):
            KVCacheConfigurator._build_hybrid_mla_swa_kv_pool(
                fake,
                full_max_total_num_tokens=96,
                swa_max_total_num_tokens=32,
                is_dsa_model=True,
                mla_pool_class=MlaSpy,
                dsa_pool_class=DsaSpy,
            )
        kwargs = swa_pool.call_args.kwargs
        self.assertIs(kwargs["full_kv_pool_class"], DsaSpy)
        self.assertIs(kwargs["swa_kv_pool_class"], MlaSpy)

    def test_hybrid_linear_full_attention_class_follows_the_model(self):
        A, B = MagicMock(name="MHA"), MagicMock(name="MLA")
        with _runtime_context():
            self.assertIs(
                _fake_configurator()._hybrid_full_attention_pool_class(
                    mha_pool_class=A, mla_pool_class=B
                ),
                A,
            )
            self.assertIs(
                _fake_configurator(
                    use_mla_backend=True
                )._hybrid_full_attention_pool_class(mha_pool_class=A, mla_pool_class=B),
                B,
            )
            self.assertIs(
                _fake_configurator(
                    kv_cache_dtype_str="mxfp8"
                )._hybrid_full_attention_pool_class(mha_pool_class=A, mla_pool_class=B),
                MHATokenToKVPoolMXFP8,
            )

    def test_hybrid_linear_composite_builds_the_mla_leaf_from_the_given_class(self):
        Spy = MagicMock(name="MLALeaf")
        HybridLinearKVPool(
            size=128,
            dtype=torch.float16,
            page_size=1,
            head_num=0,
            head_dim=0,
            full_attention_layer_ids=[0, 2],
            device="cpu",
            mamba_pool=MagicMock(),
            use_mla=True,
            kv_lora_rank=8,
            qk_rope_head_dim=16,
            full_kv_pool_class=Spy,
        )
        Spy.assert_called_once()
        self.assertEqual(Spy.call_args.kwargs["layer_num"], 2)
        self.assertEqual(Spy.call_args.kwargs["kv_lora_rank"], 8)


class TestPathsWithoutASeamRejectOutOfTree(CustomTestCase):
    def test_block_scaled_kv_cache(self):
        fake = _fake_configurator(kv_cache_dtype_str="mxfp8")
        platform = _fake_platform(out_of_tree=True)
        with (
            _runtime_context(),
            patch(f"{_CONFIGURATOR_MODULE}.current_platform", platform),
            self.assertRaises(NotImplementedError) as ctx,
        ):
            KVCacheConfigurator._build_token_to_kv_pool(
                fake,
                sizes=SimpleNamespace(max_total_num_tokens=128),
                is_dsa_model=False,
                is_dsv4_model=False,
                req_to_token_pool=MagicMock(),
            )
        self.assertIn("mxfp8", str(ctx.exception))


_SIZES = SimpleNamespace(
    max_total_num_tokens=64,
    full_max_total_num_tokens=48,
    swa_max_total_num_tokens=16,
)


class TestAllocatorSelection(CustomTestCase):
    def _build(self, fake, platform, *, page_size, is_hybrid_swa=False):
        fake.is_hybrid_swa = is_hybrid_swa
        with (
            _runtime_context(page_size=page_size),
            patch(f"{_CONFIGURATOR_MODULE}.current_platform", platform),
            patch(f"{_PAGED_MODULE}.current_platform", platform),
        ):
            return KVCacheConfigurator._build_token_to_kv_pool_allocator(
                fake,
                sizes=_SIZES,
                token_to_kv_pool=MagicMock(),
                is_dsv4_model=False,
                req_to_token_pool=MagicMock(spec=[]),
                token_to_kv_pool_allocator=None,
            )

    def test_page_size_one_uses_the_flat_allocator_even_with_a_platform_class(self):
        """A platform's *paged* allocator class is for paged mode; at page size
        1 core picks the flat allocator, on every platform, so a vendor need
        not re-implement the flat fast path inside its paged class."""
        platform = _fake_platform(out_of_tree=True, allocator_cls=MagicMock())
        allocator = self._build(_fake_configurator(), platform, page_size=1)
        self.assertIsInstance(allocator, TokenToKVPoolAllocator)
        platform.get_paged_allocator_cls.assert_not_called()

    def test_paged_site_honors_the_platform_class(self):
        Spy = MagicMock(name="VendorPagedAllocator")
        platform = _fake_platform(out_of_tree=True, allocator_cls=Spy)
        allocator = self._build(
            _fake_configurator(page_size=16), platform, page_size=16
        )
        self.assertIs(allocator, Spy.return_value)
        self.assertEqual(Spy.call_args.kwargs["page_size"], 16)

    def test_paged_site_defaults_when_the_platform_has_no_opinion(self):
        """The in-tree paged allocator falls back to torch-native kernels when the
        platform lacks Triton, so an out-of-tree platform without an allocator
        class gets it instead of an error."""
        platform = _fake_platform(out_of_tree=True, allocator_cls=None)
        allocator = self._build(
            _fake_configurator(page_size=16), platform, page_size=16
        )
        self.assertIsInstance(allocator, PagedTokenToKVPoolAllocator)
        self.assertFalse(allocator.use_triton_kernels)


class TestTorchNativePagedAllocation(CustomTestCase):
    def _allocator(self, *, size=64, page_size=4):
        platform = _fake_platform(out_of_tree=True)
        with patch(f"{_PAGED_MODULE}.current_platform", platform):
            return PagedTokenToKVPoolAllocator(
                size, page_size, torch.float16, "cpu", MagicMock(), need_sort=False
            )

    def test_alloc_decode_naive_places_boundary_requests_on_fresh_pages(self):
        """Post-decode seq_len s starts a page iff (s - 1) % page_size == 0; those
        requests take free pages in batch order, the others take last_loc + 1."""
        free_pages = torch.tensor([7, 9, 11], dtype=torch.int64)
        out = torch.empty(3, dtype=torch.int64)
        alloc_decode_naive(
            torch.tensor([5, 8, 9]),
            torch.tensor([3, 22, 35]),
            free_pages,
            out,
            page_size=4,
            num_new_pages=2,
        )
        self.assertEqual(out.tolist(), [28, 23, 36])

    def test_alloc_extend_then_decode_keep_slots_page_aligned(self):
        allocator = self._allocator()
        self.assertFalse(allocator.use_triton_kernels)
        out = allocator.alloc_extend(
            prefix_lens=torch.tensor([0]),
            prefix_lens_cpu=torch.tensor([0]),
            seq_lens=torch.tensor([6]),
            seq_lens_cpu=torch.tensor([6]),
            last_loc=torch.tensor([-1]),
            extend_num_tokens=6,
        )
        self.assertEqual((out // 4).tolist(), [1, 1, 1, 1, 2, 2])
        self.assertEqual((out % 4).tolist(), [0, 1, 2, 3, 0, 1])
        self.assertEqual(len(allocator.free_pages), 16 - 2)

        same_page = allocator.alloc_decode(
            seq_lens=torch.tensor([7]),
            seq_lens_cpu=torch.tensor([7]),
            last_loc=out[-1:],
        )
        self.assertEqual(same_page.tolist(), [out[-1].item() + 1])
        self.assertEqual(len(allocator.free_pages), 16 - 2)

        new_page = allocator.alloc_decode(
            seq_lens=torch.tensor([9]),
            seq_lens_cpu=torch.tensor([9]),
            last_loc=torch.tensor([11]),
        )
        self.assertEqual(new_page.tolist(), [3 * 4])
        self.assertEqual(len(allocator.free_pages), 16 - 3)


class TestSupportTritonFollowsThePlatform(CustomTestCase):
    def test_platform_without_triton_disables_the_triton_writers(self):
        """Backend-keyed alone, ``support_triton`` was True for every backend
        an out-of-tree device could name, sending it into Triton kernels."""
        from sglang.srt.utils import common

        without = SimpleNamespace(capabilities=PlatformCapabilities())
        with_triton = SimpleNamespace(
            capabilities=PlatformCapabilities(supports_triton=True)
        )
        with patch.object(common, "current_platform", without):
            self.assertFalse(common.support_triton("fa3"))
        with patch.object(common, "current_platform", with_triton):
            self.assertTrue(common.support_triton("fa3"))
            self.assertFalse(common.support_triton("torch_native"))


class TestAllocForDecodeDispatch(CustomTestCase):
    def test_registered_device_type_wins_over_the_default(self):
        spy = MagicMock(return_value="slots")
        batch = SimpleNamespace(device="cpu")
        allocation.ALLOC_FOR_DECODE_FUNCS["cpu"] = spy
        try:
            self.assertEqual(
                allocation.alloc_for_decode(batch, token_per_req=1), "slots"
            )
        finally:
            del allocation.ALLOC_FOR_DECODE_FUNCS["cpu"]
        spy.assert_called_once_with(batch, 1)


class TestKvBufferDescsDeclareTheLayout(CustomTestCase):
    def _descs(self, *, layout, k_shape, v_shape, size=60, page_size=4):
        fake = SimpleNamespace(
            store_dtype=torch.float16,
            size=size,
            page_size=page_size,
            layer_num=1,
            kv_cache_layout=layout,
            k_buffer=[torch.zeros(k_shape, dtype=torch.float16)],
            v_buffer=[torch.zeros(v_shape, dtype=torch.float16)],
        )
        fake._kv_tokens_per_row = MHATokenToKVPool._kv_tokens_per_row.__get__(fake)
        return MHATokenToKVPool._build_kv_buffer_descs(fake)

    def test_token_major_and_page_major_rows(self):
        nhd = self._descs(layout="nhd", k_shape=(64, 4, 8), v_shape=(64, 4, 8))
        self.assertEqual(nhd[0].tokens_per_row, 1)
        self.assertEqual(nhd[0].row_bytes, 4 * 8 * 2)
        hnd = self._descs(layout="hnd", k_shape=(16, 4, 4, 8), v_shape=(16, 4, 4, 8))
        self.assertEqual(hnd[0].tokens_per_row, 4)
        self.assertEqual(hnd[0].item_len_bytes(4), 4 * 4 * 8 * 2)

    def test_head_leading_layout_is_rejected_instead_of_inferred(self):
        """A [heads, slots, dim] buffer used to be described as slot-major with
        one token per row, giving PD transfer and post-capture backing a
        span quadratic in the pool size."""
        with self.assertRaises(ValueError) as ctx:
            self._descs(layout="nhd", k_shape=(4, 64, 8), v_shape=(4, 64, 8))
        self.assertIn("_build_kv_buffer_descs", str(ctx.exception))


class TestPlatformInterfaceDefaults(CustomTestCase):
    def test_every_kv_hook_defaults_to_no_opinion(self):
        platform = SRTPlatform()
        self.assertIsNone(platform.get_kv_pool_cls(kind="mha"))
        self.assertIsNone(platform.get_paged_allocator_cls())
        self.assertIsNone(platform.get_graph_runner_cls())
        self.assertIsNone(platform.get_piecewise_backend_cls())
        self.assertEqual(platform.capabilities, PlatformCapabilities())


if __name__ == "__main__":
    unittest.main()
