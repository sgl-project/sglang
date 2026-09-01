"""CPU-only unit tests for the platform KV seam.

Pins the resolution order of KV pool construction:
SRTPlatform.build_kv_pool(request) first, then the deprecated
get_*_kv_pool_cls class hooks (out-of-tree platforms only), then the
in-tree defaults — both for standalone pools
(KVCacheConfigurator._build_platform_kv_pool) and for the full-attention
leaf of hybrid-linear composites (_build_platform_hybrid_full_kv_pool).
Also pins the base-class contract that every hook defaults to None
("no platform opinion"), which is what keeps in-tree CUDA/ROCm behavior
untouched by the seam.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.kv_pool_request import KVPoolRequest
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_CONFIGURATOR_MODULE = "sglang.srt.mem_cache.kv_cache_configurator"


def _make_request(**overrides) -> KVPoolRequest:
    fields = dict(
        kind="mha",
        size=128,
        page_size=1,
        dtype=torch.float16,
        device="cpu",
        layer_num=2,
        start_layer=0,
        end_layer=2,
        enable_memory_saver=False,
        head_num=4,
        head_dim=64,
    )
    fields.update(overrides)
    return KVPoolRequest(**fields)


def _fake_platform(*, pool=None, out_of_tree=False, legacy_cls=None):
    platform = SimpleNamespace()
    platform.device_name = "oot" if out_of_tree else "cuda"
    platform.build_kv_pool = MagicMock(return_value=pool)
    platform.is_out_of_tree = MagicMock(return_value=out_of_tree)
    platform.get_mha_kv_pool_cls = MagicMock(return_value=legacy_cls)
    platform.get_mla_kv_pool_cls = MagicMock(return_value=legacy_cls)
    platform.get_dsa_kv_pool_cls = MagicMock(return_value=legacy_cls)
    platform.get_paged_allocator_cls = MagicMock(return_value=None)
    platform.get_graph_runner_cls = MagicMock(return_value=None)
    platform.get_piecewise_backend_cls = MagicMock(return_value=None)
    return platform


def _fake_configurator(
    *,
    mambaish_config=None,
    use_mla_backend=False,
    is_hybrid_swa=False,
    page_major=False,
    kv_cache_dtype_str=None,
):
    """A configurator double for the seam methods, bound to the real logic
    under test and MagicMocks for the heavyweight collaborators."""
    fake = SimpleNamespace()
    fake.mambaish_config = mambaish_config
    fake.use_mla_backend = use_mla_backend
    fake.layer_info = SimpleNamespace(
        num_effective_layers=2, start_layer=0, end_layer=2
    )
    fake.pool_page_size = 1
    fake.kv_cache_dtype = torch.float16
    fake.kv_cache_dtype_str = kv_cache_dtype_str
    fake.device = "cpu"
    fake.post_capture_kv_active = False
    fake.is_hybrid_swa = is_hybrid_swa
    fake.is_draft_worker = False
    fake.server_args = object()
    fake.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(),
        get_num_kv_heads=lambda tp, dcp: 4,
        head_dim=64,
        v_head_dim=64,
        kv_lora_rank=8,
        qk_rope_head_dim=16,
    )
    for name in ("_kv_pool_kind", "_make_kv_pool_request", "_page_major_applies"):
        setattr(fake, name, getattr(KVCacheConfigurator, name).__get__(fake))
    fake._page_major_enabled = lambda: page_major
    fake._build_legacy_oot_kv_pool = MagicMock(return_value=None)
    return fake


@contextlib.contextmanager
def _runtime_context(*, attention_backend="fa3"):
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
            return_value=SimpleNamespace(attn_tp_size=1, attn_dcp_size=1),
        ),
    ):
        yield


_SIZES = SimpleNamespace(
    max_total_num_tokens=128,
    full_max_total_num_tokens=96,
    swa_max_total_num_tokens=32,
)


class TestPlatformInterfaceDefaults(CustomTestCase):
    def test_every_kv_hook_defaults_to_no_opinion(self):
        platform = SRTPlatform()
        self.assertIsNone(platform.build_kv_pool(request=_make_request()))
        self.assertIsNone(platform.get_mha_kv_pool_cls())
        self.assertIsNone(platform.get_mla_kv_pool_cls())
        self.assertIsNone(platform.get_dsa_kv_pool_cls())
        self.assertIsNone(platform.get_paged_allocator_cls())
        self.assertIsNone(platform.get_graph_runner_cls())
        self.assertIsNone(platform.get_piecewise_backend_cls())


class TestKVPoolKind(CustomTestCase):
    def test_kind_mapping(self):
        fake = _fake_configurator(use_mla_backend=True)
        self.assertEqual(fake._kv_pool_kind(is_dsa_model=True), "dsa")
        self.assertEqual(fake._kv_pool_kind(is_dsa_model=False), "mla")
        fake = _fake_configurator(use_mla_backend=False)
        self.assertEqual(fake._kv_pool_kind(is_dsa_model=False), "mha")


class TestBuildPlatformKVPool(CustomTestCase):
    def _run(self, fake, platform, *, is_dsa_model=False):
        with (
            patch(f"{_CONFIGURATOR_MODULE}.current_platform", platform),
            _runtime_context(),
        ):
            return KVCacheConfigurator._build_platform_kv_pool(
                fake, sizes=_SIZES, is_dsa_model=is_dsa_model
            )

    def test_platform_pool_wins(self):
        pool = object()
        fake = _fake_configurator()
        platform = _fake_platform(pool=pool, out_of_tree=True)
        self.assertIs(self._run(fake, platform), pool)
        fake._build_legacy_oot_kv_pool.assert_not_called()

    def test_in_tree_platform_falls_through(self):
        fake = _fake_configurator()
        platform = _fake_platform(pool=None, out_of_tree=False)
        self.assertIsNone(self._run(fake, platform))
        fake._build_legacy_oot_kv_pool.assert_not_called()

    def test_oot_platform_falls_back_to_legacy_hooks(self):
        legacy_pool = object()
        fake = _fake_configurator()
        fake._build_legacy_oot_kv_pool = MagicMock(return_value=legacy_pool)
        platform = _fake_platform(pool=None, out_of_tree=True)
        self.assertIs(self._run(fake, platform), legacy_pool)
        fake._build_legacy_oot_kv_pool.assert_called_once_with(
            kind="mha", max_total_num_tokens=128, is_dsa_model=False
        )

    def test_mambaish_models_skip_the_standalone_seam(self):
        fake = _fake_configurator(mambaish_config=object())
        platform = _fake_platform(pool=object(), out_of_tree=True)
        self.assertIsNone(self._run(fake, platform))
        platform.build_kv_pool.assert_not_called()

    def test_oot_platform_with_no_opinion_at_all_raises(self):
        """Falling through to the in-tree default hands a device main does not
        know about the CUDA pool; the actionable error must survive."""
        fake = _fake_configurator()
        platform = _fake_platform(pool=None, out_of_tree=True)
        with self.assertRaises(NotImplementedError) as ctx:
            self._run(fake, platform)
        self.assertIn("build_kv_pool()", str(ctx.exception))

    def test_dsa_request_is_buildable(self):
        """DSA-indexed models must reach the seam without raising: the seam is
        consulted on every platform, so a broken DSA request path took down
        DeepSeek-V3.2 startup everywhere, not just out-of-tree."""
        fake = _fake_configurator(use_mla_backend=True)
        platform = _fake_platform(pool=object(), out_of_tree=False)
        with patch(f"{_CONFIGURATOR_MODULE}.get_dsa_index_head_dim", return_value=128):
            self._run(fake, platform, is_dsa_model=True)
        request = platform.build_kv_pool.call_args.kwargs["request"]
        self.assertEqual(request.kind, "dsa")
        self.assertEqual(request.index_head_dim, 128)
        self.assertIsNotNone(request.kv_cache_dim)

    def test_hybrid_swa_request_carries_both_pool_sizes(self):
        """A hybrid-SWA platform sizing its full pool from `size` would build
        the combined budget; the full side is `full_size`."""
        fake = _fake_configurator(is_hybrid_swa=True)
        platform = _fake_platform(pool=object(), out_of_tree=False)
        self._run(fake, platform)
        request = platform.build_kv_pool.call_args.kwargs["request"]
        self.assertTrue(request.is_hybrid_swa)
        self.assertEqual(request.size, 128)
        self.assertEqual(request.full_size, 96)
        self.assertEqual(request.swa_size, 32)


class TestLegacyOOTBuilders(CustomTestCase):
    def test_missing_legacy_hook_means_in_tree_default(self):
        platform = _fake_platform(legacy_cls=None)
        with patch(f"{_CONFIGURATOR_MODULE}.current_platform", platform):
            self.assertIsNone(
                KVCacheConfigurator._build_oot_mha_kv_pool(
                    None, max_total_num_tokens=128
                )
            )


class TestHybridFullAttentionLeaf(CustomTestCase):
    def _run(self, fake, platform):
        with (
            patch(f"{_CONFIGURATOR_MODULE}.current_platform", platform),
            _runtime_context(),
        ):
            return KVCacheConfigurator._build_platform_hybrid_full_kv_pool(
                fake,
                max_total_num_tokens=128,
                layer_num=2,
                quant_method=None,
            )

    def test_platform_leaf_pool_wins(self):
        leaf = object()
        fake = _fake_configurator()
        platform = _fake_platform(pool=leaf, out_of_tree=True)
        self.assertIs(self._run(fake, platform), leaf)

    def test_oot_legacy_class_is_constructed_with_leaf_dims(self):
        legacy_cls = MagicMock(name="OOTMHAPool")
        fake = _fake_configurator()
        platform = _fake_platform(pool=None, out_of_tree=True, legacy_cls=legacy_cls)
        result = self._run(fake, platform)
        self.assertIs(result, legacy_cls.return_value)
        legacy_cls.assert_called_once_with(
            size=128,
            page_size=1,
            dtype=torch.float16,
            head_num=4,
            head_dim=64,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            enable_kv_cache_copy=False,
        )

    def test_in_tree_platform_defers_to_composite_default(self):
        fake = _fake_configurator()
        platform = _fake_platform(pool=None, out_of_tree=False)
        self.assertIsNone(self._run(fake, platform))

    def test_oot_platform_with_no_opinion_at_all_raises(self):
        fake = _fake_configurator()
        platform = _fake_platform(pool=None, out_of_tree=True)
        with self.assertRaises(NotImplementedError):
            self._run(fake, platform)


class TestKVPoolRequestFields(CustomTestCase):
    def _request(self, *, kind, **kwargs):
        fake = _fake_configurator(**kwargs)
        with _runtime_context(), patch(
            f"{_CONFIGURATOR_MODULE}.get_dsa_index_head_dim", return_value=128
        ):
            return fake._make_kv_pool_request(
                kind=kind, size=128, layer_num=2, start_layer=0, end_layer=2
            )

    def test_page_major_only_reaches_plain_mha(self):
        page_major = dict(page_major=True)
        self.assertEqual(self._request(kind="mha", **page_major).layout, "page_major")
        self.assertEqual(self._request(kind="mla", **page_major).layout, "contiguous")
        self.assertEqual(self._request(kind="dsa", **page_major).layout, "contiguous")
        self.assertEqual(
            self._request(
                kind="mha", page_major=True, kv_cache_dtype_str="mxfp8"
            ).layout,
            "contiguous",
        )
        self.assertEqual(self._request(kind="mha").layout, "contiguous")

    def test_unset_kv_cache_dtype_str_is_empty_not_none(self):
        self.assertEqual(self._request(kind="mha").kv_cache_dtype_str, "")

    def test_index_head_dim_is_dsa_only(self):
        self.assertIsNone(self._request(kind="mla").index_head_dim)
        self.assertEqual(self._request(kind="dsa").index_head_dim, 128)


class TestPlatformAllocatorPreemption(CustomTestCase):
    def _run(self, *, platform, pool):
        with patch(f"{_CONFIGURATOR_MODULE}.current_platform", platform):
            return KVCacheConfigurator._platform_preempting_allocator_cls(
                SimpleNamespace(), token_to_kv_pool=pool
            )

    def test_flat_platform_pool_is_preempted(self):
        allocator_cls = MagicMock(name="OOTAllocator")
        platform = _fake_platform(out_of_tree=True)
        platform.get_paged_allocator_cls = MagicMock(return_value=allocator_cls)
        self.assertIs(self._run(platform=platform, pool=object()), allocator_cls)

    def test_swa_platform_pool_defers_to_the_composite_allocator(self):
        """An SWA pool paired with a flat paged allocator loses the sliding
        sub-pool; the composite must win so it can consult the same hook."""
        allocator_cls = MagicMock(name="OOTAllocator")
        platform = _fake_platform(out_of_tree=True)
        platform.get_paged_allocator_cls = MagicMock(return_value=allocator_cls)
        swa_pool = MagicMock(spec=BaseSWAKVPool)
        self.assertIsNone(self._run(platform=platform, pool=swa_pool))

    def test_in_tree_platform_never_preempts(self):
        platform = _fake_platform(out_of_tree=False)
        platform.get_paged_allocator_cls = MagicMock(return_value=MagicMock())
        self.assertIsNone(self._run(platform=platform, pool=object()))

    def test_oot_platform_without_the_hook_raises(self):
        platform = _fake_platform(out_of_tree=True)
        with self.assertRaises(NotImplementedError) as ctx:
            self._run(platform=platform, pool=object())
        self.assertIn("get_paged_allocator_cls()", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
