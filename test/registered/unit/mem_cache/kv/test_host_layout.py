"""Host layout construction preserves persistent rows and storage page bytes."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa
from sglang.srt.mem_cache.kv import HostPoolOptions
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _unpinned(cls):
    def build(*args, **kwargs):
        return cls(*args, **(kwargs | {"pin_memory": False}))

    return build


class TestHostLayouts(CustomTestCase):
    def test_rows_and_storage_pages_match_existing_host_layouts(self):
        for mla in (False, True):
            setup_allocator_context()
            bundle = build_unified_bundle(
                UnifiedGeometry(
                    chain="mamba",
                    mla=mla,
                    page_size=16,
                    full_layer_ids=(0, 1),
                    state_layer_ids=(2, 3),
                    full_tokens=1024,
                    swa_tokens=1024,
                )
            )
            storage = bundle.token_to_kv_pool_allocator.storage
            for name in ("full", "mamba"):
                layouts = (
                    ("page_first_direct",)
                    if name == "mamba"
                    else ("page_first_direct", "page_unified")
                )
                for layout in layouts:
                    with self.subTest(mla=mla, name=name, layout=layout):
                        options = HostPoolOptions(
                            ratio=1.0,
                            size=0,
                            page_size=16,
                            layout=layout,
                            allocator_type="default",
                            pool_label=name,
                            pin_memory=False,
                        )
                        host = storage.host_layout(name).build(options)
                        self.addCleanup(host.destroy)
                        view = storage.view(name)
                        with (
                            patch.object(
                                hpa,
                                "get_memory",
                                return_value=SimpleNamespace(
                                    hicache_ratio=1.0,
                                    hicache_size=0,
                                    hicache_mem_layout=layout,
                                ),
                            ),
                            patch.object(
                                hpa, "_get_allocator_type", return_value="default"
                            ),
                            patch.object(
                                hpa,
                                "get_mha_host_pool_cls",
                                side_effect=lambda info: _unpinned(
                                    get_mha_host_pool_cls(info)
                                ),
                            ),
                            patch.object(
                                hpa,
                                "MLATokenToKVPoolHost",
                                _unpinned(MLATokenToKVPoolHost),
                            ),
                            patch.object(
                                hpa, "MambaPoolHost", _unpinned(MambaPoolHost)
                            ),
                        ):
                            if name == "mamba":
                                (info,) = view.get_device_pool_infos()
                                previous = hpa.build_mamba_host_pool(
                                    view, info, host_size=0, layout=layout
                                )
                            else:
                                previous = hpa.build_kv_host_pool(
                                    kv_pool=view, page_size=16, pool_label=name
                                )
                        self.addCleanup(previous.destroy)
                        self.assertEqual(host.size, previous.size)
                        self.assertEqual(
                            host.storage_format_tag, previous.storage_format_tag
                        )
                        dummy = host.get_dummy_flat_data_page()
                        old_dummy = previous.get_dummy_flat_data_page()
                        self.assertEqual(
                            (dummy.shape, dummy.dtype),
                            (old_dummy.shape, old_dummy.dtype),
                        )
                        payload = (
                            (torch.arange(dummy.numel()) % 31)
                            .reshape(dummy.shape)
                            .to(dummy.dtype)
                        )
                        host.set_from_flat_data_page(0, payload)
                        previous.set_from_flat_data_page(0, payload)
                        torch.testing.assert_close(
                            host.get_data_page(0), previous.get_data_page(0)
                        )
                        torch.testing.assert_close(host.get_data_page(0), payload)

    def test_state_layout_accepts_kv_split_without_splitting_state(self):
        setup_allocator_context()
        storage = build_unified_bundle(
            UnifiedGeometry(chain="mamba", full_layer_ids=(0,), state_layer_ids=(1,))
        ).token_to_kv_pool_allocator.storage
        layout = storage.host_layout("mamba")
        host = layout.build(
            HostPoolOptions(
                ratio=1.0,
                size=0,
                page_size=16,
                layout="page_first_kv_split",
                allocator_type="default",
                pool_label="mamba",
                pin_memory=False,
            )
        )
        self.addCleanup(host.destroy)
        self.assertEqual(host.layout, "page_first_direct")
        self.assertEqual(
            host.buffer_info.temporal.data_ptr(), layout.buffer_info.temporal.data_ptr()
        )
        self.assertEqual(layout.device_layer_num, len(layout.layer_ids))


if __name__ == "__main__":
    unittest.main()
