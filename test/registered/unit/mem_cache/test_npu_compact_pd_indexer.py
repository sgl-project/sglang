"""CPU allocation and geometry checks for opt-in NPU PD indexer elision."""

import unittest
from enum import Enum
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.utils import (
    build_kv_layer_ids,
    build_transfer_entry_pairs,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.mem_cache import kv_cache_configurator as config
from sglang.srt.mem_cache.pool_host import mla
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage
from sglang.srt.model_executor import pool_configurator
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNPUCompactPDIndexer(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            page_size=128,
            enable_hierarchical_cache=True,
            disaggregation_mode="prefill",
        )
        override.install()
        self.addCleanup(override.restore)
        self.enterContext(get_parallel().override(attn_dcp_rank=0))
        self.enterContext(patch.object(config, "_is_npu", True))
        self.enterContext(patch.object(pool_configurator, "_is_npu", True))
        self.enterContext(
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                return_value=False,
            )
        )

    def configurator(self, dtype=torch.bfloat16):
        return SimpleNamespace(
            modelslim_cache_layout=None,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["GlmMoeDsaForCausalLM"],
                    index_topk=2048,
                    index_head_dim=128,
                    indexer_types=[
                        "full",
                        "shared",
                        "shared",
                        "full",
                        "shared",
                        "shared",
                        "full",
                        "shared",
                    ],
                ),
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                index_head_dim=128,
            ),
            layer_info=SimpleNamespace(
                start_layer=0, end_layer=8, num_effective_layers=8
            ),
            server_args=SimpleNamespace(enable_hisparse=False),
            mambaish_config=None,
            is_draft_worker=False,
            kv_cache_dtype=dtype,
            device="cpu",
        )

    def build_pool(self, compact=True, dtype=torch.bfloat16):
        with envs.SGLANG_NPU_COMPACT_DSA_INDEXER.override(compact):
            return config.KVCacheConfigurator._build_ascend_mla_kv_pool(
                self.configurator(dtype), max_total_num_tokens=256, is_dsa_model=True
            )

    def enabled(
        self,
        *,
        flag=True,
        role="prefill",
        hicache=True,
        arch35=False,
        draft=False,
        pp=1,
        cp=1,
        dcp=1,
        hisparse=False,
        external=False,
    ):
        with (
            patch.object(config, "_is_npu", True),
            patch.object(envs.SGLANG_NPU_COMPACT_DSA_INDEXER, "get", return_value=flag),
            patch.object(
                config,
                "get_parallel",
                return_value=SimpleNamespace(
                    pp_size=pp, attn_cp_size=cp, attn_dcp_size=dcp
                ),
            ),
            patch.object(
                config,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode=role),
            ),
            patch.object(
                config,
                "get_memory",
                return_value=SimpleNamespace(
                    enable_hierarchical_cache=hicache,
                    enable_hisparse=hisparse,
                    enable_unified_cache_external_linker=external,
                ),
            ),
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                return_value=arch35,
            ),
        ):
            return config._should_elide_dsa_index_k(is_draft_worker=draft)

    def test_opt_in_enables_both_pd_roles(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                self.assertFalse(self.enabled(flag=False, role=role))
                self.assertTrue(self.enabled(role=role))

    def test_unsupported_layouts_keep_dense_indices(self):
        for args in (
            {"pp": 2},
            {"cp": 2},
            {"dcp": 2},
            {"arch35": True},
            {"draft": True},
            {"hisparse": True},
            {"external": True},
        ):
            with self.subTest(args=args):
                self.assertFalse(self.enabled(**args))

    def test_monolithic_elision_remains_available(self):
        self.assertTrue(self.enabled(flag=False, role="null", hicache=False))

    def test_physical_pd_entries_and_host_mapping_match_producers(self):
        override = get_context().override_server_args(page_size=128)
        override.install()
        self.addCleanup(override.restore)
        with get_parallel().override(attn_dcp_rank=0):
            pool = NPUMLATokenToKVPool(
                size=128,
                page_size=128,
                dtype=torch.bfloat16,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                layer_num=8,
                device="cpu",
                enable_memory_saver=False,
                index_head_dim=128,
                indexer_layer_ids=(0, 3, 6),
            )
        pointers, lengths, row_bytes = pool.get_contiguous_buf_infos()
        layers = pool.get_kv_layer_ids()
        self.assertEqual(len(pointers), 19)
        self.assertEqual(len(lengths), len(layers))
        self.assertEqual(layers, list(range(8)) * 2 + [0, 3, 6])
        self.assertEqual(pool.index_k_buffer.shape[0], 3)
        self.assertEqual(pool._mla_buffer_info().buffers.index_layer_indices, (0, 3, 6))
        self.assertEqual(sum(row_bytes), 128 * (8 * (512 + 64) * 2 + 3 * 128 * 2))
        for layer, slot in ((0, 0), (3, 1), (6, 2)):
            self.assertEqual(
                pool.get_index_k_buffer(layer).data_ptr(),
                pool.index_k_buffer[slot].data_ptr(),
            )

    def test_allocation_and_budget_share_the_opt_in(self):
        for dtype in (torch.bfloat16, torch.int8):
            for compact, indexer_layers in ((False, 8), (True, 3)):
                with self.subTest(dtype=dtype, compact=compact):
                    pool = self.build_pool(compact, dtype)
                    with envs.SGLANG_NPU_COMPACT_DSA_INDEXER.override(compact):
                        indexer_bytes = pool_configurator.DefaultPoolConfigurator._compute_dsa_indexer_cell_size(
                            None, kvc=self.configurator(dtype), num_layers=8
                        )
                    self.assertEqual(pool.num_indexer_layers, indexer_layers)
                    self.assertEqual(
                        indexer_bytes, 128 * dtype.itemsize * indexer_layers
                    )
                    self.assertEqual(
                        pool.index_k_buffer.nbytes,
                        (pool.size + pool.page_size) * indexer_bytes,
                    )

    def test_static_storage_publishes_compact_target_and_dense_draft_ids(self):
        target = self.build_pool()
        draft = self.build_pool(False)
        allocator = SimpleNamespace(
            storage=None, state=None, get_kvcache=lambda: target
        )
        storage = StaticKVStorage(
            None, allocator, draft_token_to_kv_pool=draft, num_hidden_layers=8
        )
        self.assertEqual(
            list(storage.regions(storage.owner).layer_ids), target.get_kv_layer_ids()
        )
        ids = build_kv_layer_ids(
            token_to_kv_pool=target,
            draft_token_to_kv_pool=draft,
            num_draft_entries=len(draft.get_kv_layer_ids()),
            num_hidden_layers=8,
            storage=storage,
        )
        self.assertEqual(ids, list(range(8)) * 2 + [0, 3, 6] + list(range(8, 16)) * 3)
        self.assertEqual(
            build_transfer_entry_pairs(ids, ids, len(ids), len(ids)),
            list(enumerate(range(len(ids)))),
        )
        dense = draft.get_kv_layer_ids()
        with self.assertRaisesRegex(RuntimeError, "missing a transfer entry"):
            build_transfer_entry_pairs(
                dense,
                target.get_kv_layer_ids(),
                len(dense),
                len(target.get_kv_layer_ids()),
            )

    def test_packed_registration_has_no_empty_value_entries(self):
        pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
        pool.start_layer = 2
        pool.layer_num = 3
        pool.indexer_layer_ids = (2, 4)
        pool.index_k_scale_buffer = object()
        pool.dsa_kv_cache_store_fp8 = True
        self.assertEqual(pool.get_kv_layer_ids(), [2, 3, 4, 2, 4, 2, 4])

    def test_l2_backup_and_restore_preserve_compact_rows(self):
        class Direction(Enum):
            H2D = 1
            D2H = 2

        def exchange(**kwargs):
            for name in ("k", "v", "index_k", "index_k_scale"):
                device, host = kwargs["device_" + name], kwargs["host_" + name]
                if device is None or host is None or not device.numel():
                    continue
                self.assertEqual(device.shape[0], host.shape[1])
                for dev, src in zip(
                    kwargs["device_indices"][::128], kwargs["host_indices"][::128]
                ):
                    dev, src = int(dev) // 128, int(src) // 128
                    if kwargs["direction"] == Direction.D2H:
                        host[src].copy_(device[:, dev])
                    else:
                        device[:, dev].copy_(host[src])

        for dtype in (torch.bfloat16, torch.int8):
            with self.subTest(dtype=dtype):
                pool = self.build_pool(dtype=dtype)
                for tensor in (pool.k_buffer, pool.v_buffer, pool.index_k_buffer):
                    tensor.copy_(
                        (torch.arange(tensor.numel()).reshape(tensor.shape) % 63).to(
                            dtype
                        )
                    )
                expected = [
                    tensor.clone()
                    for tensor in (pool.k_buffer, pool.v_buffer, pool.index_k_buffer)
                ]
                with (
                    patch.object(mla, "_is_npu", True),
                    patch.object(
                        mla, "memfabric_host_memory_enabled", return_value=False
                    ),
                    patch.object(mla, "ascendc_io_enabled", return_value=False),
                    patch.object(mla, "TransferDirection", Direction, create=True),
                    patch.object(
                        mla,
                        "transfer_kv_dim_exchange",
                        side_effect=exchange,
                        create=True,
                    ) as copy,
                    patch.object(mla, "make_kernel_ptr_table", return_value=None),
                    patch.dict(
                        mla.ALLOC_MEMORY_FUNCS,
                        {
                            "cpu": lambda dims, dtype, **kwargs: torch.empty(
                                dims, dtype=dtype
                            )
                        },
                    ),
                    patch(
                        "sglang.srt.mem_cache.pool_host.base.host_memory_budget_bytes",
                        return_value=1 << 40,
                    ),
                ):
                    host = mla.MLATokenToKVPoolHost(
                        pool._mla_buffer_info(),
                        host_to_device_ratio=2,
                        host_size=0,
                        page_size=128,
                        layout="page_first_kv_split",
                        pin_memory=False,
                        device="cpu",
                        device_capacity_tokens=pool.size,
                        start_layer=0,
                        end_layer=8,
                        target_layer_num=8,
                    )
                    self.assertEqual(host.index_k_buffer.shape[1], 3)
                    device_ids = torch.arange(256)
                    host_ids = torch.cat((torch.arange(256, 384), torch.arange(128)))
                    host.backup_from_device_all_layer(
                        pool, host_ids, device_ids, "kernel_ascend"
                    )
                    for tensor in (pool.k_buffer, pool.v_buffer, pool.index_k_buffer):
                        tensor[:, :2].fill_(-1)
                    for layer in range(8):
                        host.load_to_device_per_layer(
                            pool, host_ids, device_ids, layer, "kernel_ascend"
                        )
                    self.assertEqual(copy.call_count, 2)
                    for actual, original in zip(
                        (pool.k_buffer, pool.v_buffer, pool.index_k_buffer), expected
                    ):
                        torch.testing.assert_close(actual, original, rtol=0, atol=0)
                    native = pool._mla_buffer_info().buffers
                    self.assertEqual(
                        [
                            host._indexer_slot_range_for_layer(native, layer)
                            for layer in range(8)
                        ],
                        [
                            (0, 1),
                            (0, 0),
                            (0, 0),
                            (1, 1),
                            (0, 0),
                            (0, 0),
                            (2, 1),
                            (0, 0),
                        ],
                    )


if __name__ == "__main__":
    unittest.main()
