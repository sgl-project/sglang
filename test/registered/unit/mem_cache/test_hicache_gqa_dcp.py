"""Logical KV pages must round-trip through physical MHA host rows under DCP."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups import hicache_hook
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_pools(dcp_size, rank, *, device="cpu", layout="page_first_direct"):
    pool = MHATokenToKVPool(
        size=64,
        page_size=4,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=8,
        layer_num=2,
        device=device,
        enable_memory_saver=False,
    )
    host = MHATokenToKVPoolHost(
        pool.get_device_pool_infos()[0].buffer_info,
        host_to_device_ratio=2,
        host_size=0,
        page_size=4 * dcp_size,
        layout=layout,
        pin_memory=device != "cpu",
        device="cpu",
        device_capacity_tokens=pool.size,
        start_layer=pool.start_layer,
        end_layer=pool.end_layer,
        dcp_size=dcp_size,
        dcp_rank=rank,
    )
    return pool, host


class TestMhaHostDcp(CustomTestCase):
    def test_sizing_and_allocator_use_distinct_spaces(self):
        for size in (1, 2, 4):
            for rank in range(size):
                with self.subTest(size=size, rank=rank):
                    _, host = make_pools(size, rank)
                    self.addCleanup(host.destroy)
                    self.assertEqual(host.page_size, 4)
                    self.assertEqual(host.logical_page_size, 4 * size)
                    self.assertEqual(host.logical_size, host.size * size)
                    self.assertEqual(
                        host.kv_buffer.numel() * host.dtype.itemsize,
                        host.size * host.size_per_token,
                    )
                    ids = host.alloc(host.logical_size)
                    self.assertEqual(host.available_size(), 0)
                    self.assertIsNone(host.alloc(host.logical_page_size))
                    host.free(ids)
                    self.assertEqual(host.available_size(), host.logical_size)

    def test_transfer_pairs_fragmented_and_highest_pages(self):
        for size in (1, 2, 4):
            for rank in range(size):
                with self.subTest(size=size, rank=rank):
                    pool, host = make_pools(size, rank)
                    self.addCleanup(host.destroy)
                    page = 4 * size
                    host_ids = torch.cat(
                        (
                            torch.arange(host.logical_size - page, host.logical_size),
                            torch.arange(page, 2 * page),
                        )
                    )
                    device_ids = torch.cat(
                        (torch.arange(3 * page, 4 * page), torch.arange(page))
                    )
                    expected_host = host_ids[host_ids % size == rank] // size
                    expected_device = device_ids[device_ids % size == rank] // size
                    with patch(
                        "sglang.srt.mem_cache.pool_host.mha.transfer_kv_all_layer_direct_lf_pf",
                        create=True,
                    ) as backup:
                        host.backup_from_device_all_layer(
                            pool, host_ids, device_ids, "direct"
                        )
                    torch.testing.assert_close(
                        backup.call_args.kwargs["src_indices"], expected_device
                    )
                    torch.testing.assert_close(
                        backup.call_args.kwargs["dst_indices"], expected_host
                    )
                    self.assertEqual(backup.call_args.kwargs["page_size"], 4)
                    with patch(
                        "sglang.srt.mem_cache.pool_host.mha.transfer_kv_per_layer_direct_pf_lf",
                        create=True,
                    ) as load:
                        host.load_to_device_per_layer(
                            pool, host_ids, device_ids, 0, "direct"
                        )
                    torch.testing.assert_close(
                        load.call_args.kwargs["src_indices"], expected_host
                    )
                    torch.testing.assert_close(
                        load.call_args.kwargs["dst_indices"], expected_device
                    )

    def test_empty_transfer_does_not_launch_a_copy(self):
        pool, host = make_pools(4, 3)
        self.addCleanup(host.destroy)
        empty = torch.empty(0, dtype=torch.int64)
        with (
            patch(
                "sglang.srt.mem_cache.pool_host.mha.transfer_kv_all_layer_direct_lf_pf",
                create=True,
            ) as backup,
            patch(
                "sglang.srt.mem_cache.pool_host.mha.transfer_kv_per_layer_direct_pf_lf",
                create=True,
            ) as load,
        ):
            host.backup_from_device_all_layer(pool, empty, empty, "direct")
            host.load_to_device_per_layer(pool, empty, empty, 0, "direct")
        backup.assert_not_called()
        load.assert_not_called()

    def test_byte_accounting_does_not_shard_mamba_checkpoints(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.is_kv_sharded = False
        anchor = SimpleNamespace(
            name=PoolName.KV, host_pool=SimpleNamespace(size_per_token=64, dcp_size=4)
        )
        state = SimpleNamespace(
            host_pool=SimpleNamespace(size_per_token=128, dcp_size=1)
        )
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=anchor, entry_map={PoolName.MAMBA: state}
        )
        op = SimpleNamespace(
            device_indices=torch.arange(256),
            pool_transfers=[
                SimpleNamespace(
                    name=PoolName.MAMBA,
                    host_indices=torch.arange(3),
                    indices_from_pool=None,
                )
            ],
        )
        self.assertEqual(controller._transfer_num_bytes(op), 64 * 64 + 3 * 128)


class TestMhaDcpAdmission(CustomTestCase):
    def config(self, **overrides):
        fields = dict(
            attention_backend="triton",
            prefill_attention_backend=None,
            decode_attention_backend=None,
            dtype="bfloat16",
            kv_cache_dtype="bf16",
            hicache_mem_layout="page_first_direct",
            hicache_io_backend="direct",
            hicache_write_policy="write_through",
            hicache_host_memory_mode="cache",
            hicache_storage_backend=None,
            speculative_algorithm=None,
            disaggregation_mode="null",
            tp_size=4,
            dcp_size=2,
            pp_size=1,
            dp_size=1,
            attn_dp_size=1,
            ep_size=1,
            enable_unified_memory=False,
            enable_kv_cache_sharding=False,
        )
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def validate(
        self, cfg, *, cuda=True, swa=False, qwen=True, sparse=False, heads=(1, 1)
    ):
        with (
            patch.object(hicache_hook, "resolving_view", return_value=cfg),
            patch.object(
                hicache_hook, "get_platform", return_value=SimpleNamespace(is_cuda=cuda)
            ),
            patch.object(
                hicache_hook,
                "model_config_of",
                return_value=SimpleNamespace(
                    is_hybrid_swa=swa,
                    hf_config=object(),
                    hf_text_config=SimpleNamespace(indexer_n_heads=4 if sparse else 0),
                    get_num_kv_heads=lambda tp, dcp=1: (
                        heads[1] if dcp > 1 else heads[0]
                    ),
                ),
            ),
            patch.object(hicache_hook, "is_qwen3_5", return_value=qwen),
        ):
            hicache_hook._validate_mha_dcp_hicache(cfg)

    def test_admits_only_validated_io_pairs(self):
        self.validate(self.config())
        self.validate(
            self.config(hicache_mem_layout="page_first", hicache_io_backend="kernel")
        )

    def test_admits_auto_fp16(self):
        for dtype in ("float16", "half"):
            with self.subTest(dtype=dtype):
                self.validate(self.config(dtype=dtype, kv_cache_dtype="auto"))

    def test_builder_checks_resolved_dtype(self):
        memory = SimpleNamespace(
            hicache_ratio=2, hicache_size=0, hicache_mem_layout="page_first"
        )
        parallel = SimpleNamespace(dcp_enabled=True, attn_dcp_size=2, attn_dcp_rank=0)
        for dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float32,
        ):
            descriptor = SimpleNamespace(dtype=dtype)
            device_pool = SimpleNamespace(
                layer_num=2,
                get_device_pool_infos=lambda: (
                    SimpleNamespace(buffer_info=descriptor),
                ),
            )
            with (
                self.subTest(dtype=dtype),
                patch.object(hybrid_pool_assembler, "MHATokenToKVPoolHost") as pool,
                patch.object(
                    hybrid_pool_assembler,
                    "_mha_host_inputs",
                    return_value=(pool, SimpleNamespace(dtype=dtype), {}),
                ),
                patch.object(hybrid_pool_assembler, "get_memory", return_value=memory),
                patch.object(
                    hybrid_pool_assembler, "get_parallel", return_value=parallel
                ),
                patch.object(hybrid_pool_assembler, "is_npu", return_value=False),
                patch.object(
                    hybrid_pool_assembler, "_get_allocator_type", return_value="default"
                ),
            ):
                if dtype in (torch.bfloat16, torch.float16):
                    hybrid_pool_assembler.build_kv_host_pool(
                        kv_pool=device_pool, page_size=128, use_mla=False
                    )
                    self.assertEqual(pool.call_args.kwargs["dcp_size"], 2)
                else:
                    with self.assertRaisesRegex(ValueError, "FP16/BF16"):
                        hybrid_pool_assembler.build_kv_host_pool(
                            kv_pool=device_pool, page_size=128, use_mla=False
                        )
                    pool.assert_not_called()

    def test_rejects_unvalidated_compositions(self):
        cases = [
            dict(kv_cache_dtype="fp8_e4m3"),
            dict(dtype="float16", kv_cache_dtype="fp8_e4m3"),
            dict(dtype="float16", kv_cache_dtype="fp8_e5m2"),
            dict(dtype="float16", kv_cache_dtype="mxfp8"),
            dict(dtype="float16", kv_cache_dtype="nvfp4"),
            dict(dtype="float32", kv_cache_dtype="auto"),
            dict(kv_cache_dtype="auto"),
            dict(attention_backend="fa3"),
            dict(decode_attention_backend="flashinfer"),
            dict(hicache_mem_layout="layer_first"),
            dict(hicache_io_backend="kernel"),
            dict(hicache_write_policy="write_back"),
            dict(hicache_storage_backend="file"),
            dict(hicache_host_memory_mode="buffer_only"),
            dict(speculative_algorithm="NEXTN"),
            dict(speculative_algorithm="DSPARK"),
            dict(disaggregation_mode="decode"),
            dict(pp_size=2),
            dict(dp_size=2),
            dict(attn_dp_size=2),
            dict(ep_size=2),
            dict(enable_unified_memory=True),
            dict(enable_kv_cache_sharding=True),
        ]
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(NotImplementedError):
                self.validate(self.config(**fields))
        with self.assertRaises(NotImplementedError):
            self.validate(self.config(), cuda=False)
        with self.assertRaises(NotImplementedError):
            self.validate(self.config(), swa=True)
        with self.assertRaises(NotImplementedError):
            self.validate(self.config(), sparse=True)

    def test_existing_kv_projection_layout_must_cover_the_dcp_group(self):
        self.validate(self.config(), qwen=False, heads=(1, 1))
        with self.assertRaises(NotImplementedError):
            self.validate(self.config(), qwen=False, heads=(1, 2))
        self.validate(self.config(), qwen=True, heads=(1, 2))


if __name__ == "__main__":
    unittest.main()
