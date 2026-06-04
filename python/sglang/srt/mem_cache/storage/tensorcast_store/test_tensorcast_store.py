"""Unit tests for the Tensorcast HiCache backend."""

# ruff: noqa: E402

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.test.mem_cache.host_pool_test_utils import install_memory_pool_host_stub

install_memory_pool_host_stub()

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.host_shared_slot_state import (
    HostSharedPageSlotStaleTokenError,
    HostSharedPageSlotToken,
)
from sglang.srt.mem_cache.storage.tensorcast_store.client import (
    TensorcastBatchExistsResult,
    TensorcastBatchTransferResult,
    _compact_cgid_segment,
    _engine_key_payload,
)
from sglang.srt.mem_cache.storage.tensorcast_store.host_allocator import (
    TensorcastHostRegionBinding,
)
from sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store import (
    TensorcastStore,
)


class FakeTensorcastPageClient:
    def __init__(self) -> None:
        self.data: dict[str, torch.Tensor] = {}
        self.last_put_slot_tokens: list[HostSharedPageSlotToken] | None = None
        self.last_get_slot_tokens: list[HostSharedPageSlotToken] | None = None
        self.last_put_source_region_binding: TensorcastHostRegionBinding | None = None
        self.last_get_target_region_binding: TensorcastHostRegionBinding | None = None
        self.activate_stable_local_backing_calls: list[tuple[str, int]] = []

    def activate_stable_local_backing(self, region_id: str, *, slot_bytes: int) -> None:
        self.activate_stable_local_backing_calls.append((region_id, slot_bytes))

    def artifact_id_for(self, logical_key: str) -> str:
        return f"artifact::{logical_key}"

    def batch_exists(self, logical_keys: list[str]) -> TensorcastBatchExistsResult:
        return TensorcastBatchExistsResult(
            existence_mask=tuple(key in self.data for key in logical_keys),
            rpc_elapsed_s=0.0,
        )

    def batch_put(
        self,
        logical_keys: list[str],
        pages: list[torch.Tensor],
        slot_tokens: list[HostSharedPageSlotToken] | None = None,
        source_region_binding: TensorcastHostRegionBinding | None = None,
    ) -> TensorcastBatchTransferResult:
        self.last_put_slot_tokens = (
            list(slot_tokens) if slot_tokens is not None else None
        )
        self.last_put_source_region_binding = source_region_binding
        success_mask: list[bool] = []
        duplicate_count = 0
        for logical_key, page in zip(logical_keys, pages, strict=True):
            if logical_key in self.data:
                duplicate_count += 1
                success_mask.append(True)
                continue
            self.data[logical_key] = page.clone()
            success_mask.append(True)
        return TensorcastBatchTransferResult(
            success_mask=tuple(success_mask),
            adopted_duplicate_count=duplicate_count,
            pack_elapsed_s=0.0,
            stage_copy_elapsed_s=0.0,
            rpc_elapsed_s=0.0,
            host_fill_elapsed_s=0.0,
        )

    def batch_get_into(
        self,
        logical_keys: list[str],
        targets: list[torch.Tensor],
        slot_tokens: list[HostSharedPageSlotToken] | None = None,
        target_region_binding: TensorcastHostRegionBinding | None = None,
    ) -> TensorcastBatchTransferResult:
        self.last_get_slot_tokens = (
            list(slot_tokens) if slot_tokens is not None else None
        )
        self.last_get_target_region_binding = target_region_binding
        success_mask: list[bool] = []
        stop_copying = False
        for logical_key, target in zip(logical_keys, targets, strict=True):
            if stop_copying or logical_key not in self.data:
                stop_copying = True
                success_mask.append(False)
                continue
            target.copy_(self.data[logical_key])
            success_mask.append(True)
        return TensorcastBatchTransferResult(
            success_mask=tuple(success_mask),
            adopted_duplicate_count=0,
            pack_elapsed_s=0.0,
            stage_copy_elapsed_s=0.0,
            rpc_elapsed_s=0.0,
            host_fill_elapsed_s=0.0,
        )


class FakeHostKVCache:
    def __init__(
        self,
        values: list[float],
        *,
        page_size: int = 2,
        layout: str = "page_first",
        host_region_binding: object | None = None,
        size_per_token: int = 4,
    ) -> None:
        self.page_size = page_size
        self.layout = layout
        self.dtype = torch.float32
        self.size_per_token = size_per_token
        self.kv_buffer = torch.tensor(values, dtype=self.dtype).reshape(-1, page_size)
        self._host_region_binding = host_region_binding
        self._slot_generations: dict[int, int] = {
            page_start: (page_start // self.page_size) + 100
            for page_start in range(0, self.kv_buffer.numel(), self.page_size)
        }
        self.reserve_calls: list[tuple[list[int], list[str] | None]] = []
        self.mark_get_inflight_calls: list[list[HostSharedPageSlotToken]] = []
        self.commit_get_success_calls: list[
            tuple[list[HostSharedPageSlotToken], list[str] | None]
        ] = []
        self.fail_get_calls: list[list[HostSharedPageSlotToken]] = []

    @property
    def host_region_binding(self) -> object | None:
        return self._host_region_binding

    def get_data_page(self, index: int, flat: bool = True) -> torch.Tensor:
        page = self.kv_buffer[index // self.page_size]
        return page.view(-1) if flat else page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros((self.page_size,), dtype=self.dtype)

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        self.kv_buffer[index // self.page_size] = data_page.reshape(self.page_size)

    def slot_tokens_for_page_starts(
        self, page_starts: list[int]
    ) -> tuple[HostSharedPageSlotToken, ...]:
        return tuple(
            HostSharedPageSlotToken(
                slot_index=page_start // self.page_size,
                slot_generation=self._slot_generations[page_start],
            )
            for page_start in page_starts
        )

    def reserve_page_slots(
        self,
        page_starts: list[int],
        logical_keys: list[str] | None = None,
    ) -> tuple[HostSharedPageSlotToken, ...]:
        self.reserve_calls.append(
            (
                list(page_starts),
                list(logical_keys) if logical_keys is not None else None,
            )
        )
        return self.slot_tokens_for_page_starts(page_starts)

    def mark_page_get_inflight(
        self, slot_tokens: list[HostSharedPageSlotToken]
    ) -> None:
        self.mark_get_inflight_calls.append(list(slot_tokens))

    def commit_page_get_success(
        self,
        slot_tokens: list[HostSharedPageSlotToken],
        logical_keys: list[str] | None = None,
    ) -> None:
        self.commit_get_success_calls.append(
            (
                list(slot_tokens),
                list(logical_keys) if logical_keys is not None else None,
            )
        )

    def fail_page_get(self, slot_tokens: list[HostSharedPageSlotToken]) -> None:
        self.fail_get_calls.append(list(slot_tokens))


class StaleCommitHostKVCache(FakeHostKVCache):
    def __init__(self, values: list[float], **kwargs: object) -> None:
        super().__init__(values, **kwargs)
        self.stale_commit_attempts: list[
            tuple[list[HostSharedPageSlotToken], list[str] | None]
        ] = []

    def commit_page_get_success(
        self,
        slot_tokens: list[HostSharedPageSlotToken],
        logical_keys: list[str] | None = None,
    ) -> None:
        self.stale_commit_attempts.append(
            (
                list(slot_tokens),
                list(logical_keys) if logical_keys is not None else None,
            )
        )
        raise HostSharedPageSlotStaleTokenError(
            "stale slot token rejected before page becomes visible"
        )


def build_storage_config(
    *,
    is_mla_model: bool = False,
    tp_rank: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    extra_config: dict[str, object] | str | None = None,
) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=4,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=is_mla_model,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="Qwen3-32B",
        extra_config=extra_config
        or {
            "daemon_address": "127.0.0.1:50052",
            "namespace": "unit-test",
        },
    )


class TensorcastStoreTest(unittest.TestCase):
    def test_tensorcast_store_batch_set_and_get_v1(self) -> None:
        client = FakeTensorcastPageClient()
        host_cache = FakeHostKVCache([1.0, 2.0, 3.0, 4.0])
        store = TensorcastStore(
            build_storage_config(),
            host_cache,
            page_client=client,
        )

        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        self.assertEqual(
            store.batch_set_v1(["hash_a", "hash_b"], host_indices), [True, True]
        )
        self.assertEqual(store.batch_exists(["hash_a", "hash_b"]), 2)

        host_cache.kv_buffer.zero_()
        self.assertEqual(
            store.batch_get_v1(["hash_a", "hash_b"], host_indices), [True, True]
        )
        self.assertTrue(
            torch.equal(
                host_cache.kv_buffer.flatten(),
                torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
            )
        )
        self.assertIsNone(client.last_put_slot_tokens)
        self.assertIsNone(client.last_get_slot_tokens)

    def test_tensorcast_store_can_forward_host_slot_tokens_when_enabled(self) -> None:
        client = FakeTensorcastPageClient()
        host_cache = FakeHostKVCache([21.0, 22.0, 23.0, 24.0])
        store = TensorcastStore(
            build_storage_config(),
            host_cache,
            page_client=client,
        )

        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        extra_info = HiCacheStorageExtraInfo(
            extra_info={"tensorcast_use_host_slot_tokens": True}
        )
        self.assertEqual(
            store.batch_set_v1(["hash_a", "hash_b"], host_indices, extra_info),
            [True, True],
        )
        self.assertEqual(
            client.last_put_slot_tokens,
            [
                HostSharedPageSlotToken(slot_index=0, slot_generation=100),
                HostSharedPageSlotToken(slot_index=1, slot_generation=101),
            ],
        )

        host_cache.kv_buffer.zero_()
        self.assertEqual(
            store.batch_get_v1(["hash_a", "hash_b"], host_indices, extra_info),
            [True, True],
        )
        self.assertEqual(
            client.last_get_slot_tokens,
            [
                HostSharedPageSlotToken(slot_index=0, slot_generation=100),
                HostSharedPageSlotToken(slot_index=1, slot_generation=101),
            ],
        )

    def test_tensorcast_store_duplicate_batch_set_is_idempotent(self) -> None:
        client = FakeTensorcastPageClient()
        host_cache = FakeHostKVCache([7.0, 8.0])
        store = TensorcastStore(
            build_storage_config(),
            host_cache,
            page_client=client,
        )

        host_indices = torch.tensor([0, 1], dtype=torch.int64)
        self.assertEqual(store.batch_set_v1(["hash_dup"], host_indices), [True])

        host_cache.kv_buffer = torch.tensor([9.0, 10.0], dtype=torch.float32).reshape(
            1, 2
        )
        self.assertEqual(store.batch_set_v1(["hash_dup"], host_indices), [True])
        self.assertEqual(store._publication_stats.duplicate_pages, 1)
        self.assertTrue(
            torch.equal(
                client.data["hash_dup"], torch.tensor([7.0, 8.0], dtype=torch.float32)
            )
        )

    def test_tensorcast_store_batch_exists_stops_at_first_missing_key(self) -> None:
        client = FakeTensorcastPageClient()
        host_cache = FakeHostKVCache([11.0, 12.0, 13.0, 14.0])
        store = TensorcastStore(
            build_storage_config(),
            host_cache,
            page_client=client,
        )

        host_indices = torch.tensor([0, 1], dtype=torch.int64)
        self.assertEqual(store.batch_set_v1(["hash_present"], host_indices), [True])
        self.assertEqual(
            store.batch_exists(["hash_present", "hash_missing", "hash_after"]),
            1,
        )

    def test_tensorcast_store_accepts_json_string_extra_config(self) -> None:
        client = FakeTensorcastPageClient()
        host_cache = FakeHostKVCache([15.0, 16.0])
        store = TensorcastStore(
            build_storage_config(
                extra_config='{"daemon_address":"127.0.0.1:50052","namespace":"json-test"}'
            ),
            host_cache,
            page_client=client,
        )

        self.assertEqual(store._tensorcast_config.namespace, "json-test")
        self.assertEqual(store._tensorcast_config.model_id, "Qwen3-32B")
        self.assertEqual(store._tensorcast_config.model_version, "default")

    def test_tensorcast_store_layout_and_rank_suffix_for_mha(self) -> None:
        store = TensorcastStore(
            build_storage_config(tp_rank=1, pp_rank=0, pp_size=1),
            FakeHostKVCache([1.0, 2.0]),
            page_client=FakeTensorcastPageClient(),
        )

        self.assertEqual(store._rank_suffix, "tp1of4")
        self.assertEqual(
            store._layout_id,
            "sglang_kv_page_v1_page_first_torch.float32_ps2_mha",
        )

    def test_tensorcast_store_mla_uses_pp_rank_suffix(self) -> None:
        store = TensorcastStore(
            build_storage_config(is_mla_model=True, tp_rank=3, pp_rank=2, pp_size=4),
            FakeHostKVCache([5.0, 6.0]),
            page_client=FakeTensorcastPageClient(),
        )

        self.assertEqual(store._rank_suffix, "pp2of4")
        self.assertEqual(
            store._layout_id,
            "sglang_kv_page_v1_page_first_torch.float32_ps2_mla",
        )

    def test_tensorcast_store_rejects_allocator_backed_wrong_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "page_blob_direct"):
            TensorcastStore(
                build_storage_config(
                    extra_config={
                        "daemon_address": "127.0.0.1:50052",
                        "namespace": "unit-test",
                        "host_allocator_enabled": True,
                    }
                ),
                FakeHostKVCache([1.0, 2.0], layout="page_first"),
                page_client=FakeTensorcastPageClient(),
            )

    def test_tensorcast_store_rejects_allocator_backed_missing_binding(self) -> None:
        with self.assertRaisesRegex(ValueError, "host_region_binding"):
            TensorcastStore(
                build_storage_config(
                    extra_config={
                        "daemon_address": "127.0.0.1:50052",
                        "namespace": "unit-test",
                        "host_allocator_enabled": True,
                    }
                ),
                FakeHostKVCache([1.0, 2.0], layout="page_blob_direct"),
                page_client=FakeTensorcastPageClient(),
            )

    def test_tensorcast_store_accepts_allocator_backed_page_blob_layout(self) -> None:
        binding = TensorcastHostRegionBinding(
            region_id="region-1",
            capacity_bytes=4096,
            base_ptr=1234,
            handle=SimpleNamespace(),
            region_name="test-region",
        )
        store = TensorcastStore(
            build_storage_config(
                extra_config={
                    "daemon_address": "127.0.0.1:50052",
                    "namespace": "unit-test",
                    "host_allocator_enabled": True,
                }
            ),
            FakeHostKVCache(
                [1.0, 2.0],
                layout="page_blob_direct",
                host_region_binding=binding,
            ),
            page_client=FakeTensorcastPageClient(),
        )

        self.assertEqual(store.layout, "page_blob_direct")
        self.assertIs(store.mem_pool_host.host_region_binding, binding)

    def test_allocator_backed_batch_set_uses_resident_slot_region(self) -> None:
        binding = TensorcastHostRegionBinding(
            region_id="region-2",
            capacity_bytes=8192,
            base_ptr=4321,
            handle=SimpleNamespace(),
            region_name="test-region",
        )
        client = FakeTensorcastPageClient()
        store = TensorcastStore(
            build_storage_config(
                extra_config={
                    "daemon_address": "127.0.0.1:50052",
                    "namespace": "unit-test",
                    "host_allocator_enabled": True,
                }
            ),
            FakeHostKVCache(
                [31.0, 32.0, 33.0, 34.0],
                layout="page_blob_direct",
                host_region_binding=binding,
            ),
            page_client=client,
        )

        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        self.assertEqual(
            store.batch_set_v1(["hash_a", "hash_b"], host_indices), [True, True]
        )
        self.assertEqual(
            client.last_put_slot_tokens,
            [
                HostSharedPageSlotToken(slot_index=0, slot_generation=100),
                HostSharedPageSlotToken(slot_index=1, slot_generation=101),
            ],
        )
        self.assertIs(client.last_put_source_region_binding, binding)

    def test_allocator_backed_batch_get_uses_resident_slot_region(self) -> None:
        binding = TensorcastHostRegionBinding(
            region_id="region-3",
            capacity_bytes=8192,
            base_ptr=5678,
            handle=SimpleNamespace(),
            region_name="test-region",
        )
        client = FakeTensorcastPageClient()
        host_cache = FakeHostKVCache(
            [41.0, 42.0, 43.0, 44.0],
            layout="page_blob_direct",
            host_region_binding=binding,
        )
        store = TensorcastStore(
            build_storage_config(
                extra_config={
                    "daemon_address": "127.0.0.1:50052",
                    "namespace": "unit-test",
                    "host_allocator_enabled": True,
                }
            ),
            host_cache,
            page_client=client,
        )

        client.data["hash_a"] = torch.tensor([41.0, 42.0], dtype=torch.float32)
        client.data["hash_b"] = torch.tensor([43.0, 44.0], dtype=torch.float32)
        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        host_cache.kv_buffer.zero_()

        self.assertEqual(
            store.batch_get_v1(["hash_a", "hash_b"], host_indices), [True, True]
        )
        self.assertEqual(
            client.last_get_slot_tokens,
            [
                HostSharedPageSlotToken(slot_index=0, slot_generation=100),
                HostSharedPageSlotToken(slot_index=1, slot_generation=101),
            ],
        )
        self.assertIs(client.last_get_target_region_binding, binding)
        self.assertEqual(
            host_cache.reserve_calls,
            [([0, 2], ["hash_a", "hash_b"])],
        )
        self.assertEqual(
            host_cache.mark_get_inflight_calls,
            [
                [
                    HostSharedPageSlotToken(slot_index=0, slot_generation=100),
                    HostSharedPageSlotToken(slot_index=1, slot_generation=101),
                ]
            ],
        )
        self.assertEqual(
            host_cache.commit_get_success_calls,
            [
                (
                    [
                        HostSharedPageSlotToken(slot_index=0, slot_generation=100),
                        HostSharedPageSlotToken(slot_index=1, slot_generation=101),
                    ],
                    ["hash_a", "hash_b"],
                )
            ],
        )
        self.assertEqual(host_cache.fail_get_calls, [])
        self.assertTrue(
            torch.equal(
                host_cache.kv_buffer.flatten(),
                torch.tensor([41.0, 42.0, 43.0, 44.0], dtype=torch.float32),
            )
        )

    def test_allocator_backed_batch_get_marks_failed_suffix(self) -> None:
        binding = TensorcastHostRegionBinding(
            region_id="region-4",
            capacity_bytes=8192,
            base_ptr=9012,
            handle=SimpleNamespace(),
            region_name="test-region",
        )
        client = FakeTensorcastPageClient()
        host_cache = FakeHostKVCache(
            [51.0, 52.0, 0.0, 0.0],
            layout="page_blob_direct",
            host_region_binding=binding,
        )
        store = TensorcastStore(
            build_storage_config(
                extra_config={
                    "daemon_address": "127.0.0.1:50052",
                    "namespace": "unit-test",
                    "host_allocator_enabled": True,
                }
            ),
            host_cache,
            page_client=client,
        )

        client.data["hash_a"] = torch.tensor([51.0, 52.0], dtype=torch.float32)
        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        host_cache.kv_buffer.zero_()

        self.assertEqual(
            store.batch_get_v1(["hash_a", "hash_missing"], host_indices),
            [True, False],
        )
        self.assertEqual(
            host_cache.commit_get_success_calls,
            [
                (
                    [HostSharedPageSlotToken(slot_index=0, slot_generation=100)],
                    ["hash_a"],
                )
            ],
        )
        self.assertEqual(
            host_cache.fail_get_calls,
            [[HostSharedPageSlotToken(slot_index=1, slot_generation=101)]],
        )

    def test_allocator_backed_batch_get_propagates_stale_commit_before_visibility(
        self,
    ) -> None:
        binding = TensorcastHostRegionBinding(
            region_id="region-5",
            capacity_bytes=8192,
            base_ptr=2468,
            handle=SimpleNamespace(),
            region_name="test-region",
        )
        client = FakeTensorcastPageClient()
        host_cache = StaleCommitHostKVCache(
            [61.0, 62.0],
            layout="page_blob_direct",
            host_region_binding=binding,
        )
        store = TensorcastStore(
            build_storage_config(
                extra_config={
                    "daemon_address": "127.0.0.1:50052",
                    "namespace": "unit-test",
                    "host_allocator_enabled": True,
                }
            ),
            host_cache,
            page_client=client,
        )

        client.data["hash_a"] = torch.tensor([61.0, 62.0], dtype=torch.float32)
        host_indices = torch.tensor([0, 1], dtype=torch.int64)
        host_cache.kv_buffer.zero_()

        with self.assertRaises(HostSharedPageSlotStaleTokenError):
            store.batch_get_v1(["hash_a"], host_indices)

        self.assertEqual(host_cache.commit_get_success_calls, [])
        self.assertEqual(
            host_cache.stale_commit_attempts,
            [
                (
                    [HostSharedPageSlotToken(slot_index=0, slot_generation=100)],
                    ["hash_a"],
                )
            ],
        )
        self.assertEqual(host_cache.fail_get_calls, [])

    def test_cgid_helpers_compact_long_segments_and_hash_keys(self) -> None:
        compact_namespace = _compact_cgid_segment(
            "share_local:20260325-142807_tensorcast_tp2_pairs1",
            prefix="ns",
        )
        compact_layout = _compact_cgid_segment(
            "sglang_kv_page_v1_page_first_torch.bfloat16_ps64_mha",
            prefix="ly",
        )
        engine_key_payload = _engine_key_payload(
            "tp0of2",
            "fc6267db9ba02bbb251960231bc87251c581110e4e2b5f3f8a0d60b884d57ccf",
        )

        self.assertLessEqual(len(compact_namespace), 19)
        self.assertLessEqual(len(compact_layout), 19)
        self.assertEqual(engine_key_payload[:7], b"tp0of2:")
        self.assertEqual(len(engine_key_payload), 39)


if __name__ == "__main__":
    unittest.main()
