import argparse
import inspect

import pytest
import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import PoolTransferResult
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import (
    MLATokenToKVPool,
    should_enable_dsa_cp_shared_kvcache,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs


def _server_args_for_dsa_backend_defaults(
    *,
    attn_cp_size=8,
    dsa_prefill_backend=None,
    dsa_decode_backend=None,
    kv_cache_dtype="bfloat16",
):
    args = object.__new__(ServerArgs)
    args.attn_cp_size = attn_cp_size
    args.dsa_prefill_backend = dsa_prefill_backend
    args.dsa_decode_backend = dsa_decode_backend
    args.enable_dsa_cp_shared_kv_cache = False
    args.enable_hisparse = False
    args.kv_cache_dtype = kv_cache_dtype
    return args


def _fake_shared_mla_pool(*, cp_rank: int, cp_size: int, page_size: int):
    pool = object.__new__(MLATokenToKVPool)
    pool.enable_cp_shared_kvcache = True
    pool.cp_shared_cp_rank = cp_rank
    pool.cp_shared_cp_size = cp_size
    pool.cp_shared_pages_per_rank = 8
    pool.page_size = page_size
    return pool


class _FakeSharedDevicePool:
    enable_cp_shared_kvcache = True

    def __init__(self):
        self.kv_buffer = [torch.empty(1)]
        self.data_ptrs = torch.empty(1, dtype=torch.uint64)
        self.calls = []

    def get_cp_shared_hicache_transfer_indices(
        self, host_indices, device_indices, *, load_shared_l2=False
    ):
        self.calls.append("translate")
        return host_indices, device_indices + 1000

    def synchronize_cp_shared_hicache_transfer(self):
        self.calls.append("sync")


class _FakeSharedHiCacheDevicePool:
    enable_cp_shared_kvcache = True
    cp_shared_cp_size = 2
    device = "cpu"
    layer_num = 1

    def __init__(self, is_io_rank: bool):
        self._is_io_rank = is_io_rank

    def is_cp_shared_hicache_io_rank(self):
        return self._is_io_rank


class _FakeHostPool:
    def __init__(self, alloc_result, available_result=0):
        self.alloc_result = alloc_result
        self.available_result = available_result
        self.alloc_calls = []
        self.free_calls = []
        self.available_calls = 0

    def alloc(self, size):
        self.alloc_calls.append(size)
        return self.alloc_result

    def free(self, indices):
        self.free_calls.append(indices)

    def available_size(self):
        self.available_calls += 1
        return self.available_result


def _mark_fake_shared_l2_host(host, *, is_owner: bool):
    host.enable_cp_shared_dsa_l2 = True
    host.cp_shared_l2_is_owner = lambda: is_owner
    return host


def _fake_hicache_controller(
    *, is_io_rank: bool, alloc_result=None, available_result=0
):
    controller = object.__new__(HiCacheController)
    controller.mem_pool_device = _FakeSharedHiCacheDevicePool(is_io_rank)
    controller.mem_pool_host = _mark_fake_shared_l2_host(
        _FakeHostPool(alloc_result, available_result), is_owner=is_io_rank
    )
    controller.device = "cpu"
    controller.write_queue = []
    controller.started = 0
    controller.start_writing = lambda: setattr(
        controller, "started", controller.started + 1
    )
    return controller


class _CountingSharedL2Host:
    layout = "page_first"
    page_size = 64
    device = "cpu"
    size = 128
    can_use_write_back_jit = False
    size_per_token = 1
    allocator = None
    dtype = torch.float8_e4m3fn
    start_layer = 0
    end_layer = 1

    def __init__(self):
        self.load_calls = []
        self.backup_calls = []

    def get_ksize_per_token(self):
        return 1

    def load_to_device_per_layer(self, *args, **kwargs):
        self.load_calls.append((args, kwargs))

    def backup_from_device_all_layer(self, *args, **kwargs):
        self.backup_calls.append((args, kwargs))


def _shared_l2_group(host, *, cp_rank, device_pool=None):
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry

    entry = PoolEntry(
        name=PoolName.KV,
        host_pool=host,
        device_pool=device_pool or object(),
        layer_mapper=lambda layer_id: layer_id,
        is_primary_index_anchor=True,
    )
    return HostPoolGroup(
        [entry], enable_cp_shared_dsa_l2=True, owner_rank=0, cp_rank=cp_rank
    )


class TestDsaCpSharedBackendDefaults:
    def test_dsa_cp_shared_kvcache_cli_flag_is_registered(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        parsed = parser.parse_args(
            ["--model-path", "dummy", "--enable-dsa-cp-shared-kv-cache"]
        )

        assert parsed.enable_dsa_cp_shared_kv_cache

    def test_dsa_cp_shared_kvcache_defaults_off(self):
        assert not should_enable_dsa_cp_shared_kvcache(
            enable_hisparse=False,
            is_hip_platform=False,
        )

    def test_dsa_cp_shared_kvcache_enables_main_kv_only_when_supported(self):
        assert should_enable_dsa_cp_shared_kvcache(
            enable_hisparse=False,
            enabled=True,
            is_hip_platform=False,
        )
        assert not should_enable_dsa_cp_shared_kvcache(
            enable_hisparse=True,
            enabled=True,
            is_hip_platform=False,
        )
        assert not should_enable_dsa_cp_shared_kvcache(
            enable_hisparse=False,
            enabled=True,
            is_hip_platform=True,
        )

    def test_dsa_cp_shared_write_path_avoids_scalar_sync(self):
        names = MLATokenToKVPool.set_mla_kv_buffer.__code__.co_names
        assert "any" not in names
        assert "_owned_cp_shared_write_mask" not in names
        assert "_translate_cp_shared_write_loc" not in names
        assert "set_mla_kv_buffer_triton_cp_shared" in names

    def test_dsa_cp_shared_write_path_synchronizes_before_read(self):
        names = MLATokenToKVPool.set_mla_kv_buffer.__code__.co_names

        assert "set_mla_kv_buffer_triton_cp_shared" in names
        assert "synchronize_cp_shared_kv_write" in names

    def test_dsa_cp_shared_page_table_uses_int32_fast_path(self):
        consts = (
            DeepseekSparseAttnBackend._translate_page_table_for_main_kv.__code__.co_consts
        )
        assert "translate_loc_to_cp_shared_device_int32" in consts

    def test_dsa_cp_shared_flashmla_sparse_prefix_dequant_translates_page_table(
        self,
    ):
        source = inspect.getsource(DeepseekSparseAttnBackend.forward_extend)
        translate_pos = source.index(
            "page_table_1_flattened = self._translate_page_table_for_main_kv"
        )
        dequant_pos = source.index("dequantize_k_cache_paged")

        assert translate_pos < dequant_pos

    def test_dsa_cp_shared_defaults_hopper_bf16_prefill_to_flashmla_kv(self):
        args = _server_args_for_dsa_backend_defaults()

        args.enable_dsa_cp_shared_kv_cache = True
        args._set_default_dsa_backends("bfloat16", 9)

        assert args.dsa_prefill_backend == "flashmla_kv"
        assert args.dsa_decode_backend == "fa3"

    def test_dsa_cp_shared_defaults_hopper_fp8_prefill_to_flashmla_sparse(self):
        args = _server_args_for_dsa_backend_defaults(kv_cache_dtype="fp8_e4m3")

        args.enable_dsa_cp_shared_kv_cache = True
        args._set_default_dsa_backends("fp8_e4m3", 9)

        assert args.dsa_prefill_backend == "flashmla_sparse"
        assert args.dsa_decode_backend == "flashmla_kv"

    def test_dsa_cp_shared_server_arg_defaults_hopper_fp8_prefill_to_flashmla_sparse(
        self,
    ):
        args = _server_args_for_dsa_backend_defaults(kv_cache_dtype="fp8_e4m3")
        args.enable_dsa_cp_shared_kv_cache = True

        args._set_default_dsa_backends("fp8_e4m3", 9)

        assert args.dsa_prefill_backend == "flashmla_sparse"
        assert args.dsa_decode_backend == "flashmla_kv"

    def test_dsa_cp_shared_does_not_override_explicit_prefill_backend(self):
        args = _server_args_for_dsa_backend_defaults(
            dsa_prefill_backend="flashmla_sparse"
        )

        args.enable_dsa_cp_shared_kv_cache = True
        args._set_default_dsa_backends("bfloat16", 9)

        assert args.dsa_prefill_backend == "flashmla_sparse"
        assert args.dsa_decode_backend == "fa3"


class TestDsaCpSharedL1:
    def test_dsa_cp_shared_hicache_io_rank_translates_full_logical_indices(self):
        pool = _fake_shared_mla_pool(cp_rank=0, cp_size=4, page_size=2)
        logical_device_indices = torch.arange(16, dtype=torch.int64)
        host_indices = torch.arange(100, 116, dtype=torch.int64)

        host_out, device_shared = pool.get_cp_shared_hicache_transfer_indices(
            host_indices, logical_device_indices
        )

        assert host_out is host_indices
        assert torch.equal(
            device_shared,
            torch.tensor([0, 1, 16, 17, 32, 33, 48, 49, 2, 3, 18, 19, 34, 35, 50, 51]),
        )

    def test_dsa_cp_shared_hicache_non_io_rank_uses_empty_transfer(self):
        pool = _fake_shared_mla_pool(cp_rank=1, cp_size=4, page_size=2)
        logical_device_indices = torch.arange(16, dtype=torch.int64)
        host_indices = torch.arange(100, 116, dtype=torch.int64)

        host_out, device_out = pool.get_cp_shared_hicache_transfer_indices(
            host_indices, logical_device_indices
        )

        assert host_out.numel() == 0
        assert device_out.numel() == 0

    def test_dsa_cp_shared_l2_load_keeps_owned_transfer_on_non_io_rank(self):
        pool = _fake_shared_mla_pool(cp_rank=1, cp_size=4, page_size=2)
        logical_device_indices = torch.arange(16, dtype=torch.int64)
        host_indices = torch.arange(100, 116, dtype=torch.int64)

        host_out, device_out = pool.get_cp_shared_hicache_transfer_indices(
            host_indices, logical_device_indices, load_shared_l2=True
        )

        assert torch.equal(host_out, torch.tensor([102, 103, 110, 111]))
        assert torch.equal(device_out, torch.tensor([0, 1, 2, 3]))

    def test_dsa_cp_shared_hicache_transfer_noops_for_dense_pool(self):
        pool = object.__new__(MLATokenToKVPool)
        pool.enable_cp_shared_kvcache = False
        host_indices = torch.arange(4, dtype=torch.int64)
        device_indices = torch.arange(10, 14, dtype=torch.int64)

        host_out, device_out = pool.get_cp_shared_hicache_transfer_indices(
            host_indices, device_indices
        )

        assert host_out is host_indices
        assert device_out is device_indices


class TestDsaCpSharedHiCacheL2:
    def test_dsa_cp_shared_hicache_load_uses_translated_indices(self, monkeypatch):
        memory_pool_host = pytest.importorskip("sglang.srt.mem_cache.memory_pool_host")
        host = object.__new__(memory_pool_host.MLATokenToKVPoolHost)
        host.layout = "layer_first"
        host.can_use_jit = False
        host.kv_buffer = [torch.empty(1)]
        host.token_stride_size = 1
        host.kv_cache_dim = 1
        device_pool = _FakeSharedDevicePool()
        captured = {}

        def fake_transfer(**kwargs):
            captured["dst_indices"] = kwargs["dst_indices"]
            device_pool.calls.append("copy")

        monkeypatch.setattr(
            memory_pool_host, "transfer_kv_per_layer_mla", fake_transfer
        )

        host.load_to_device_per_layer(
            device_pool,
            torch.arange(2, dtype=torch.int64),
            torch.arange(2, dtype=torch.int64),
            0,
            "kernel",
        )

        assert device_pool.calls == ["translate", "copy", "sync"]
        assert torch.equal(captured["dst_indices"], torch.tensor([1000, 1001]))

    def test_dsa_cp_shared_hicache_backup_syncs_around_translated_copy(
        self, monkeypatch
    ):
        memory_pool_host = pytest.importorskip("sglang.srt.mem_cache.memory_pool_host")
        host = object.__new__(memory_pool_host.MLATokenToKVPoolHost)
        host.layout = "layer_first"
        host.can_use_jit = False
        host.data_ptrs = torch.empty(1, dtype=torch.uint64)
        host.token_stride_size = 1
        host.kv_cache_dim = 1
        host.layer_num = 1
        device_pool = _FakeSharedDevicePool()
        captured = {}

        def fake_transfer(**kwargs):
            captured["src_indices"] = kwargs["src_indices"]
            device_pool.calls.append("copy")

        monkeypatch.setattr(
            memory_pool_host, "transfer_kv_all_layer_mla", fake_transfer
        )

        host.backup_from_device_all_layer(
            device_pool,
            torch.arange(2, dtype=torch.int64),
            torch.arange(2, dtype=torch.int64),
            "kernel",
        )

        assert device_pool.calls == ["sync", "translate", "copy", "sync"]
        assert torch.equal(captured["src_indices"], torch.tensor([1000, 1001]))

    def test_dsa_cp_shared_hicache_enabled_requires_shared_l2_host(self):
        controller = object.__new__(HiCacheController)
        controller.mem_pool_device = _FakeSharedHiCacheDevicePool(is_io_rank=True)
        controller.mem_pool_host = _FakeHostPool(None)

        assert not controller._cp_shared_hicache_enabled()

    def test_dsa_cp_shared_hicache_io_rank_uses_shared_l2_host_owner(self):
        controller = object.__new__(HiCacheController)
        controller.mem_pool_device = _FakeSharedHiCacheDevicePool(is_io_rank=True)
        controller.mem_pool_host = _mark_fake_shared_l2_host(
            _FakeHostPool(None), is_owner=False
        )

        assert controller._cp_shared_hicache_enabled()
        assert not controller._cp_shared_hicache_is_io_rank()

    def test_dsa_cp_shared_hicache_follower_write_shadow_allocates(self):
        local_indices = torch.tensor([10, 11], dtype=torch.int64)
        controller = _fake_hicache_controller(
            is_io_rank=False, alloc_result=local_indices
        )

        result = controller.write(torch.arange(2, dtype=torch.int64), node_id=7)

        assert torch.equal(result, local_indices)
        assert controller.mem_pool_host.alloc_calls == [2]
        assert len(controller.write_queue) == 1
        assert torch.equal(controller.write_queue[0].host_indices, local_indices)
        assert controller.started == 1

    def test_dsa_cp_shared_hicache_write_failure_stays_local(self):
        controller = _fake_hicache_controller(is_io_rank=True, alloc_result=None)

        result = controller.write(torch.arange(2, dtype=torch.int64), node_id=7)

        assert result is None
        assert controller.mem_pool_host.alloc_calls == [2]
        assert controller.write_queue == []
        assert controller.started == 0

    def test_dsa_cp_shared_hicache_follower_host_evict_frees_shadow_state(self):
        controller = _fake_hicache_controller(is_io_rank=False)
        indices = torch.tensor([10, 11], dtype=torch.int64)

        assert controller.evict_host(indices) == 2
        assert len(controller.mem_pool_host.free_calls) == 1
        assert torch.equal(controller.mem_pool_host.free_calls[0], indices)

    def test_dsa_cp_shared_hicache_host_available_size_is_local(self):
        controller = _fake_hicache_controller(is_io_rank=False, available_result=1024)

        assert controller.host_available_size() == 1024
        assert controller.mem_pool_host.available_calls == 1

    def test_dsa_cp_shared_hicache_hot_paths_do_not_broadcast_counts(self):
        assert "_sync_cp_shared_hicache_count" not in inspect.getsource(
            HiCacheController.write
        )
        assert "_sync_cp_shared_hicache_count" not in inspect.getsource(
            HiCacheController.backup_thread_func
        )
        assert "_sync_cp_shared_hicache_count" not in inspect.getsource(
            HybridCacheController._page_transfer
        )
        assert "_sync_cp_shared_hicache_count" not in inspect.getsource(
            HybridCacheController._page_backup
        )

    def test_dsa_cp_shared_write_backup_uses_synchronized_host_available_size(self):
        source = inspect.getsource(UnifiedRadixCache.write_backup)

        assert "self.cache_controller.host_available_size()" in source

    def test_dsa_cp_shared_hybrid_follower_l3_query_skips_storage_io(self):
        controller = object.__new__(HybridCacheController)
        controller.mem_pool_device = _FakeSharedHiCacheDevicePool(is_io_rank=False)
        controller.mem_pool_host = _mark_fake_shared_l2_host(
            _FakeHostPool(None), is_owner=False
        )
        controller.page_size = 2
        controller.get_hash_str = lambda tokens, last_hash, **kwargs: ["h0", "h1", "h2"]

        class _NoStorage:
            def batch_exists(self, *args, **kwargs):
                raise AssertionError("follower should not query L3 storage")

            def batch_exists_v2(self, *args, **kwargs):
                raise AssertionError("follower should not query L3 storage")

        controller.storage_backend = _NoStorage()
        operation = type(
            "FakeOperation",
            (),
            {
                "last_hash": None,
                "token_ids": [1, 2, 3, 4, 5, 6],
                "prefix_keys": None,
                "pool_transfers": None,
                "pool_storage_result": PoolTransferResult.empty(),
            },
        )()

        hashes, hit_tokens = controller._storage_hit_query(operation)

        assert hit_tokens == 6
        assert len(hashes) == 3
        assert operation.pool_storage_result.kv_hit_pages == 3

    def test_dsa_cp_shared_hybrid_follower_l3_query_marks_sidecars_hit(self):
        from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer

        controller = object.__new__(HybridCacheController)
        controller.mem_pool_device = _FakeSharedHiCacheDevicePool(is_io_rank=False)
        controller.mem_pool_host = _mark_fake_shared_l2_host(
            _FakeHostPool(None), is_owner=False
        )
        controller.page_size = 2
        controller.get_hash_str = lambda tokens, last_hash, **kwargs: ["h0", "h1", "h2"]
        operation = type(
            "FakeOperation",
            (),
            {
                "last_hash": None,
                "token_ids": [1, 2, 3, 4, 5, 6],
                "prefix_keys": None,
                "pool_transfers": [PoolTransfer(name=PoolName.INDEXER)],
                "pool_storage_result": PoolTransferResult.empty(),
            },
        )()

        hashes, hit_tokens = controller._storage_hit_query(operation)

        assert hit_tokens == 6
        assert len(hashes) == 3
        assert operation.pool_storage_result.extra_pool_hit_pages[PoolName.INDEXER] == 3

    def test_dsa_cp_shared_hicache_follower_l3_get_skips_storage_io(self):
        controller = _fake_hicache_controller(is_io_rank=False)
        controller.page_size = 2

        class _NoStorage:
            def batch_get_v1(self, *args, **kwargs):
                raise AssertionError("follower should not read L3 storage")

        class _Operation:
            request_id = "req"

            def __init__(self):
                self.completed_tokens = 0

            def increment(self, inc):
                self.completed_tokens += inc
                return True

        controller.storage_backend = _NoStorage()
        operation = _Operation()

        controller._page_get_zero_copy(
            operation,
            ["a", "b"],
            torch.arange(4, dtype=torch.int64),
        )

        assert operation.completed_tokens == 4


class TestDsaCpSharedAllocator:
    def test_dsa_cp_shared_host_allocator_owner_creates_segment(self, monkeypatch):
        from sglang.srt.mem_cache.pool_host import common

        broadcasts = []

        class _FakeSharedMemory:
            def __init__(self, create=False, size=0, name=None):
                self.name = name or "sglang-test-shm"
                self.size = size
                self.buf = bytearray(size)
                self.closed = False
                self.unlinked = False

            def close(self):
                self.closed = True

            def unlink(self):
                self.unlinked = True

        def fake_broadcast_object_list(obj, src, group):
            broadcasts.append((list(obj), src, group))

        monkeypatch.setattr(common.shared_memory, "SharedMemory", _FakeSharedMemory)
        monkeypatch.setattr(common.torch.distributed, "get_rank", lambda group=None: 0)
        monkeypatch.setattr(
            common.torch.distributed,
            "get_process_group_ranks",
            lambda group: [10, 11],
        )
        monkeypatch.setattr(
            common.torch.distributed,
            "broadcast_object_list",
            fake_broadcast_object_list,
        )

        allocator = common.CpSharedHostTensorAllocator(
            cpu_group="cp", owner_rank=0, kind="unit"
        )
        tensor = allocator.allocate((4,), torch.float32, "cpu")

        assert allocator.is_owner
        assert tensor.shape == (4,)
        assert tensor.dtype == torch.float32
        assert broadcasts[0][1] == 10
        allocator.destroy()

    def test_dsa_cp_shared_host_allocator_follower_attaches_segment(self, monkeypatch):
        from sglang.srt.mem_cache.pool_host import common

        class _FakeSharedMemory:
            def __init__(self, create=False, size=0, name=None):
                assert not create
                assert name == "owner-name"
                self.name = name
                self.size = 16
                self.buf = bytearray(16)

            def close(self):
                pass

        def fake_broadcast_object_list(obj, src, group):
            obj[0] = {"name": "owner-name", "nbytes": 16}

        monkeypatch.setattr(common.shared_memory, "SharedMemory", _FakeSharedMemory)
        monkeypatch.setattr(common.torch.distributed, "get_rank", lambda group=None: 1)
        monkeypatch.setattr(
            common.torch.distributed,
            "get_process_group_ranks",
            lambda group: [10, 11],
        )
        monkeypatch.setattr(
            common.torch.distributed,
            "broadcast_object_list",
            fake_broadcast_object_list,
        )

        allocator = common.CpSharedHostTensorAllocator(
            cpu_group="cp", owner_rank=0, kind="unit"
        )
        tensor = allocator.allocate((4,), torch.float32, "cpu")

        assert not allocator.is_owner
        assert tensor.shape == (4,)

    def test_dsa_cp_shared_host_allocator_logs_create_or_attach(self, monkeypatch):
        from sglang.srt.mem_cache.pool_host import common

        monkeypatch.setattr(
            common.torch.distributed,
            "get_process_group_ranks",
            lambda group: [10, 11],
        )

        messages = []
        logger = type(
            "Logger",
            (),
            {"info": lambda self, msg, *args: messages.append(msg % args)},
        )()

        monkeypatch.setattr(common.torch.distributed, "get_rank", lambda group=None: 0)
        owner = common.CpSharedHostTensorAllocator(
            cpu_group="cp", owner_rank=0, kind="unit"
        )
        owner.log_host_allocation(
            16,
            logger,
            pool_name="hierarchical KV cache",
            token_capacity=128,
            page_num=2,
            page_size=64,
        )

        monkeypatch.setattr(common.torch.distributed, "get_rank", lambda group=None: 1)
        follower = common.CpSharedHostTensorAllocator(
            cpu_group="cp", owner_rank=0, kind="unit"
        )
        follower.log_host_allocation(
            16,
            logger,
            pool_name="DSA indexer",
            token_capacity=128,
            page_num=2,
            page_size=64,
        )

        assert "create 0.00 GB host memory for hierarchical KV cache" in messages[0]
        assert "cp_rank=0 owner_rank=0 is_owner=True" in messages[0]
        assert "shared_group_key=unit:0:10,11" in messages[0]
        assert "attach 0.00 GB host memory for DSA indexer" in messages[1]
        assert "cp_rank=1 owner_rank=0 is_owner=False" in messages[1]
        assert "shared_group_key=unit:0:10,11" in messages[1]

    def test_dsa_cp_shared_host_allocator_logs_segment_name(self, monkeypatch, caplog):
        from sglang.srt.mem_cache.pool_host import common

        class _FakeSharedMemory:
            def __init__(self, create=False, size=0, name=None):
                self.name = name or "sgl_shm_unit_1234_abcd"
                self.size = size
                self.buf = bytearray(size)

            def close(self):
                pass

            def unlink(self):
                pass

        def fake_broadcast_object_list(obj, src, group):
            if obj[0] is None:
                obj[0] = {"name": "sgl_shm_unit_1234_abcd", "nbytes": 16}

        monkeypatch.setattr(common.shared_memory, "SharedMemory", _FakeSharedMemory)
        monkeypatch.setattr(
            common, "make_shm_name", lambda kind: "sgl_shm_unit_1234_abcd"
        )
        monkeypatch.setattr(common.torch.distributed, "get_rank", lambda group=None: 0)
        monkeypatch.setattr(
            common.torch.distributed,
            "get_process_group_ranks",
            lambda group: [10, 11],
        )
        monkeypatch.setattr(
            common.torch.distributed,
            "broadcast_object_list",
            fake_broadcast_object_list,
        )

        allocator = common.CpSharedHostTensorAllocator(
            cpu_group="cp", owner_rank=0, kind="unit"
        )
        with caplog.at_level("INFO", logger=common.__name__):
            allocator.allocate((4,), torch.float32, "cpu")

        assert "DSA CP shared L2 host memory segment: create" in caplog.text
        assert "shm_name=sgl_shm_unit_1234_abcd" in caplog.text
        assert "shared_group_key=unit:0:10,11" in caplog.text

    def test_dsa_cp_shared_host_cache_destroy_releases_allocator(self):
        from sglang.srt.mem_cache.pool_host.base import HostKVCache

        class _FakeAllocator:
            def __init__(self):
                self.destroyed = 0

            def destroy(self):
                self.destroyed += 1

        class _FakeHostCache(HostKVCache):
            def get_size_per_token(self):
                return 1

            def init_kv_buffer(self):
                return torch.empty(1)

            def load_to_device_per_layer(self, *args, **kwargs):
                pass

            def backup_from_device_all_layer(self, *args, **kwargs):
                pass

            def get_data_page(self, *args, **kwargs):
                pass

            def get_dummy_flat_data_page(self):
                pass

            def set_from_flat_data_page(self, *args, **kwargs):
                pass

        host = object.__new__(_FakeHostCache)
        allocator = _FakeAllocator()
        host.allocator = allocator
        host.pin_memory = False
        host.kv_buffer = torch.empty(1)

        HostKVCache.destroy(host)

        assert allocator.destroyed == 1

    def test_dsa_cp_shared_l2_builds_dsa_host_pools_with_shared_allocator(
        self, monkeypatch
    ):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa
        from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost

        assert "allocator" in inspect.signature(MLATokenToKVPoolHost).parameters

        captured = {}

        class _FakeAllocator:
            pass

        class _FakeKvHostPool:
            def __init__(self, *args, allocator=None, **kwargs):
                captured["kv_allocator"] = allocator
                self.page_size = 64
                self.size = 128
                self.page_num = 3
                self.layout = "page_first"
                self.device = "cpu"
                self.size_per_token = 1
                self.can_use_write_back_jit = False

        fake_allocator = _FakeAllocator()
        monkeypatch.setattr(hpa, "MLATokenToKVPoolHost", _FakeKvHostPool)

        server_args = argparse.Namespace(
            hicache_ratio=1.0,
            hicache_size=0,
            hicache_mem_layout="page_first",
            hicache_storage_backend="default",
        )
        kv_pool = type("FakePool", (), {"store_dtype": torch.float8_e4m3fn})()

        hpa.build_kv_host_pool(
            kv_pool=kv_pool,
            page_size=64,
            server_args=server_args,
            use_mla=True,
            override_kv_cache_dim=128,
            allocator=fake_allocator,
        )

        assert captured["kv_allocator"] is fake_allocator

    def test_dsa_cp_shared_l2_allocator_uses_cp_cache_group(self, monkeypatch):
        from sglang.srt.distributed import parallel_state
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa
        from sglang.srt.mem_cache.pool_host import common

        captured = {}

        class _FakeAllocator:
            def __init__(self, cpu_group, owner_rank, kind):
                captured["cpu_group"] = cpu_group
                captured["owner_rank"] = owner_rank
                captured["kind"] = kind

        monkeypatch.setattr(
            parallel_state, "in_the_same_node_as", lambda *a, **k: [True]
        )
        monkeypatch.setattr(common, "CpSharedHostTensorAllocator", _FakeAllocator)

        allocator = hpa._maybe_create_dsa_cp_shared_l2_allocator(
            params=argparse.Namespace(attn_cp_cache_group="cp", tp_cache_group="tp"),
            server_args=argparse.Namespace(enable_dsa_cp_shared_kv_cache=True),
            kv_pool=argparse.Namespace(enable_cp_shared_kvcache=True),
        )

        assert isinstance(allocator, _FakeAllocator)
        assert captured == {
            "cpu_group": "cp",
            "owner_rank": 0,
            "kind": "dsa_l2",
        }

    def test_dsa_cp_shared_l2_hiradix_attach_passes_shared_allocator(self, monkeypatch):
        from sglang.srt.mem_cache.hicache_storage import PoolName
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa

        captured = {}
        fake_allocator = object()

        monkeypatch.setattr(
            hpa,
            "_maybe_create_dsa_cp_shared_l2_allocator",
            lambda **kwargs: fake_allocator,
        )

        class _FakeHostPoolGroup:
            def get_pool(self, name):
                assert name == PoolName.KV
                return "kv_host_pool"

        def _fake_build_anchor_sidecar_stack(**kwargs):
            captured["allocator"] = kwargs["allocator"]
            captured["sidecar_pool"] = kwargs["sidecar_host_pool_factory"](
                "kv_host_pool"
            )
            return _FakeHostPoolGroup(), "cache_controller"

        def _fake_indexer_pool_host(*args, allocator=None, **kwargs):
            captured["indexer_allocator"] = allocator
            return "indexer_host_pool"

        monkeypatch.setattr(
            hpa, "build_anchor_sidecar_stack", _fake_build_anchor_sidecar_stack
        )
        monkeypatch.setattr(hpa, "DSAIndexerPoolHost", _fake_indexer_pool_host)

        kv = argparse.Namespace(layer_num=2, kv_cache_dim=128)
        radix_cache = argparse.Namespace(
            kv_cache=kv,
            page_size=64,
            tp_group="tp",
            pp_group="pp",
        )
        server_args = argparse.Namespace(
            hicache_mem_layout="page_first",
            hicache_storage_backend="file",
            served_model_name="model",
        )

        hpa.attach_hybrid_dsa_pool_to_hiradix_cache(
            radix_cache,
            params=argparse.Namespace(),
            server_args=server_args,
            extra_config={},
            prefetch_threshold=256,
            enable_storage_metrics=False,
            load_cache_event=None,
        )

        assert captured["allocator"] is fake_allocator
        assert captured["indexer_allocator"] is fake_allocator
        assert captured["sidecar_pool"] == "indexer_host_pool"
        assert radix_cache.full_kv_pool_host == "kv_host_pool"
        assert radix_cache.token_to_kv_pool_host.get_pool(PoolName.KV) == "kv_host_pool"
        assert radix_cache.cache_controller == "cache_controller"

    def test_dsa_cp_shared_l2_anchor_sidecar_stack_marks_shared_group(
        self, monkeypatch
    ):
        from sglang.srt.mem_cache.hicache_storage import PoolName
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa

        class _FakeAllocator:
            pass

        class _FakeKvHostPool:
            def __init__(self, *args, allocator=None, **kwargs):
                self.allocator = allocator
                self.page_size = 64
                self.size = 128
                self.page_num = 3
                self.layout = "page_first"
                self.device = "cpu"
                self.size_per_token = 1
                self.can_use_write_back_jit = False
                self.dtype = torch.float8_e4m3fn
                self.start_layer = 0
                self.end_layer = 1

            def get_ksize_per_token(self):
                return 1

        class _FakeController:
            def __init__(self, *args, **kwargs):
                pass

        fake_allocator = _FakeAllocator()
        monkeypatch.setattr(hpa, "MLATokenToKVPoolHost", _FakeKvHostPool)
        monkeypatch.setattr(hpa, "HybridCacheController", _FakeController)

        server_args = argparse.Namespace(
            hicache_ratio=1.0,
            hicache_size=0,
            hicache_mem_layout="page_first",
            hicache_storage_backend="default",
            hicache_write_policy="write_through_selective",
            hicache_io_backend="kernel",
        )
        params = argparse.Namespace(token_to_kv_pool_allocator=object())
        kv_pool = type(
            "FakePool",
            (),
            {
                "store_dtype": torch.float8_e4m3fn,
                "cp_shared_cp_rank": 1,
            },
        )()

        host_pool_group, _ = hpa.build_anchor_sidecar_stack(
            params=params,
            server_args=server_args,
            kv_pool=kv_pool,
            sidecar_pool_name=PoolName.INDEXER,
            full_layer_mapping={0: 0},
            page_size=64,
            tp_group=None,
            load_cache_event=None,
            storage_backend=None,
            use_mla=True,
            override_kv_cache_dim=128,
            sidecar_host_pool_factory=lambda kv_host_pool: kv_host_pool,
            allocator=fake_allocator,
        )

        assert host_pool_group.enable_cp_shared_dsa_l2
        assert host_pool_group.cp_shared_l2_cp_rank == 1
        assert host_pool_group.anchor_entry.host_pool.allocator is fake_allocator

    def test_dsa_cp_shared_l2_host_group_marks_owner_mode(self):
        from sglang.srt.mem_cache.hicache_storage import PoolName
        from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry

        host = type(
            "Host",
            (),
            {
                "layout": "page_first",
                "page_size": 64,
                "device": "cpu",
                "size": 128,
                "can_use_write_back_jit": False,
                "size_per_token": 1,
                "allocator": None,
                "dtype": torch.float8_e4m3fn,
                "start_layer": 0,
                "end_layer": 1,
                "get_ksize_per_token": lambda self: 1,
            },
        )()
        entry = PoolEntry(
            name=PoolName.KV,
            host_pool=host,
            device_pool=object(),
            layer_mapper=lambda layer_id: layer_id,
            is_primary_index_anchor=True,
        )

        group = HostPoolGroup(
            [entry], enable_cp_shared_dsa_l2=True, owner_rank=0, cp_rank=0
        )

        assert group.enable_cp_shared_dsa_l2
        assert group.cp_shared_l2_is_owner()

    def test_dsa_cp_shared_l2_follower_skips_backup_payload(self):
        host = _CountingSharedL2Host()
        device_pool = _FakeSharedDevicePool()
        group = _shared_l2_group(host, cp_rank=1, device_pool=device_pool)

        group.backup_from_device_all_layer(
            object(), torch.arange(2), torch.arange(2), "kernel"
        )

        assert host.backup_calls == []
        assert device_pool.calls == ["sync", "sync"]

    def test_dsa_cp_shared_l2_follower_still_loads_payload(self):
        host = _CountingSharedL2Host()
        group = _shared_l2_group(host, cp_rank=1)

        group.load_to_device_per_layer(
            object(), torch.arange(2), torch.arange(2), 0, "kernel"
        )

        assert len(host.load_calls) == 1
        assert host.load_calls[0][1]["load_shared_l2"]
