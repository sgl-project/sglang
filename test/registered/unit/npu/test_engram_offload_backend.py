"""CPU tests for MemFabric Engram addressing, checkpoint loading and graph buffers."""

import contextlib
import ctypes
import mmap
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
from sglang.srt.arg_groups.engram_config import (
    EngramConfig,
    EngramPlacement,
    EngramPoolConfig,
    EngramReplication,
)
from sglang.srt.hardware_backend.npu.modules import engram_offload
from sglang.srt.layers import engram as engram_mod
from sglang.srt.layers.engram import EngramEmbedding, EngramLayout
from sglang.test.test_utils import CustomTestCase

_HEAD_DIM = 64
_N_HASH_COLS = 4  # (max_ngram_size - 1) * n_heads
_NUM_EMBEDDINGS = 1000
_SLOT_GIB = 1 << 30

_LAYOUT = EngramLayout(
    max_ngram_size=3,
    layer_ids=(5, 9),
    num_embeddings=(_NUM_EMBEDDINGS, 701),
    primes=(((11, 13), (17, 19)), ((23, 29), (31, 37))),
    n_heads=2,
    head_dim=_HEAD_DIM,
)

_NODE_LOCAL_ENV_KEYS = (
    "SGLANG_NODE_LOCAL_WORLD",
    "SGLANG_NODE_LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "LOCAL_RANK",
)


class _FakeScene:
    LOCAL = 0
    SHARED = 1


class _FakeOffloadConfig:
    def __init__(self):
        self.device_id = 0
        self.reserve_size = 0
        self.alloc_size = 0
        self.world_size = 1
        self.rank_id = 0
        self.scene = _FakeScene.LOCAL
        self.flags = 0
        self.store_url = ""


class _FakeOffloadEntity:
    """In-process pool using the native slot-segmented row addresses."""

    Scene = _FakeScene
    OffloadConfig = _FakeOffloadConfig

    def __init__(self):
        self._config = None
        self._tables = []
        self._buf = None
        self._slot = 0
        self.uninitialized = False

    def initialize(self, config):
        assert self._config is None, "fake entity initialized twice"
        self._config = config
        total = config.reserve_size * config.world_size
        # Keep real slot strides without touching GiB of physical memory per test.
        self._mapping = mmap.mmap(-1, total)
        self._buf = (ctypes.c_char * total).from_buffer(self._mapping)
        self._slot = config.reserve_size
        return 0

    def malloc(self, size, flags=0):
        assert self._buf is not None and size <= self._slot
        return ctypes.addressof(self._buf) + self._config.rank_id * self._slot

    def register_entry_table(self, entry_bytes, rows):
        assert not self._tables, "only one uniform grid may be registered"
        self._tables.append((entry_bytes, rows))
        return 0

    def entry_gather(self, dst, ids, count, device):
        entry_bytes, rows = self._tables[0]
        n = int(count) if not torch.is_tensor(count) else int(count.item())
        dst_np = dst.numpy()
        for j, row in enumerate(ids[:n].to(torch.int64).cpu().tolist()):
            slot_i, in_slot = divmod(row, rows)
            start = slot_i * self._slot + in_slot * entry_bytes
            dst_np[j * entry_bytes : (j + 1) * entry_bytes] = np.frombuffer(
                self._buf, dtype=np.uint8, count=entry_bytes, offset=start
            )
        return 0

    def get_last_err_msg(self):
        return "fake memfabric error detail"

    def uninitialize(self):
        self.uninitialized = True


def _fake_memfabric(multi_instance: bool) -> types.ModuleType:
    default = _FakeOffloadEntity()
    if multi_instance:
        default.Offload = _FakeOffloadEntity
    module = types.ModuleType("memfabric_hybrid")
    module.offload = default
    return module


@contextlib.contextmanager
def _without_node_local_env():
    """Isolate the worker-coordinate environment from the test launcher."""
    saved = {k: os.environ.pop(k) for k in _NODE_LOCAL_ENV_KEYS if k in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def _default_parallel(**overrides):
    fields = dict(tp_size=4, tp_rank=0, pp_size=1, nnodes=1, dp_size=1, dp_rank=None)
    fields.update(overrides)
    return SimpleNamespace(**fields)


_WORLD_ENV = {"SGLANG_NODE_LOCAL_WORLD": "4", "SGLANG_NODE_LOCAL_RANK": "1"}


class _OffloadHarness:
    """Isolate MemFabric, runtime configuration and the process-wide manager."""

    def __init__(
        self,
        multi_instance: bool = True,
        parallel=None,
        chunked_prefill_size: int = 512,
        decode_max_bs: int = 64,
        decode_width: int = 1,
        node_local_env: dict | None = None,
        pool: EngramPoolConfig | None = None,
    ):
        self.module = _fake_memfabric(multi_instance)
        self.parallel = parallel or _default_parallel()
        self.schedule = SimpleNamespace(chunked_prefill_size=chunked_prefill_size)
        self.exec_ctx = SimpleNamespace(
            offload=SimpleNamespace(
                engram_config=EngramConfig(pool=pool or EngramPoolConfig())
            ),
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(max_bs=decode_max_bs)
                )
            ),
        )
        self.node_local_env = node_local_env
        self.decode_width = decode_width

    def __enter__(self):
        self._stack = contextlib.ExitStack()
        stack = self._stack
        stack.enter_context(patch.object(engram_offload.atexit, "register"))
        stack.enter_context(
            patch.object(
                engram_offload,
                "max_speculative_num_draft_tokens",
                return_value=self.decode_width,
            )
        )
        stack.enter_context(patch.dict(sys.modules, {"memfabric_hybrid": self.module}))
        stack.enter_context(
            patch.object(engram_offload, "get_parallel", return_value=self.parallel)
        )
        stack.enter_context(
            patch.object(engram_offload, "get_schedule", return_value=self.schedule)
        )
        stack.enter_context(
            patch.object(engram_offload, "get_exec", return_value=self.exec_ctx)
        )
        stack.enter_context(
            patch.object(engram_offload, "_current_device_id", return_value=0)
        )
        stack.enter_context(patch.object(engram_offload, "_MANAGER", None))
        stack.enter_context(
            patch.object(engram_offload, "is_ascend_910c", return_value=False)
        )
        stack.enter_context(_without_node_local_env())
        if self.node_local_env is not None:
            stack.enter_context(patch.dict(os.environ, self.node_local_env))
        return self

    def __exit__(self, *exc_info):
        self._stack.close()
        return False


def _open_manager(layout=_LAYOUT):
    return engram_offload.get_engram_offload_manager(layout)


def _write_all_chunks(entity, manager, table, src_u8: np.ndarray) -> None:
    """Populate every owner slot for full-table gather tests."""
    pool = np.frombuffer(entity._buf, dtype=np.uint8)
    for chunk_rank, pool_rank in enumerate(manager.owner_ranks):
        lo = chunk_rank * table.rows
        hi = min(lo + table.rows, table.num_embeddings)
        if lo >= hi:
            break
        start = pool_rank * manager.slot + table.block_offset
        n = (hi - lo) * table.entry_bytes
        pool[start : start + n] = src_u8.reshape(-1)[
            lo * table.entry_bytes : hi * table.entry_bytes
        ].reshape(-1)


def _make_backend(holder: nn.Module, layer_id: int = 5):
    backend = engram_offload.NpuEngramOffloadBackend(_LAYOUT, _HEAD_DIM, layer_id)
    backend.install_params(holder)
    return backend


class TestPoolOpenAndAddressing(CustomTestCase):
    def test_chunk_math_and_full_range_addressing(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            manager = _open_manager()
            self.assertEqual((manager.local_world, manager.rank), (4, 1))
            t0, t1 = manager.table(0), manager.table(1)
            # ceil sharding: 1000/4 and ceil(701/4)
            self.assertEqual((t0.rows, t0.block_offset), (250, 0))
            self.assertEqual((t1.rows, t1.block_offset), (176, 250 * _HEAD_DIM))
            self.assertEqual(manager.slot, _SLOT_GIB)
            self.assertEqual(t0.entry_bytes, _HEAD_DIM)

            # every row of both tables, gathered back through the fake kernel,
            # matches the source bytes: covers chunk boundaries (249/250,
            # 499/500, ...) and proves the slot tail gap is never read
            for ti, e in enumerate(_LAYOUT.num_embeddings):
                table = manager.table(ti)
                src = np.fromiter(
                    (
                        (ti * 7 + r + j) % 251
                        for r in range(e)
                        for j in range(_HEAD_DIM)
                    ),
                    dtype=np.uint8,
                ).reshape(e, _HEAD_DIM)
                _write_all_chunks(manager._entity, manager, table, src)
                ids = torch.arange(e, dtype=torch.int64)
                staging = torch.empty(e * _HEAD_DIM, dtype=torch.uint8)
                self.assertEqual(
                    manager.entry_gather(
                        staging,
                        manager.encode_ids(ids, table),
                        torch.tensor(e, dtype=torch.int32),
                        torch.device("cpu"),
                    ),
                    0,
                )
                np.testing.assert_array_equal(
                    staging.numpy().reshape(e, _HEAD_DIM), src
                )

    def test_dedicated_entity_on_multi_instance_wheel(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV) as h:
            manager = _open_manager()
            # the engram pool gets its own entity, not the module-level
            # default the HiCache L2 host pool would initialize
            self.assertIsNot(manager._entity, h.module.offload)

    def test_single_node_keeps_legacy_config_compatible(self):
        defaults = vars(_FakeOffloadConfig())
        legacy_type = type("LegacyConfig", (), {"__slots__": tuple(defaults)})

        def make_config():
            config = legacy_type()
            for key, value in defaults.items():
                setattr(config, key, value)
            return config

        with (
            _OffloadHarness(multi_instance=False, node_local_env=_WORLD_ENV) as h,
            patch.object(engram_offload, "_GIB", 1 << 16),
        ):
            h.module.offload.OffloadConfig = make_config
            manager = _open_manager()
            self.assertEqual(manager._entity._config.world_size, 4)
            self.assertFalse(hasattr(manager._entity._config, "local_world_size"))

    def test_store_url_reaches_the_pool_config(self):
        with (
            _OffloadHarness(
                node_local_env=_WORLD_ENV,
                pool=EngramPoolConfig(store_url="tcp://127.0.0.1:8500"),
            ),
        ):
            _open_manager()
            config = engram_offload._MANAGER._entity._config
            self.assertEqual(config.store_url, "tcp://127.0.0.1:8500")
            self.assertEqual(config.scene, _FakeScene.SHARED)
            self.assertEqual(config.flags, 0)
            self.assertEqual(config.world_size, 4)
            self.assertEqual(config.rank_id, 1)
            self.assertEqual(config.alloc_size, config.reserve_size)


class TestPoolConfig(CustomTestCase):
    def test_pool_config_rejects_invalid_fields_and_topologies(self):
        for pool in (
            None,
            [],
            {"world_size": "8"},
            {"world_size": True},
            {"unknown": 1},
            {"owner_ranks": "0,1"},
            {"owner_ranks": []},
            {"owner_ranks": [0, True]},
            {"owner_ranks": [0, 0]},
            {"owner_ranks": [-1]},
            {"world_size": -1},
            {"rank_offset": -1},
            {"world_size": 2, "owner_ranks": [2]},
            {"world_size": 2, "rank_offset": 2},
        ):
            with self.subTest(pool=pool), self.assertRaises(ValueError):
                EngramConfig.from_dict({"placement": "cpu", "pool": pool})
        for raw in (
            {"pool": {}},
            {"placement": "cpu", "replication": "per_node", "pool": {"world_size": 8}},
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                EngramConfig.from_dict(raw)


class TestPoolCoords(CustomTestCase):
    def test_cross_instance_consumer_reads_noncontiguous_owners(self):
        store = Mock()
        store.get.side_effect = lambda key: store.set.call_args_list[0].args[1]
        with (
            _OffloadHarness(
                node_local_env=_WORLD_ENV,
                pool=EngramPoolConfig(
                    store_url="tcp://127.0.0.1:18500", world_size=8, owner_ranks=[4, 6]
                ),
            ),
            patch.object(engram_offload, "_GIB", 1 << 16),
            patch.object(torch.distributed, "TCPStore", return_value=store),
        ):
            manager = _open_manager()
            self.assertIsNone(manager.owner_index)
            self.assertEqual(manager._entity._config.alloc_size, 0)
            self.assertEqual(manager._entity._config.local_world_size, 4)
            for index, size in enumerate(_LAYOUT.num_embeddings):
                table = manager.table(index)
                src = (np.arange(size * _HEAD_DIM) % 251).astype(np.uint8)
                _write_all_chunks(manager._entity, manager, table, src)
                ids = torch.tensor([0, table.rows - 1, table.rows, size - 1])
                dst = torch.empty(ids.numel() * _HEAD_DIM, dtype=torch.uint8)
                self.assertEqual(
                    manager.entry_gather(
                        dst,
                        manager.encode_ids(ids, table),
                        torch.tensor(ids.numel(), dtype=torch.int32),
                        torch.device("cpu"),
                    ),
                    0,
                )
                np.testing.assert_array_equal(
                    dst.numpy().reshape(-1, _HEAD_DIM),
                    src.reshape(size, _HEAD_DIM)[ids.numpy()],
                )
            manager.finish_load(1)
            store.wait.assert_called_with(["loaded/1/4", "loaded/1/6"])

    def test_owner_slice_uses_owner_index_instead_of_pool_rank(self):
        store = Mock()
        store.get.side_effect = lambda key: store.set.call_args_list[0].args[1]
        with (
            _OffloadHarness(
                node_local_env={
                    "SGLANG_NODE_LOCAL_WORLD": "4",
                    "SGLANG_NODE_LOCAL_RANK": "2",
                },
                pool=EngramPoolConfig(
                    store_url="tcp://127.0.0.1:18500",
                    world_size=8,
                    rank_offset=4,
                    owner_ranks=[4, 6],
                ),
            ),
            patch.object(engram_offload, "_GIB", 1 << 16),
            patch.object(torch.distributed, "TCPStore", return_value=store),
        ):
            backend = _make_backend(nn.Module())
            self.assertEqual(
                (backend._manager.rank, backend._manager.owner_index), (6, 1)
            )
            self.assertEqual((backend.rows, backend.row_start), (500, 500))

    def test_coord_fallback_order(self):
        cases = [
            (
                {
                    "SGLANG_NODE_LOCAL_WORLD": "8",
                    "SGLANG_NODE_LOCAL_RANK": "3",
                    "LOCAL_WORLD_SIZE": "4",
                    "LOCAL_RANK": "1",
                },
                (8, 3),
            ),
            ({"LOCAL_WORLD_SIZE": "4", "LOCAL_RANK": "2"}, (4, 0)),
            (None, (4, 0)),
        ]
        for env, expected in cases:
            with self.subTest(env=env):
                with _OffloadHarness(node_local_env=env):
                    self.assertEqual(
                        engram_offload._node_local_pool_coords(4, 0), expected
                    )

    def test_ordinary_dp_replicas_get_unique_pool_ranks(self):
        seen = []
        for replica in range(2):
            with _OffloadHarness(
                parallel=_default_parallel(dp_size=2, dp_rank=replica)
            ):
                seen.extend(
                    engram_offload._node_local_pool_coords(4, rank) for rank in range(4)
                )
        self.assertEqual(seen, [(8, rank) for rank in range(8)])

    def test_inherited_torchrun_rank_does_not_override_scheduler_rank(self):
        with _OffloadHarness(
            node_local_env={"LOCAL_WORLD_SIZE": "1", "LOCAL_RANK": "0"}
        ):
            self.assertEqual(
                [engram_offload._node_local_pool_coords(4, rank) for rank in range(4)],
                [(4, rank) for rank in range(4)],
            )

    def test_partial_node_local_override_is_rejected(self):
        with _OffloadHarness(node_local_env={"SGLANG_NODE_LOCAL_WORLD": "4"}):
            with self.assertRaisesRegex(ValueError, "Set both"):
                engram_offload._node_local_pool_coords(4, 0)

    def test_independent_dp_groups_wait_for_all_pool_owners(self):
        store = Mock()
        store.get.side_effect = lambda key: store.set.call_args_list[0].args[1]
        with (
            _OffloadHarness(
                parallel=_default_parallel(dp_size=2, dp_rank=1),
                pool=EngramPoolConfig(store_url="tcp://127.0.0.1:18500"),
            ),
            patch.object(engram_offload, "_GIB", 1 << 16),
            patch.object(torch.distributed, "TCPStore", return_value=store),
        ):
            manager = _open_manager()
            self.assertEqual((manager.local_world, manager.rank), (8, 4))
            manager.finish_load(0)
            store.wait.assert_called_with([f"loaded/0/{rank}" for rank in range(8)])

    def test_multi_node_tp_derives_local_coordinates(self):
        with _OffloadHarness(parallel=_default_parallel(nnodes=2)):
            for rank in range(4):
                self.assertEqual(
                    engram_offload._node_local_pool_coords(4, rank), (2, rank % 2)
                )

    def test_multi_node_pipeline_requires_explicit_coordinates(self):
        with _OffloadHarness(parallel=_default_parallel(nnodes=2, pp_size=2)):
            with self.assertRaisesRegex(RuntimeError, "node-local"):
                _open_manager()

    def test_same_node_pp_fails_fast(self):
        with _OffloadHarness(parallel=_default_parallel(pp_size=2)):
            with self.assertRaises(AssertionError):
                _open_manager()


class TestFailFastSurface(CustomTestCase):
    def test_missing_memfabric_fails_fast_with_remedy(self):
        with _OffloadHarness(), patch.dict(sys.modules, {"memfabric_hybrid": None}):
            with self.assertRaises(RuntimeError) as cm:
                _open_manager()
            message = str(cm.exception)
            self.assertIn("MEMFABRIC_HYBRID_EXTEND_LIB_PATH", message)
            self.assertIn("--engram-config", message)

    def test_old_wheel_conflicts_with_hicache_memfabric(self):
        with (
            _OffloadHarness(multi_instance=False),
            patch.object(
                engram_offload, "memfabric_host_memory_enabled", return_value=True
            ),
        ):
            with self.assertRaises(RuntimeError) as cm:
                _open_manager()
            self.assertIn("SGLANG_HICACHE_HOST_MEM_BACKEND", str(cm.exception))

    def test_old_wheel_without_hicache_warns_and_uses_default(self):
        with _OffloadHarness(multi_instance=False, node_local_env=_WORLD_ENV) as h:
            with self.assertLogs(
                "sglang.srt.hardware_backend.npu.modules.engram_offload",
                level="WARNING",
            ):
                manager = _open_manager()
            self.assertIs(manager._entity, h.module.offload)
            self.assertFalse(h.module.offload.uninitialized)
            manager.uninitialize()
            self.assertTrue(h.module.offload.uninitialized)

    def test_initialize_failure_carries_error_detail(self):
        with _OffloadHarness(multi_instance=False, node_local_env=_WORLD_ENV) as h:
            h.module.offload.initialize = lambda config: -3
            with self.assertRaises(RuntimeError) as cm:
                _open_manager()
            message = str(cm.exception)
            self.assertIn("-3", message)
            self.assertIn("fake memfabric error detail", message)

    def test_malloc_failure_fails_fast(self):
        with _OffloadHarness(multi_instance=False, node_local_env=_WORLD_ENV) as h:
            h.module.offload.malloc = lambda size, flags=0: 0
            with self.assertRaises(RuntimeError) as cm:
                _open_manager()
            self.assertIn("host DRAM", str(cm.exception))

    def test_registration_failure_fails_fast(self):
        with _OffloadHarness(multi_instance=False, node_local_env=_WORLD_ENV) as h:
            h.module.offload.register_entry_table = lambda entry_bytes, rows: -1
            with self.assertRaises(RuntimeError) as cm:
                _open_manager()
            self.assertIn("register_entry_table failed", str(cm.exception))

    def test_oversized_row_fails_fast(self):
        layout = EngramLayout(
            max_ngram_size=2,
            layer_ids=(0,),
            num_embeddings=(10,),
            primes=(((13, 17),),),
            n_heads=1,
            head_dim=engram_offload._MAX_ENTRY_BYTES + 1,
        )
        with _OffloadHarness(
            node_local_env={
                "SGLANG_NODE_LOCAL_WORLD": "1",
                "SGLANG_NODE_LOCAL_RANK": "0",
            }
        ):
            with self.assertRaises(AssertionError) as cm:
                _open_manager(layout)
            self.assertIn("entry_gather limit", str(cm.exception))


class TestBackendLifecycle(CustomTestCase):
    def test_a3_loads_packed_codes_and_scales_directly_into_pool(self):
        rank_env = {"SGLANG_NODE_LOCAL_WORLD": "4", "SGLANG_NODE_LOCAL_RANK": "3"}
        with (
            _OffloadHarness(node_local_env=rank_env),
            patch.object(engram_offload, "is_ascend_910c", return_value=True),
            patch.object(
                _FakeOffloadEntity,
                "get_mapped_pool",
                lambda entity: (ctypes.addressof(entity._buf), entity._slot),
                create=True,
            ),
        ):
            backend = engram_offload.NpuEngramOffloadBackend(
                _LAYOUT, _HEAD_DIM, 9, {"format": "int8_sym"}
            )
            holder = nn.Module()
            backend.install_params(holder)
            weight = (
                torch.arange(701 * _HEAD_DIM)
                .reshape(701, _HEAD_DIM)
                .to(torch.uint8)
                .view(torch.int8)
            )
            scales = (
                torch.arange(701 * (_HEAD_DIM // FP8_BLOCK_SIZE))
                .reshape(701, -1)
                .to(torch.uint8)
            )
            holder.weight.weight_loader(holder.weight, weight)
            holder.scale.weight_loader(holder.scale, scales)
            packed = backend._table.pool_weight
            self.assertTrue(
                torch.equal(packed[:173, :_HEAD_DIM], weight[528:].view(torch.uint8))
            )
            self.assertTrue(torch.equal(packed[:173, _HEAD_DIM:], scales[528:]))
            self.assertEqual(packed[173:].count_nonzero().item(), 0)
            backend.finish_load("a3")
            self.assertIsNone(backend._buffers)

    def test_a3_uses_mapped_lookup_without_staging_buffers(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            with patch.object(engram_offload, "is_ascend_910c", return_value=True):
                manager = _open_manager()
                manager._entity.get_mapped_pool = lambda: (0x280000000000, manager.slot)
                backend = _make_backend(nn.Module())
            backend.finish_load("a3")
            self.assertIsNone(backend._buffers)
            self.assertEqual(
                backend._table.entry_bytes, _HEAD_DIM + _HEAD_DIM // FP8_BLOCK_SIZE
            )
            self.assertEqual(backend._scale.device.type, "meta")
            expected = [0x280000000000 + rank * manager.slot for rank in range(4)]
            self.assertEqual(backend._mapped_bases.tolist(), expected)
            ids = torch.zeros((2, _N_HASH_COLS), dtype=torch.int64)
            result = torch.empty((2, _N_HASH_COLS, _HEAD_DIM), dtype=torch.bfloat16)
            with patch(
                "sglang.kernels.ops.embeddings.engram_mapped_gather.mapped_engram_gather",
                return_value=result,
            ) as gather:
                self.assertIs(backend.gather(ids), result)
            gather.assert_called_once_with(
                backend._mapped_bases,
                ids,
                backend.rows,
                _NUM_EMBEDDINGS,
                _HEAD_DIM,
                FP8_BLOCK_SIZE,
                True,
            )
            self.assertIsNone(backend._buffers)

    def test_capture_buffers_cover_prefill_and_speculative_decode(self):
        for width, tokens in [(1, 512), (5, 256 * 5)]:
            with (
                self.subTest(decode_width=width),
                _OffloadHarness(
                    node_local_env=_WORLD_ENV,
                    chunked_prefill_size=512,
                    decode_max_bs=256,
                    decode_width=width,
                ),
            ):
                backend = _make_backend(nn.Module())
                backend.finish_load("capture")
                buffers = backend._buffers
                self.assertEqual(buffers.capacity, tokens * _N_HASH_COLS)
                backend.finish_load("capture")
                self.assertIs(backend._buffers, buffers)
                with patch.object(
                    engram_offload, "is_device_stream_capturing", return_value=True
                ):
                    self.assertIs(
                        backend._ensure_buffers(
                            tokens * _N_HASH_COLS, torch.device("cpu")
                        ),
                        buffers,
                    )

    def test_online_updates_rejected_without_pool_mutation(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            holder = nn.Module()
            backend = _make_backend(holder)
            weight = torch.zeros((_NUM_EMBEDDINGS, _HEAD_DIM), dtype=torch.uint8).view(
                torch.float8_e4m3fn
            )
            scales = torch.full(
                (_NUM_EMBEDDINGS, _HEAD_DIM // FP8_BLOCK_SIZE), 127, dtype=torch.uint8
            )
            holder.weight.weight_loader(holder.weight, weight)
            holder.scale.weight_loader(holder.scale, scales)
            backend.finish_load("initial")
            before = backend._table.pool_weight.clone()
            for param, value in [(holder.weight, weight), (holder.scale, scales)]:
                with self.subTest(param=param.shape):
                    with self.assertRaisesRegex(RuntimeError, "online weight updates"):
                        param.weight_loader(param, value)
            self.assertTrue(torch.equal(before, backend._table.pool_weight))
            self.assertIn("Online weight updates", holder._derived_weight_cache_error)

    def test_malformed_checkpoint_shape_is_rejected(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            holder = nn.Module()
            _make_backend(holder)
            with self.assertRaisesRegex(ValueError, "does not match"):
                holder.weight.weight_loader(
                    holder.weight,
                    torch.zeros(
                        (_NUM_EMBEDDINGS - 1, _HEAD_DIM), dtype=torch.uint8
                    ).view(torch.float8_e4m3fn),
                )
            with self.assertRaisesRegex(ValueError, "scales"):
                holder.scale.weight_loader(
                    holder.scale,
                    torch.zeros(
                        (_NUM_EMBEDDINGS - 1, _HEAD_DIM // FP8_BLOCK_SIZE),
                        dtype=torch.uint8,
                    ),
                )

    def test_weight_loader_writes_pool_slice_and_zeroes_tail(self):
        # table 1: E=701 over 4 slots -> 176 rows/slot; rank 3 owns rows
        # [528, 701) = 173 rows; the last three pool rows must be zeroed.
        rank_env = {"SGLANG_NODE_LOCAL_WORLD": "4", "SGLANG_NODE_LOCAL_RANK": "3"}
        with _OffloadHarness(node_local_env=rank_env):
            holder = nn.Module()
            backend = _make_backend(holder, layer_id=9)
            self.assertEqual((backend.rows, backend.row_start), (176, 528))
            self.assertEqual((holder.rows, holder.row_start), (176, 528))
            self.assertEqual(holder.weight.shape, (176, _HEAD_DIM))
            self.assertEqual(holder.weight.device.type, "meta")
            self.assertEqual(holder.scale.shape, (701, _HEAD_DIM // FP8_BLOCK_SIZE))
            full = np.array(
                [(r + j) % 251 for r in range(701) for j in range(_HEAD_DIM)],
                dtype=np.uint8,
            ).reshape(701, _HEAD_DIM)
            holder.weight.weight_loader(
                holder.weight, torch.from_numpy(full).view(torch.float8_e4m3fn)
            )
            staged = backend._table.pool_weight.numpy()
            np.testing.assert_array_equal(staged[:173], full[528:701])
            self.assertEqual(int(staged[173:].sum()), 0)

    def test_gather_before_finish_raises(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            holder = nn.Module()
            backend = _make_backend(holder)
            with self.assertRaises(AssertionError):
                backend.gather(torch.zeros(2, _N_HASH_COLS, dtype=torch.int64))


class TestGatherEquivalence(CustomTestCase):
    def test_int8_codes_use_modelslim_format_without_fp8_conversion(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            storage = {"format": "int8_sym"}
            backend = engram_offload.NpuEngramOffloadBackend(
                _LAYOUT, _HEAD_DIM, 5, storage
            )
            holder = nn.Module()
            backend.install_params(holder)
            weight = torch.randint(
                -128, 128, (_NUM_EMBEDDINGS, _HEAD_DIM), dtype=torch.int8
            )
            scale = torch.full(
                (_NUM_EMBEDDINGS, _HEAD_DIM // FP8_BLOCK_SIZE), 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            _write_all_chunks(
                backend._manager._entity,
                backend._manager,
                backend._table,
                weight.view(torch.uint8).numpy(),
            )
            backend._load_weight_rows(holder.weight, weight)
            backend._load_scale_rows(holder.scale, scale)
            backend.finish_load("int8")
            ids = torch.tensor(
                [[0, 249, 250, 999], [500, 501, 749, 750]], dtype=torch.int64
            )
            self.assertEqual(holder.weight.dtype, torch.int8)
            torch.testing.assert_close(
                backend.gather(ids), weight[ids].to(torch.bfloat16), rtol=0, atol=0
            )

    def test_gather_matches_device_path_bitwise(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            holder = nn.Module()
            backend = _make_backend(holder)
            gen = torch.Generator().manual_seed(7)
            # e4m3fn has exactly two NaN encodings (0x7F / 0xFF); keep them out
            # so torch.equal's element-wise == stays meaningful
            full_u8 = torch.randint(
                0, 256, (_NUM_EMBEDDINGS, _HEAD_DIM), dtype=torch.uint8, generator=gen
            )
            full_u8[full_u8 == 0x7F] = 0
            full_u8[full_u8 == 0xFF] = 0
            full = full_u8.view(torch.float8_e4m3fn)
            full_scale = torch.randint(
                0,
                256,
                (_NUM_EMBEDDINGS, _HEAD_DIM // FP8_BLOCK_SIZE),
                dtype=torch.uint8,
                generator=gen,
            ).view(torch.float8_e8m0fnu)
            holder.weight.weight_loader(holder.weight, full)
            holder.scale.weight_loader(holder.scale, full_scale)
            backend.finish_load("layer 5")
            # fill every rank's chunk so the gather can serve arbitrary ids
            _write_all_chunks(
                backend._manager._entity,
                backend._manager,
                backend._table,
                full.view(torch.uint8).numpy(),
            )

            ids = torch.randint(0, _NUM_EMBEDDINGS, (17, _N_HASH_COLS), generator=gen)
            out = backend.gather(ids)
            self.assertEqual(out.shape, (17, _N_HASH_COLS, _HEAD_DIM))
            self.assertEqual(out.dtype, torch.bfloat16)

            # same bytes through the device-resident path's torch branch
            with (
                patch.object(
                    engram_mod, "resolve_engram_config", return_value=EngramConfig()
                ),
                patch.object(engram_mod, "_needs_fp8_byte_view", return_value=True),
                patch.object(
                    engram_mod,
                    "get_parallel",
                    return_value=SimpleNamespace(tp_size=1, tp_rank=0),
                ),
            ):
                resident = EngramEmbedding(
                    _NUM_EMBEDDINGS, _HEAD_DIM, 5, column_bounds=(0, _NUM_EMBEDDINGS)
                )
                resident.weight.data.copy_(full)
                resident.scale.data.copy_(full_scale)
                device_path = resident._owned_rows(ids)
            self.assertTrue(torch.equal(out, device_path))

    def test_capture_oversize_fails_fast(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            holder = nn.Module()
            backend = _make_backend(holder)
            backend.finish_load("layer 5")
            self.assertEqual(backend._buffers.capacity, 512 * _N_HASH_COLS)
            ids = torch.zeros((512 * _N_HASH_COLS + 1, 1), dtype=torch.int64)
            with (
                patch.object(
                    engram_offload, "is_device_stream_capturing", return_value=True
                ),
                self.assertRaises(RuntimeError) as cm,
            ):
                backend.gather(ids)
            self.assertIn("undersized", str(cm.exception))

    def test_eager_oversize_uses_throwaway_and_keeps_frozen(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            holder = nn.Module()
            backend = _make_backend(holder)
            backend.finish_load("layer 5")
            ids = torch.zeros(600 * _N_HASH_COLS, dtype=torch.int64).reshape(
                600, _N_HASH_COLS
            )
            out = backend.gather(ids)
            self.assertEqual(out.shape, (600, _N_HASH_COLS, _HEAD_DIM))
            # the frozen buffer is intact for graph replays
            self.assertEqual(backend._buffers.capacity, 512 * _N_HASH_COLS)

    def test_empty_batch_skips_gather_and_sizing(self):
        with _OffloadHarness(
            node_local_env=_WORLD_ENV, chunked_prefill_size=0, decode_max_bs=0
        ):
            holder = nn.Module()
            backend = _make_backend(holder)
            backend.finish_load("layer 5")
            self.assertIsNone(backend._buffers)
            out = backend.gather(torch.empty(0, _N_HASH_COLS, dtype=torch.int64))
            self.assertEqual(out.shape, (0, _N_HASH_COLS, _HEAD_DIM))
            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertIsNone(backend._buffers)


def _patch_engram_config(config: EngramConfig):
    """Select the layer configuration without argument resolution."""
    return patch.object(engram_mod, "resolve_engram_config", return_value=config)


@contextlib.contextmanager
def _routing_patches(config, is_arch35: bool):
    """Patch NPU detection while constructing CPU modules."""
    with contextlib.ExitStack() as stack:
        for cm in (
            _patch_engram_config(config),
            patch.object(engram_mod, "is_npu", return_value=True),
            patch.object(
                engram_mod.torch,
                "npu",
                SimpleNamespace(get_device_name=lambda: "Ascend910B"),
                create=True,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                return_value=is_arch35,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_ascend_910c",
                return_value=False,
            ),
            patch.object(
                engram_mod,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=2, tp_rank=1),
            ),
        ):
            stack.enter_context(cm)
        yield


class TestPlacementRouting(CustomTestCase):
    def test_mxfp4_offload_rejected_before_pool_allocation(self):
        config = EngramConfig(placement=EngramPlacement.CPU)
        with (
            _OffloadHarness(node_local_env=_WORLD_ENV),
            _routing_patches(config, is_arch35=True),
        ):
            with self.assertRaisesRegex(ValueError, "MXFP4"):
                EngramEmbedding(
                    _NUM_EMBEDDINGS,
                    _HEAD_DIM,
                    5,
                    column_bounds=(0, _NUM_EMBEDDINGS),
                    mxfp4=True,
                    layout=_LAYOUT,
                )
            self.assertIsNone(engram_offload._MANAGER)

    def test_a3_int8_host_placement_uses_mapped_pool(self):
        config = EngramConfig(
            placement=EngramPlacement.CPU, replication=EngramReplication.PER_NODE
        )
        storage = {
            "format": "int8_sym",
            "weight_dtype": "I8",
            "layout": "row_major",
            "group_size": FP8_BLOCK_SIZE,
            "axis": 1,
            "scale_suffix": "scale",
            "scale_dtype": "F8_E8M0",
            "weight_shape": [_NUM_EMBEDDINGS, _HEAD_DIM],
            "scale_shape": [_NUM_EMBEDDINGS, _HEAD_DIM // FP8_BLOCK_SIZE],
        }
        with (
            _OffloadHarness(node_local_env=_WORLD_ENV),
            _routing_patches(config, is_arch35=False),
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_ascend_910c",
                return_value=True,
            ),
            patch.object(engram_offload, "is_ascend_910c", return_value=True),
            patch.object(
                _FakeOffloadEntity,
                "get_mapped_pool",
                lambda entity: (ctypes.addressof(entity._buf), entity._slot),
                create=True,
            ),
        ):
            embed = EngramEmbedding(
                _NUM_EMBEDDINGS,
                _HEAD_DIM,
                5,
                column_bounds=(0, _NUM_EMBEDDINGS),
                layout=_LAYOUT,
                storage=storage,
            )
            self.assertEqual(embed.weight.dtype, torch.int8)
            self.assertEqual(embed.weight.device.type, "meta")
            self.assertEqual(embed.scale.device.type, "meta")
            self.assertIsNotNone(embed._offload_backend._mapped_bases)
        # Keep the existing A5 INT8 host-placement rejection; its FP8 staged
        # entry_gather path is covered separately by the equivalence tests.
        with (
            _OffloadHarness(node_local_env=_WORLD_ENV),
            _routing_patches(config, is_arch35=True),
            self.assertRaisesRegex(ValueError, "device-local table shards"),
        ):
            EngramEmbedding(
                _NUM_EMBEDDINGS,
                _HEAD_DIM,
                5,
                column_bounds=(0, _NUM_EMBEDDINGS),
                layout=_LAYOUT,
                storage=storage,
            )

    def test_multinode_npu_replication_config_is_accepted(self):
        from sglang.srt.arg_groups import deepseek_v4_hook as hook

        cfg = SimpleNamespace(
            engram_config={"placement": "cpu", "replication": "per_node"},
            nnodes=2,
        )
        with (
            patch.object(hook, "resolving_view", return_value=cfg),
            patch.object(
                hook,
                "model_config_of",
                return_value=SimpleNamespace(
                    hf_config=SimpleNamespace(model_type="deepseek_v41")
                ),
            ),
            patch.object(hook, "is_npu", return_value=True),
            patch.object(hook, "declare_resolution") as declared,
        ):
            hook._resolve_engram_config(object())
        self.assertEqual(
            declared.call_args.kwargs["engram_config"].replication, "per_node"
        )
        self.assertEqual(
            declared.call_args.kwargs["engram_config"].pool, EngramPoolConfig()
        )
        with (
            patch.object(hook, "resolving_view", return_value=cfg),
            patch.object(
                hook,
                "model_config_of",
                return_value=SimpleNamespace(
                    hf_config=SimpleNamespace(model_type="deepseek_v41")
                ),
            ),
            patch.object(hook, "is_npu", return_value=False),
            self.assertRaisesRegex(ValueError, "TP group on one node"),
        ):
            hook._resolve_engram_config(object())

    def test_gpu_placement_keeps_device_tables(self):
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            with _routing_patches(EngramConfig(), is_arch35=True):
                embed = EngramEmbedding(
                    _NUM_EMBEDDINGS,
                    _HEAD_DIM,
                    5,
                    column_bounds=(0, _NUM_EMBEDDINGS),
                )
                self.assertIsNone(embed._offload_backend)
                self.assertIsNone(embed.host_table)

    def test_cpu_placement_engages_offload_backend(self):
        for replication in (EngramReplication.NONE, EngramReplication.PER_NODE):
            config = EngramConfig(
                placement=EngramPlacement.CPU, replication=replication
            )
            with (
                self.subTest(replication=replication),
                _OffloadHarness(node_local_env=_WORLD_ENV),
                _routing_patches(config, is_arch35=True),
            ):
                embed = EngramEmbedding(
                    _NUM_EMBEDDINGS,
                    _HEAD_DIM,
                    5,
                    column_bounds=(0, _NUM_EMBEDDINGS),
                    layout=_LAYOUT,
                )
                self.assertIsNotNone(embed._offload_backend)
                self.assertIsNone(embed.host_table)
                self.assertEqual(embed.weight.device.type, "meta")
                self.assertEqual((embed.rows, embed.row_start), (250, 250))

    def test_pre_arch35_degrades_to_device_tables(self):
        config = EngramConfig(placement=EngramPlacement.CPU)
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            with _routing_patches(config, is_arch35=False):
                with self.assertLogs("sglang.srt.layers.engram", level="WARNING"):
                    embed = EngramEmbedding(
                        _NUM_EMBEDDINGS,
                        _HEAD_DIM,
                        5,
                        column_bounds=(0, _NUM_EMBEDDINGS),
                    )
                self.assertIsNone(embed._offload_backend)
                self.assertIsNone(embed.host_table)

    def test_missing_layout_fails_fast(self):
        config = EngramConfig(
            placement=EngramPlacement.CPU, replication=EngramReplication.PER_NODE
        )
        with _OffloadHarness(node_local_env=_WORLD_ENV):
            with _routing_patches(config, is_arch35=True):
                with self.assertRaises(AssertionError) as cm:
                    EngramEmbedding(
                        _NUM_EMBEDDINGS,
                        _HEAD_DIM,
                        5,
                        column_bounds=(0, _NUM_EMBEDDINGS),
                    )
                self.assertIn("EngramLayout", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
