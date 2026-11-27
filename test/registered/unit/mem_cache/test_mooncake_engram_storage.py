"""CPU contracts for the Mooncake Engram adapter; no Mooncake install required."""

import json
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.srt import runtime_context
from sglang.srt.mem_cache.engram import _parse_config, create_storage
from sglang.srt.mem_cache.engram.remote_utils import make_remote_config
from sglang.srt.mem_cache.storage.mooncake_engram import MooncakeEngramStorage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _CpuEngramStore:
    """Replace only the external byte-table API, including its -1 sentinel."""

    def __init__(self, *, tables, namespace, config):
        self.tables = tables
        self.rows = None
        self.close_count = 0
        self.lookup_count = 0

    def populate(self, table_id, rows):
        spec = self.tables[table_id]
        if rows.dtype != np.uint8 or rows.shape != (
            spec["num_rows"],
            spec["row_bytes"],
        ):
            raise ValueError("table metadata does not match the published rows")
        if not rows.flags.c_contiguous:
            raise ValueError("populate requires contiguous rows")
        self.rows = rows.copy()

    def lookup(self, table_ids, indices, output):
        if table_ids != ["rows"] or indices.shape[1:] != (1,):
            raise ValueError("expected one table column")
        if not indices.flags.c_contiguous:
            raise ValueError("lookup requires contiguous IDs")
        self.lookup_count += 1
        valid = indices[:, 0] >= 0
        output[valid, 0] = self.rows[indices[valid, 0]]

    def close(self):
        self.close_count += 1


class TestEngramStorageConfig(CustomTestCase):
    def test_local_backend_defaults_and_explicit_directory(self):
        for config, directory in (
            (None, "/dev/shm/sglang-engram"),
            ('{"backend":"local","local_dir":"/tmp/engram-test"}', "/tmp/engram-test"),
        ):
            with self.subTest(config=config):
                storage = create_storage(config)
                self.addCleanup(storage.close)
                self.assertIsInstance(storage, MooncakeEngramStorage)
                self.assertEqual(storage._local_dir, directory)

    def test_invalid_config_does_not_silently_fall_back_to_local(self):
        for config, message in (
            ("{", "JSON object"),
            ("[]", "JSON object"),
            ('{"local_dir":""}', "nonempty string"),
            ('{"local_dir":42}', "nonempty string"),
            ('{"unknown_option":true}', "Unknown.*options"),
            ('{"backend":"typo"}', "Unknown Engram storage backend"),
            ('{"backend":"remote"}', "Unknown Engram storage backend"),
        ):
            with (
                self.subTest(config=config),
                self.assertRaisesRegex(ValueError, message),
            ):
                create_storage(config)


class TestMooncakeEngramStorage(CustomTestCase):
    def setUp(self):
        self.stores = []
        module = ModuleType("mooncake.engram")
        module.EngramStore = self._new_store
        module.EngramStoreConfig = SimpleNamespace
        self.enterContext(
            mock.patch.dict(
                sys.modules,
                {"mooncake": ModuleType("mooncake"), "mooncake.engram": module},
            )
        )
        # These tests must not initialize a CUDA runtime, even on GPU hosts.
        self.enterContext(mock.patch("torch.cuda.is_initialized", return_value=False))
        self.storage = MooncakeEngramStorage(local_dir="/unused/engram-test")
        self.addCleanup(self.storage.close)

    def _new_store(self, **kwargs):
        store = _CpuEngramStore(**kwargs)
        self.stores.append(store)
        return store

    def test_publish_copies_noncontiguous_rows_and_lookup_preserves_shape(self):
        rows = torch.arange(48, dtype=torch.uint8).reshape(6, 8)[:, ::2]
        expected_rows = rows.clone()
        self.storage.initialize(rows)
        rows.zero_()  # Storage must own the data after the loader releases it.
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                ids = torch.tensor([[5, -1, 2], [0, 4, 1]], dtype=dtype).t()
                actual = self.storage.lookup(ids)
                expected = torch.zeros((*ids.shape, 4), dtype=torch.uint8)
                expected[ids >= 0] = expected_rows[ids[ids >= 0].long()]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_empty_batch_never_calls_backend(self):
        self.storage.initialize(torch.ones((6, 4), dtype=torch.uint8))
        actual = self.storage.lookup(torch.empty((0, 3), dtype=torch.int64))
        self.assertEqual(actual.shape, (0, 3, 4))
        self.assertEqual(actual.dtype, torch.uint8)
        self.assertEqual(self.stores[0].lookup_count, 0)

    def test_rejected_initialization_can_be_retried(self):
        for rows in (
            torch.ones((2, 4), dtype=torch.float32),
            torch.ones(8, dtype=torch.uint8),
            torch.empty((2, 4), dtype=torch.uint8, device="meta"),
        ):
            with self.subTest(shape=rows.shape, dtype=rows.dtype, device=rows.device):
                with self.assertRaisesRegex(ValueError, "CPU byte matrix"):
                    self.storage.initialize(rows)
        self.assertEqual(self.stores, [])
        rows = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
        self.storage.initialize(rows)
        torch.testing.assert_close(self.storage.lookup(torch.tensor([1])), rows[1:])

    def test_publication_failure_closes_store_and_allows_retry(self):
        rows = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
        with mock.patch.object(
            _CpuEngramStore, "populate", side_effect=OSError("publish failed")
        ):
            with self.assertRaisesRegex(OSError, "publish failed"):
                self.storage.initialize(rows)
        self.assertEqual(self.stores[0].close_count, 1)
        with self.assertRaisesRegex(RuntimeError, "Initialize"):
            self.storage.lookup(torch.tensor([0]))
        self.storage.initialize(rows)
        torch.testing.assert_close(self.storage.lookup(torch.tensor([0])), rows[:1])

    def test_duplicate_initialization_preserves_original_table(self):
        rows = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
        self.storage.initialize(rows)
        with self.assertRaisesRegex(RuntimeError, "already initialized"):
            self.storage.initialize(torch.zeros_like(rows))
        torch.testing.assert_close(self.storage.lookup(torch.tensor([1])), rows[1:])
        self.assertEqual(len(self.stores), 1)

    def test_graph_capture_rejects_cold_publication(self):
        with (
            mock.patch("torch.cuda.is_initialized", return_value=True),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
            self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"),
        ):
            self.storage.initialize(torch.ones((2, 4), dtype=torch.uint8))
        self.assertEqual(self.stores, [])

    def test_lookup_lifecycle_and_invalid_id_dtype(self):
        with self.assertRaisesRegex(RuntimeError, "Initialize"):
            self.storage.lookup(torch.tensor([0]))
        self.storage.initialize(torch.ones((2, 4), dtype=torch.uint8))
        with self.assertRaisesRegex(TypeError, "int32 or int64"):
            self.storage.lookup(torch.tensor([0.0]))
        self.assertEqual(self.stores[0].lookup_count, 0)
        self.storage.close()
        self.storage.close()
        self.assertEqual(self.stores[0].close_count, 1)
        with self.assertRaisesRegex(RuntimeError, "Initialize"):
            self.storage.lookup(torch.tensor([0]))


class TestEngramRemoteConfig(CustomTestCase):
    def setUp(self):
        self.options = dict(backend="deepep_rdma", assignment="auto", group_size=2)

    def test_runtime_owned_fields_cannot_be_supplied_in_json(self):
        for name, value in (
            ("caller_id", 0),
            ("num_callers", 4),
            ("max_lookup_rows", 4096),
            ("node_id", "node"),
            ("model_key", "weights"),
            ("master_addr", "host:50051"),
            ("session_id", "old"),
            ("local_dir", "/tmp"),
            ("rdma_device", "mlx5_0"),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unknown"):
                _parse_config(json.dumps(dict(self.options, **{name: value})))

    def test_auto_manual_and_invalid_identity(self):
        self.assertEqual(_parse_config(json.dumps(self.options)), self.options)
        for extra in ({"group_id": "g"}, {"shard_id": 0}, {"group_size": True}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                _parse_config(json.dumps(dict(self.options, **extra)))
        manual = dict(self.options, assignment="manual", group_id="g", shard_id=1)
        self.assertEqual(_parse_config(json.dumps(manual)), manual)
        with self.assertRaises(ValueError):
            _parse_config(json.dumps(dict(manual, shard_id=2)))

    def test_callers_identity_and_capacity_are_derived_from_runtime(self):
        module = ModuleType("mooncake.engram")
        module.EngramStoreConfig = lambda **kwargs: SimpleNamespace(**kwargs)
        parallel = SimpleNamespace(
            pp_size=1,
            dp_size=1,
            launch_world_size=8,
            nnodes=2,
            tp_rank=5,
            attn_dp_size=1,
        )
        graph = SimpleNamespace(
            decode=SimpleNamespace(max_bs=16), prefill=SimpleNamespace(max_bs=2048)
        )
        patches = dict(
            get_parallel=parallel,
            get_schedule=SimpleNamespace(
                chunked_prefill_size=1024,
                max_prefill_tokens=16384,
                max_running_requests=8,
                enable_mixed_chunk=False,
            ),
            get_model=SimpleNamespace(
                model_path="checkpoint", revision="r1", context_length=4096
            ),
            get_device=SimpleNamespace(base_gpu_id=0, gpu_id_step=1),
            get_exec=SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=graph)),
            get_spec=SimpleNamespace(speculative_num_draft_tokens=5),
            process_model_config=SimpleNamespace(context_len=131072),
        )
        with (
            mock.patch.dict(sys.modules, {"mooncake.engram": module}),
            mock.patch.dict(os.environ, {"NCCL_CUMEM_ENABLE": "1"}),
        ):
            for name, value in patches.items():
                self.enterContext(
                    mock.patch.object(runtime_context, name, return_value=value)
                )
            options = dict(self.options)
            options.pop("backend")
            first = make_remote_config(options, 24)
            self.assertEqual((first.caller_id, first.num_callers), (1, 4))
            self.assertEqual(first.max_lookup_rows, 2048 * 24)
            parallel.tp_rank = 6
            second = make_remote_config(options, 24)
            self.assertEqual(second.caller_id, 2)
            self.assertEqual(first.node_id, second.node_id)
            with mock.patch.dict(
                os.environ, {"NCCL_CUMEM_ENABLE": "0", "NCCL_IB_HCA": "=existing"}
            ):
                with self.assertRaisesRegex(ValueError, "CUMEM"):
                    make_remote_config(options, 24)
                self.assertEqual(os.environ["NCCL_CUMEM_ENABLE"], "0")
                self.assertEqual(os.environ["NCCL_IB_HCA"], "=existing")
            with mock.patch.dict(os.environ, {"NCCL_IB_HCA": "=existing"}):
                make_remote_config(options, 24)
                self.assertEqual(os.environ["NCCL_IB_HCA"], "=existing")
            patches["get_schedule"].chunked_prefill_size = -1
            self.assertEqual(
                make_remote_config(options, 24).max_lookup_rows, 131072 * 24
            )
            patches["get_schedule"].chunked_prefill_size = 2048
            patches["get_schedule"].enable_mixed_chunk = True
            self.assertEqual(
                make_remote_config(options, 24).max_lookup_rows, (2048 + 16 * 5) * 24
            )
            parallel.dp_size = 2
            with self.assertRaisesRegex(ValueError, "one model replica"):
                make_remote_config(options, 24)

    def test_head_conversion_supplies_only_node_owned_heads(self):
        stores = []

        class Store:
            shard_id, group_size, group_id = 1, 2, "g"

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                stores.append(self)

            def populate_and_sync(self, local_tables, *, table_owners):
                self.local, self.owners = local_tables, table_owners

            def close(self):
                pass

        module = ModuleType("mooncake.engram")
        module.EngramStore = Store
        with (
            mock.patch.dict(sys.modules, {"mooncake.engram": module}),
            mock.patch(
                "sglang.srt.mem_cache.engram.remote_utils.make_remote_config",
                return_value=SimpleNamespace(group_size=2),
            ),
        ):
            storage = MooncakeEngramStorage(
                remote=self.options,
                head_bounds=[0, 4, 10, 15, 21],
                namespace="layer0",
                row_bytes=8,
            )
            self.assertEqual((storage.row_start, storage.row_end), (10, 21))
            rows = torch.arange(88, dtype=torch.uint8).reshape(11, 8)
            storage.initialize(rows)
            self.assertEqual(set(stores[0].local), {"head2", "head3"})
            torch.testing.assert_close(stores[0].local["head2"], rows[:5])
            torch.testing.assert_close(stores[0].local["head3"], rows[5:])
            self.assertEqual(
                stores[0].owners, {"head0": 0, "head1": 0, "head2": 1, "head3": 1}
            )
            storage.close()


if __name__ == "__main__":
    unittest.main()
