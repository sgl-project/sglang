"""Unit tests for srt/distributed/device_communicators/tent_transfer_engine.py"""

import json
import os
import sys
import types
import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_fake_tent(
    fail_batch_index=None,
    raise_on_submit_source=None,
    pending_batch_index=None,
    peer_alive=True,
):
    """Minimal stand-in for the `tent` pybind module that records engine calls."""
    mod = types.ModuleType("tent")

    class TentException(Exception):
        pass

    class InternalError(TentException):
        pass

    class OpCode:
        READ, WRITE = "READ", "WRITE"

    class TransferStatusEnum:
        PENDING, COMPLETED, FAILED = "PENDING", "COMPLETED", "FAILED"
        TIMEOUT, CANCELED, INVALID = "TIMEOUT", "CANCELED", "INVALID"

    class TransportType:
        UNSPEC = "UNSPEC"

    class IntentType:
        INTENT_UNSPEC = "INTENT_UNSPEC"
        FOREGROUND_GET = "FOREGROUND_GET"
        BACKGROUND_PREFETCH = "BACKGROUND_PREFETCH"

    class Request:
        # Mirrors the pybind signature: tail is (priority, transport_hint,
        # policy_name, deadline_ns, intent_type), all optional.
        def __init__(self, opcode, source, target_id, target_offset, length, *tail):
            assert len(tail) in (0, 5), tail
            self.args = (opcode, source, target_id, target_offset, length)
            self.priority = tail[0] if tail else "PRIO_HIGH"
            self.intent_type = tail[4] if tail else IntentType.INTENT_UNSPEC

    class Status:
        def __init__(self, state):
            self.state = state

    class TransferEngine:
        def __init__(self, config_path):
            with open(config_path) as f:
                self.config = json.load(f)
            self.submitted = []
            self.freed = []
            self.allocated = []
            self.qos = []
            self.opened = []
            self.closed = []
            self.unregistered = []
            self.canceled = []
            self.segment_info = []
            self.calls = []
            self._next_batch = 100

        def available(self):
            return True

        def get_rpc_server_port(self):
            return 12345

        def open_segment(self, name):
            self.opened.append(name)
            self.calls.append(("open", name))
            return len(self.opened)

        def close_segment(self, handle):
            self.closed.append(handle)
            self.calls.append(("close", handle))

        def get_segment_info(self, handle):
            self.segment_info.append(handle)
            self.calls.append(("info", handle))
            if not peer_alive:
                raise InternalError("peer RPC server unreachable")

        def register_local_memory_batch(self, ptrs, lengths):
            pass

        def unregister_local_memory_batch(self, ptrs, lengths):
            self.unregistered.append((list(ptrs), list(lengths)))

        def allocate_transfer_batch(self, size):
            self._next_batch += 1
            self.allocated.append(self._next_batch)
            return self._next_batch

        def submit_transfer(self, batch_id, requests):
            if any(r.args[1] == raise_on_submit_source for r in requests):
                raise InternalError("injected submit failure")
            self.submitted.append((batch_id, [r.args for r in requests]))
            self.qos.extend((r.priority, r.intent_type) for r in requests)

        def get_transfer_status_overall(self, batch_id):
            index = [b for b, _ in self.submitted].index(batch_id)
            if index == fail_batch_index:
                return Status(TransferStatusEnum.FAILED)
            if index == pending_batch_index:
                return Status(TransferStatusEnum.PENDING)
            return Status(TransferStatusEnum.COMPLETED)

        def cancel_transfer(self, batch_id, task_id):
            self.canceled.append((batch_id, task_id))

        def free_transfer_batch(self, batch_id):
            self.freed.append(batch_id)

    for name, value in dict(
        TentException=TentException,
        InternalError=InternalError,
        OpCode=OpCode,
        TransferStatusEnum=TransferStatusEnum,
        Request=Request,
        TransferEngine=TransferEngine,
        TransportType=TransportType,
        IntentType=IntentType,
        PRIO_HIGH="PRIO_HIGH",
        PRIO_LOW="PRIO_LOW",
    ).items():
        setattr(mod, name, value)
    return mod


class TestTentTransferEngine(CustomTestCase):
    def _engine(self, ib_device="mlx5_0", gpu_id=None, **fake_kwargs):
        from sglang.srt.distributed.device_communicators import tent_transfer_engine

        fake = _make_fake_tent(**fake_kwargs)
        with patch.dict(sys.modules, {"tent": fake}):
            engine = tent_transfer_engine.TentTransferEngine(
                hostname="10.0.0.1", ib_device=ib_device, gpu_id=gpu_id
            )
        return engine, tent_transfer_engine._SUBMIT_CHUNK

    def _canceled_batches(self, engine):
        return sorted({batch_id for batch_id, _ in engine.engine.canceled})

    def test_large_write_is_chunked_and_covers_every_request_once(self):
        engine, chunk = self._engine()
        n = chunk * 2 + 3
        srcs = list(range(1000, 1000 + n))
        dsts = list(range(5000, 5000 + n))
        lens = [64] * n

        rc = engine.batch_transfer_sync("10.0.0.2:7000", srcs, dsts, lens)

        self.assertEqual(rc, 0)
        submitted = engine.engine.submitted
        self.assertEqual(sorted(len(reqs) for _, reqs in submitted), [3, chunk, chunk])
        flat = sorted(
            (req for _, reqs in submitted for req in reqs), key=lambda r: r[1]
        )
        handle = engine.engine.opened.index("10.0.0.2:7000") + 1
        self.assertEqual(
            flat,
            [("WRITE", s, handle, d, l) for s, d, l in zip(srcs, dsts, lens)],
        )
        self.assertEqual(sorted(engine.engine.freed), sorted(b for b, _ in submitted))

    def test_failed_chunk_returns_error_frees_all_and_drops_peer_handle(self):
        engine, chunk = self._engine(fail_batch_index=1)
        n = chunk * 3
        rc = engine.batch_transfer_sync("10.0.0.2:7000", [1] * n, [2] * n, [8] * n)

        self.assertLess(rc, 0)
        self.assertEqual(
            sorted(engine.engine.freed), sorted(b for b, _ in engine.engine.submitted)
        )
        # Batches still in flight when one failed are canceled before freeing.
        self.assertEqual(
            self._canceled_batches(engine),
            sorted(b for b, _ in engine.engine.submitted),
        )
        self.assertEqual(engine.engine.closed, [1])
        # The next transfer re-resolves the peer instead of reusing the stale handle.
        engine.batch_transfer_sync("10.0.0.2:7000", [1], [2], [8])
        self.assertEqual(engine.engine.opened, ["10.0.0.2:7000", "10.0.0.2:7000"])

    def test_submit_error_in_one_chunk_frees_chunks_already_submitted(self):
        """Chunks are submitted concurrently; one raising must not leak the others."""
        poison = 10**9
        engine, chunk = self._engine(raise_on_submit_source=poison)
        n = chunk * 4
        srcs = list(range(n))
        srcs[chunk * 2] = poison  # only the third chunk raises inside submit

        rc = engine.batch_transfer_sync("10.0.0.2:7000", srcs, [0] * n, [8] * n)

        self.assertLess(rc, 0)
        self.assertEqual(len(engine.engine.submitted), 3)
        # Every allocated batch, including the one whose submit raised, freed once.
        self.assertEqual(sorted(engine.engine.freed), sorted(engine.engine.allocated))
        self.assertEqual(len(engine.engine.allocated), 4)
        # Every allocated batch is canceled first: part of one may be posted.
        self.assertEqual(
            self._canceled_batches(engine), sorted(engine.engine.allocated)
        )

    def test_intent_reaches_every_request_and_none_keeps_default(self):
        """Foreground/background must map to TENT's QoS intents on every chunk."""
        from sglang.srt.distributed.device_communicators.kv_transfer_engine import (
            TransferIntent,
        )

        expected = {
            None: ("PRIO_HIGH", "INTENT_UNSPEC"),
            TransferIntent.FOREGROUND: ("PRIO_HIGH", "FOREGROUND_GET"),
            TransferIntent.BACKGROUND: ("PRIO_LOW", "BACKGROUND_PREFETCH"),
        }
        for intent, qos in expected.items():
            with self.subTest(intent=intent):
                engine, chunk = self._engine()
                n = chunk * 2 + 1
                rc = engine.batch_transfer_sync(
                    "10.0.0.2:7000", [1] * n, [2] * n, [8] * n, intent
                )
                self.assertEqual(rc, 0)
                self.assertEqual(engine.engine.qos, [qos] * n)

    def test_deregister_passes_registered_lengths(self):
        engine, _ = self._engine()
        self.assertEqual(engine.batch_register([10, 20], [4096, 8192]), 0)
        self.assertEqual(engine.batch_deregister([20, 10]), 0)
        self.assertEqual(engine.engine.unregistered, [([20, 10], [8192, 4096])])

    def test_session_id_matches_classic_host_port_format(self):
        engine, _ = self._engine()
        self.assertEqual(engine.get_session_id(), "10.0.0.1:12345")

    def test_config_pins_rdma_to_ib_devices_and_override_deep_merges(self):
        override = '{"transports": {"rdma": {"numa_penalties": [1.0, 2.0, 10.0]}}}'
        with envs.SGLANG_TENT_CONFIG_OVERRIDE.override(override):
            engine, _ = self._engine(ib_device="mlx5_0, mlx5_1")
        config = engine.engine.config
        self.assertEqual(config["topology"]["rdma_whitelist"], ["mlx5_0", "mlx5_1"])
        self.assertEqual(
            config["transports"]["rdma"],
            {"enable": True, "numa_penalties": [1.0, 2.0, 10.0]},
        )
        self.assertFalse(config["transports"]["nvlink"]["enable"])

    def test_metrics_port_is_unique_per_gpu_and_off_without_gpu_id(self):
        for gpu_id in (0, 3):
            with self.subTest(gpu_id=gpu_id):
                engine, _ = self._engine(gpu_id=gpu_id)
                self.assertEqual(
                    engine.engine.config["metrics"],
                    {"enabled": True, "http_port": 9100 + gpu_id},
                )
        engine, _ = self._engine(gpu_id=None)
        self.assertEqual(engine.engine.config["metrics"], {"enabled": False})

    def test_unsupported_protocol_is_rejected(self):
        for protocol in ("efa", "nvlink"):
            with self.subTest(protocol=protocol):
                with envs.MOONCAKE_PROTOCOL.override(protocol):
                    with self.assertRaises(ValueError):
                        self._engine()
        with envs.MOONCAKE_PROTOCOL.override("TCP"):
            engine, _ = self._engine()
        self.assertTrue(engine.engine.config["transports"]["tcp"]["enable"])
        self.assertFalse(engine.engine.config["transports"]["rdma"]["enable"])

    def test_force_tcp_disables_rdma_and_drops_whitelist(self):
        with patch.dict("os.environ", {"MC_FORCE_TCP": "1"}):
            engine, _ = self._engine(ib_device="mlx5_0")
        config = engine.engine.config
        self.assertFalse(config["transports"]["rdma"]["enable"])
        self.assertTrue(config["transports"]["tcp"]["enable"])
        self.assertNotIn("topology", config)

    def test_timeout_cancels_and_frees_the_pending_batch(self):
        engine, _ = self._engine(pending_batch_index=0)
        engine._timeout_ns = 0
        rc = engine.batch_transfer_sync("10.0.0.2:7000", [1, 2], [3, 4], [8, 8])

        self.assertLess(rc, 0)
        ((batch_id, _),) = engine.engine.submitted
        self.assertEqual(engine.engine.canceled, [(batch_id, 0), (batch_id, 1)])
        self.assertEqual(engine.engine.freed, [batch_id])
        self.assertEqual(engine.engine.closed, [1])

    def test_failed_transfer_closes_segment_only_after_last_user(self):
        """A failed layer must not close the handle other layers still write through."""
        engine, _ = self._engine()
        peer = "10.0.0.2:7000"
        first = engine._acquire_segment(peer)
        second = engine._acquire_segment(peer)
        self.assertEqual(first, second)
        self.assertEqual(engine.engine.opened, [peer])

        engine._release_segment(peer, failed=True)
        self.assertEqual(engine.engine.closed, [])
        # A transfer starting now still shares the stale handle.
        self.assertEqual(engine._acquire_segment(peer), first)
        engine._release_segment(peer, failed=False)
        self.assertEqual(engine.engine.closed, [])
        engine._release_segment(peer, failed=False)
        self.assertEqual(engine.engine.closed, [first])

        engine._acquire_segment(peer)
        self.assertEqual(engine.engine.opened, [peer, peer])

    def test_probe_refetches_descriptor_from_peer(self):
        peer = "10.0.0.2:7000"
        engine, _ = self._engine()
        self.assertEqual(engine.batch_transfer_sync(peer, [1], [2], [8]), 0)
        engine.engine.calls.clear()

        self.assertEqual(engine.send_probe(peer), 0)
        # Close first so get_segment_info cannot be served from TENT's cache.
        self.assertEqual(
            engine.engine.calls, [("close", 1), ("open", peer), ("info", 2)]
        )
        # A live peer keeps its fresh handle for the next transfer.
        self.assertEqual(engine._segments[peer].handle, 2)
        self.assertEqual(engine._segments[peer].users, 0)

    def test_probe_of_dead_peer_fails_and_drops_handle(self):
        peer = "10.0.0.2:7000"
        engine, _ = self._engine(peer_alive=False)
        self.assertLess(engine.send_probe(peer), 0)
        self.assertEqual(
            engine.engine.calls, [("open", peer), ("info", 1), ("close", 1)]
        )
        self.assertNotIn(peer, engine._segments)

    def test_probe_while_transfers_in_flight_reports_not_alive_yet(self):
        peer = "10.0.0.2:7000"
        engine, _ = self._engine()
        engine._acquire_segment(peer)
        self.assertLess(engine.send_probe(peer), 0)
        self.assertEqual(engine.engine.closed, [])
        self.assertEqual(engine.engine.segment_info, [])

    def test_transfer_timeout_matches_classic_te_floor_and_rejects_garbage(self):
        """Mirror mooncake.engine's MC_TRANSFER_TIMEOUT: 30 s default, 5 s floor."""
        from sglang.srt.distributed.device_communicators import tent_transfer_engine

        for value, seconds in {"": 30, "2": 5, "60": 60}.items():
            with self.subTest(value=value):
                with patch.dict(os.environ, {"MC_TRANSFER_TIMEOUT": value}):
                    self.assertEqual(
                        tent_transfer_engine._transfer_timeout_ns(), seconds * 10**9
                    )
        with patch.dict(os.environ, {"MC_TRANSFER_TIMEOUT": "30s"}):
            with self.assertRaises(ValueError):
                tent_transfer_engine._transfer_timeout_ns()


class TestTransferEngineSelector(CustomTestCase):
    def test_unknown_backend_is_rejected(self):
        from sglang.srt.distributed.device_communicators import (
            mooncake_transfer_engine as mte,
        )

        with patch.object(mte, "_mooncake_transfer_engine", None):
            with envs.SGLANG_MOONCAKE_TRANSFER_ENGINE.override("rdma"):
                with self.assertRaises(ValueError):
                    mte.init_mooncake_transfer_engine(hostname="10.0.0.1")

    def test_missing_tent_module_names_the_mooncake_version(self):
        from sglang.srt.distributed.device_communicators import (
            mooncake_transfer_engine as mte,
        )

        with (
            patch.object(mte, "_mooncake_transfer_engine", None),
            patch.dict(sys.modules, {"tent": None}),
        ):
            with envs.SGLANG_MOONCAKE_TRANSFER_ENGINE.override("tent"):
                with self.assertRaisesRegex(
                    ImportError, r"mooncake-transfer-engine>=0\.3\.14"
                ):
                    mte.init_mooncake_transfer_engine(hostname="10.0.0.1")
            self.assertIsNone(mte._mooncake_transfer_engine)

    def test_tent_backend_receives_gpu_id(self):
        from sglang.srt.distributed.device_communicators import (
            mooncake_transfer_engine as mte,
        )
        from sglang.srt.distributed.device_communicators import (
            tent_transfer_engine,
        )

        with (
            patch.object(mte, "_mooncake_transfer_engine", None),
            patch.object(tent_transfer_engine, "TentTransferEngine") as tent_cls,
        ):
            with envs.SGLANG_MOONCAKE_TRANSFER_ENGINE.override("tent"):
                mte.init_mooncake_transfer_engine(
                    hostname="10.0.0.1",
                    gpu_id=3,
                    ib_device='{"0": "mlx5_0", "3": "mlx5_3"}',
                )
        tent_cls.assert_called_once_with(
            hostname="10.0.0.1", ib_device="mlx5_3", gpu_id=3
        )


if __name__ == "__main__":
    unittest.main()
