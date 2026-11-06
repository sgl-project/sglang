"""CPU tests for the SGLANG_DEBUG_HICACHE_VERIFY load-back check.

Turns red if the transfer engine stops recording backups or checking loads,
or if the row hash stops covering every byte of a K/V or Mamba row.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import hicache_verify as verify_module
from sglang.srt.mem_cache import l2_transfer as transfer_module
from sglang.srt.mem_cache.hicache_verify import HiCacheVerifier
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _FakeEvent:
    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing

    def record(self):
        pass

    def wait(self, stream):
        pass

    def synchronize(self):
        pass


class _FakeDeviceModule:
    Event = _FakeEvent

    @staticmethod
    def Stream():
        return object()

    @staticmethod
    @contextmanager
    def stream(stream):
        yield


class _CopyHostPool:
    """Keeps backed-up K/V rows by host slot; `corrupt` flips a byte of one slot."""

    layer_num = 1

    def __init__(self):
        self.rows = {}

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        return host_indices, device_indices

    def backup_from_device_all_layer_physical(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        for host, device in zip(host_indices.tolist(), device_indices.tolist()):
            self.rows[host] = (
                device_pool.k_buffer[0][device].clone(),
                device_pool.v_buffer[0][device].clone(),
            )

    def load_to_device_per_layer_physical(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        is_draft=False,
    ):
        for host, device in zip(host_indices.tolist(), device_indices.tolist()):
            device_pool.k_buffer[0][device] = self.rows[host][0]
            device_pool.v_buffer[0][device] = self.rows[host][1]

    def corrupt(self, host):
        self.rows[host][1].view(torch.uint8).view(-1)[3] ^= 1


def _mha_pool(num_rows=8):
    return SimpleNamespace(
        k_buffer=[torch.randn(num_rows, 2, 4, dtype=torch.bfloat16)],
        v_buffer=[torch.randn(num_rows, 2, 4, dtype=torch.bfloat16)],
        device="cpu",
    )


def _mamba_pool(num_slots=6):
    pool = object.__new__(MambaPool)
    pool.mamba_cache = SimpleNamespace(
        conv=[torch.randn(2, num_slots, 3, 2)],
        temporal=torch.randn(2, num_slots, 2, 4, 4),
    )
    return pool


def _device_indices(transfer):
    return transfer.device_indices


def _transfer(host_pool, device_pool, host, device):
    return L2Transfer(
        host_pool=host_pool,
        device_pool=device_pool,
        host_indices=torch.tensor(host),
        device_indices=torch.tensor(device),
    )


class TestHiCacheVerify(CustomTestCase):
    def setUp(self):
        transfer_module._timing_events_supported.cache_clear()
        self.addCleanup(transfer_module._timing_events_supported.cache_clear)
        patcher = mock.patch.object(transfer_module, "device_module", _FakeDeviceModule)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _engine(self):
        with envs.SGLANG_DEBUG_HICACHE_VERIFY.override(True):
            return L2TransferEngine("kernel")

    def test_disabled_by_default(self):
        self.assertIsNone(L2TransferEngine("kernel")._verifier)

    def test_engine_checks_every_load_back(self):
        engine, host_pool, device_pool = self._engine(), _CopyHostPool(), _mha_pool()
        engine.submit_device_to_host(
            [_transfer(host_pool, device_pool, [10, 11, 12], [0, 1, 2])]
        )
        engine.submit_host_to_device(
            [_transfer(host_pool, device_pool, [10, 11, 12], [5, 6, 7])],
            transfer_layer_id_max=1,
        )
        counts = engine._verifier.counts
        self.assertEqual((counts["checked"], counts["mismatch"]), (3, 0))

        host_pool.corrupt(11)
        with self.assertLogs("sglang.srt.mem_cache.hicache_verify", "ERROR"):
            engine.submit_host_to_device(
                [_transfer(host_pool, device_pool, [10, 11], [3, 4])],
                transfer_layer_id_max=1,
            )
        self.assertEqual((counts["checked"], counts["mismatch"]), (5, 1))

    def test_mamba_rows_and_slots_without_backup(self):
        verifier = HiCacheVerifier(resolve_rows=_device_indices)
        host_pool, pool = object(), _mamba_pool()
        verifier.record_backup([_transfer(host_pool, pool, [0, 1], [0, 1])])
        cache = pool.mamba_cache
        for state in (*cache.conv, cache.temporal):
            state[:, [2, 3]] = state[:, [0, 1]]
        cache.temporal[1, 3, 0, 0, 0] += 1

        verifier.check_load([_transfer(host_pool, pool, [0, 1, 7], [2, 3, 4])])

        counts = verifier.counts
        self.assertEqual(
            (counts["checked"], counts["mismatch"], counts["unverified"]), (2, 1, 1)
        )

    def test_every_chunk_of_a_row_is_hashed(self):
        verify_module._coeffs.cache_clear()
        self.addCleanup(verify_module._coeffs.cache_clear)
        verifier = HiCacheVerifier(resolve_rows=_device_indices)
        host_pool = object()
        pool = SimpleNamespace(
            k_buffer=[torch.randn(4, 4, 8, dtype=torch.bfloat16)],
            v_buffer=[torch.randn(4, 4, 8, dtype=torch.bfloat16)],
            device="cpu",
        )
        with mock.patch.object(verify_module, "_CHUNK_WORDS", 2):
            verifier.record_backup([_transfer(host_pool, pool, [0, 1], [0, 1])])
            for buf in (*pool.k_buffer, *pool.v_buffer):
                buf[[2, 3]] = buf[[0, 1]]
            pool.k_buffer[0][3].view(torch.uint8).view(-1)[60] ^= 1
            verifier.check_load([_transfer(host_pool, pool, [0, 1], [2, 3])])
        counts = verifier.counts
        self.assertEqual((counts["checked"], counts["mismatch"]), (2, 1))

    def test_storage_backend_turns_the_check_off(self):
        engine, host_pool, device_pool = self._engine(), _CopyHostPool(), _mha_pool()
        with self.assertLogs("sglang.srt.mem_cache.l2_transfer", "WARNING"):
            engine.disable_verifier(reason="a storage backend is attached")
        self.assertIsNone(engine._verifier)
        engine.submit_device_to_host(
            [_transfer(host_pool, device_pool, [10, 11], [0, 1])]
        )
        engine.submit_host_to_device(
            [_transfer(host_pool, device_pool, [10, 11], [5, 6])],
            transfer_layer_id_max=1,
        )

    def test_unsupported_pool_rows_are_unverified(self):
        verifier = HiCacheVerifier(resolve_rows=_device_indices)
        pool = SimpleNamespace(device="cpu")
        verifier.record_backup([_transfer(object(), pool, [0, 1], [0, 1])])
        verifier.check_load([_transfer(object(), pool, [0, 1], [2, 3])])
        self.assertEqual(verifier.counts["checked"], 0)
        self.assertEqual(verifier.counts["unverified"], 2)


if __name__ == "__main__":
    unittest.main()
