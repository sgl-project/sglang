import unittest
import warnings
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.test.unit.host_sync_guard import (
    SyncOnRead,
    assert_no_cuda_sync,
    forbid_host_syncs,
    no_cuda_sync,
)


def _read_device_value():
    # what torch's sync debug mode emits for an implicit sync
    warnings.warn("called a synchronizing CUDA operation")


def _synchronize_explicitly():
    torch.cuda.synchronize()


def _calls_a_sync():
    _read_device_value()


class TestSyncOnRead(unittest.TestCase):
    def test_host_reads_raise(self):
        values = torch.tensor([1.0, 0.0]).as_subclass(SyncOnRead)
        reads = {
            "item": lambda: values[0].item(),
            "float": lambda: float(values[0]),
            "bool": lambda: bool(values[0] > 0.5),
            "index": lambda: [0, 1][values.long()[0]],
            "tolist": values.tolist,
            "nonzero": values.nonzero,
            "boolean mask": lambda: values[values > 0],
        }
        for name, read in reads.items():
            with self.subTest(read=name), self.assertRaises(AssertionError):
                read()

    def test_device_math_and_host_copies_do_not_raise(self):
        values = torch.tensor([1.0, 0.5]).as_subclass(SyncOnRead)
        delta = values[1] - values[0]
        self.assertIsInstance(delta, SyncOnRead)
        self.assertEqual(values.cpu()[1].item(), 0.5)
        self.assertIsInstance(values.to(values.device), SyncOnRead)


class TestForbidHostSyncs(unittest.TestCase):
    def test_syncing_ops_raise(self):
        values = torch.tensor([1.0, 0.0, 2.0])
        mask = values > 0.5
        syncs = {
            "item": lambda: values[0].item(),
            "bool": lambda: bool(values[0] > 0.5),
            "nonzero": values.nonzero,
            "masked_select": lambda: values.masked_select(mask),
            "boolean index": lambda: values[mask],
            "boolean index_put": lambda: values.clone().index_put_((mask,), values[:1]),
            "repeat_interleave": lambda: torch.repeat_interleave(values.long()),
            "full from a tensor": lambda: torch.full((2,), values[0]),
        }
        for name, sync in syncs.items():
            with self.subTest(op=name), self.assertRaises(AssertionError):
                with forbid_host_syncs():
                    sync()

    def test_device_side_ops_pass(self):
        values = torch.tensor([1.0, 0.0, 2.0])
        mask = values > 0.5
        with forbid_host_syncs():
            selected = torch.where(mask, values * 0.0, values)
            gathered = values.index_select(0, torch.tensor([2, 0]))
            repeated = torch.repeat_interleave(
                values, torch.tensor([1, 0, 2]), output_size=3
            )
        self.assertTrue(torch.equal(selected, torch.tensor([0.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(gathered, torch.tensor([2.0, 1.0])))
        self.assertTrue(torch.equal(repeated, torch.tensor([1.0, 2.0, 2.0])))


class TestNoCudaSync(unittest.TestCase):
    def _fake_cuda(self):
        modes = []
        synchronize = Mock()
        patches = (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_sync_debug_mode", return_value=0),
            patch.object(torch.cuda, "set_sync_debug_mode", side_effect=modes.append),
            patch.object(torch.cuda, "synchronize", synchronize),
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return modes, synchronize

    def test_reports_each_sync_with_its_call_site(self):
        modes, synchronize = self._fake_cuda()
        with self.assertRaises(AssertionError) as raised:
            with no_cuda_sync():
                _read_device_value()
                _synchronize_explicitly()
        message = str(raised.exception)
        self.assertIn("2 host-device sync(s)", message)
        self.assertIn("_read_device_value", message)
        self.assertIn("_synchronize_explicitly", message)
        # the explicit synchronize still ran, and the debug mode was restored
        synchronize.assert_called_once()
        self.assertEqual(modes, ["warn", 0])

    def test_allow_names_intentional_syncs(self):
        self._fake_cuda()
        allow = (
            "test_host_sync_guard.py:_read_device_value",
            "test_host_sync_guard.py:_synchronize_explicitly",
        )
        with no_cuda_sync(allow=allow):
            _read_device_value()
            _synchronize_explicitly()
        self.assertEqual(assert_no_cuda_sync(lambda: 3, allow=allow), 3)

    def test_allow_does_not_cover_callees(self):
        self._fake_cuda()
        with self.assertRaises(AssertionError):
            with no_cuda_sync(allow=("test_host_sync_guard.py:_calls_a_sync",)):
                _calls_a_sync()

    def test_other_warnings_pass_through(self):
        self._fake_cuda()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with no_cuda_sync():
                warnings.warn("unrelated")
        self.assertEqual([str(w.message) for w in caught], ["unrelated"])

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_reports_item_and_pageable_upload(self):
        device_value = torch.ones(2, device="cuda")
        with no_cuda_sync():
            device_value = device_value * 2 + 1
        for sync in (
            lambda: device_value.sum().item(),
            lambda: torch.tensor([1.0, 2.0], device="cuda"),
            lambda: device_value.nonzero(),
        ):
            with self.assertRaises(AssertionError), no_cuda_sync():
                sync()


if __name__ == "__main__":
    unittest.main()
