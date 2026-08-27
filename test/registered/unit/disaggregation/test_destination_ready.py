"""Unit coverage for the default-unused destination-GPU epoch helper."""

import unittest

import torch

from sglang.srt.disaggregation.common.destination_ready import (
    ReadyEpochStaleError,
    ReadyEpochTimeoutError,
    enqueue_driver_ready_epoch_acquire,
    wait_for_destination_ready_epoch,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestDestinationReadyValidation(CustomTestCase):
    def test_rejects_invalid_epoch_inputs(self):
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            wait_for_destination_ready_epoch(None, 1)
        with self.assertRaisesRegex(ValueError, "dtype"):
            wait_for_destination_ready_epoch(torch.zeros(1, dtype=torch.int64), 1)
        with self.assertRaisesRegex(ValueError, "shape"):
            wait_for_destination_ready_epoch(torch.zeros(2, dtype=torch.uint64), 1)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            wait_for_destination_ready_epoch(torch.zeros(1, dtype=torch.uint64), 1)
        with self.assertRaisesRegex(ValueError, "expected_epoch"):
            wait_for_destination_ready_epoch(torch.zeros(1, dtype=torch.uint64), 0)
        with self.assertRaisesRegex(ValueError, "max_spins"):
            wait_for_destination_ready_epoch(
                torch.zeros(1, dtype=torch.uint64), 1, max_spins=0
            )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDestinationReadyCuda(CustomTestCase):
    def _ready(self, epoch: int = 0) -> torch.Tensor:
        return torch.tensor([epoch], dtype=torch.uint64, device="cuda")

    def test_accepts_changing_epochs(self):
        ready = self._ready(1)
        self.assertEqual(wait_for_destination_ready_epoch(ready, 1), 1)
        ready.fill_(2)
        self.assertEqual(wait_for_destination_ready_epoch(ready, 2), 2)

    def test_rejects_stale_epoch(self):
        with self.assertRaisesRegex(ReadyEpochStaleError, "observed epoch 1"):
            wait_for_destination_ready_epoch(self._ready(1), 2, max_spins=1)

    def test_rejects_timeout(self):
        with self.assertRaisesRegex(ReadyEpochTimeoutError, "timed out"):
            wait_for_destination_ready_epoch(self._ready(), 1, max_spins=1)

    def test_enqueues_on_current_stream(self):
        ready = self._ready()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ready.fill_(7)
            self.assertEqual(wait_for_destination_ready_epoch(ready, 7), 7)

    def test_driver_wait_orders_consumer_stream(self):
        ready = self._ready()
        waiter = torch.cuda.Stream()
        consumer = torch.cuda.Stream()
        producer = torch.cuda.Stream()
        observed = torch.empty_like(ready)

        with torch.cuda.stream(waiter):
            event = enqueue_driver_ready_epoch_acquire(ready, 7, stream=waiter)
        with torch.cuda.stream(consumer):
            consumer.wait_event(event)
            observed.copy_(ready)
        with torch.cuda.stream(producer):
            ready.fill_(7)

        torch.cuda.synchronize()
        self.assertEqual(int(observed.item()), 7)

    def test_driver_wait_can_be_released_from_another_stream(self):
        ready = self._ready()
        waiter = torch.cuda.Stream()
        event = enqueue_driver_ready_epoch_acquire(ready, 7, stream=waiter)

        self.assertFalse(event.query())
        ready.fill_(7)
        event.synchronize()

        self.assertTrue(event.query())
