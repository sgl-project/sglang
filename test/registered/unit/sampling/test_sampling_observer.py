"""Unit tests for srt/sampling/sampling_observer.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import torch

from sglang.srt.sampling.sampling_observer import (
    CommittedTokens,
    DeviceAuxiliaryOutput,
    HostAuxiliaryOutput,
    SamplingObserver,
)
from sglang.test.test_utils import CustomTestCase


class _DeviceOutput(DeviceAuxiliaryOutput):
    """A conforming DeviceAuxiliaryOutput implementation."""

    def __init__(self, token_ids=(1, 2)):
        self.token_ids = tuple(token_ids)

    def copy_to_host(self, copy_tensor):
        return _HostOutput(self.token_ids)


class _HostOutput(HostAuxiliaryOutput):
    """A conforming HostAuxiliaryOutput implementation."""

    def __init__(self, token_ids=()):
        self.token_ids = tuple(token_ids)
        self.seen_batch = None
        self.seen_commits = None

    def consume(self, batch, commits):
        self.seen_batch = batch
        self.seen_commits = commits


class _Observer(SamplingObserver):
    """A conforming SamplingObserver implementation."""

    def __init__(self, active=True):
        self.active = active
        self.events = []

    def is_active(self, sampling_info):
        return self.active

    def before_grammar(self, logits, sampling_info):
        self.events.append(("before", logits.clone()))
        return object() if self.active else None

    def after_sample(self, state, token_ids):
        self.events.append(("after", state, token_ids))
        return None


class TestCommittedTokens(CustomTestCase):

    def test_fields(self):
        """CommittedTokens carries an output index and a tuple of token ids."""
        ct = CommittedTokens(output_index=3, token_ids=(1, 2, 3))
        self.assertEqual(ct.output_index, 3)
        self.assertEqual(ct.token_ids, (1, 2, 3))

    def test_frozen_raises_on_mutation(self):
        """CommittedTokens must be immutable (frozen dataclass)."""
        ct = CommittedTokens(output_index=0, token_ids=(1,))
        with self.assertRaises(FrozenInstanceError):
            ct.output_index = 1

    def test_equality(self):
        """Equal field values imply equal instances."""
        self.assertEqual(
            CommittedTokens(0, (1, 2)),
            CommittedTokens(output_index=0, token_ids=(1, 2)),
        )
        self.assertNotEqual(
            CommittedTokens(0, (1, 2)),
            CommittedTokens(1, (1, 2)),
        )

    def test_hashable(self):
        """CommittedTokens must be usable as a dict key / set member."""
        ct = CommittedTokens(0, (1, 2))
        self.assertEqual(hash(ct), hash(CommittedTokens(0, (1, 2))))
        self.assertIn(ct, {CommittedTokens(0, (1, 2))})

    def test_tuple_semantics(self):
        """token_ids must behave as a tuple (immutable, iterable)."""
        ct = CommittedTokens(0, (1, 2))
        self.assertIsInstance(ct.token_ids, tuple)
        self.assertEqual(list(ct.token_ids), [1, 2])


class TestProtocolConformance(CustomTestCase):
    """Lock the extension-contract surface defined by the protocols."""

    def test_device_output_surface(self):
        """DeviceAuxiliaryOutput implementers must expose copy_to_host."""
        output = _DeviceOutput()
        self.assertTrue(hasattr(output, "copy_to_host"))
        # A conforming implementer subclasses the protocol and returns a host output.
        self.assertIsInstance(output, _DeviceOutput)
        self.assertTrue(hasattr(output.copy_to_host(torch.clone), "consume"))

    def test_host_output_surface(self):
        """HostAuxiliaryOutput implementers must expose consume."""
        host = _HostOutput()
        self.assertTrue(hasattr(host, "consume"))

    def test_sampling_observer_surface(self):
        """SamplingObserver implementers must expose the three hooks."""
        observer = _Observer()
        self.assertTrue(hasattr(observer, "is_active"))
        self.assertTrue(hasattr(observer, "before_grammar"))
        self.assertTrue(hasattr(observer, "after_sample"))

    def test_non_conforming_objects_missing_surface(self):
        """Plain objects must not satisfy the protocol surfaces."""
        plain = object()
        self.assertFalse(hasattr(plain, "copy_to_host"))
        self.assertFalse(hasattr(plain, "consume"))
        self.assertFalse(hasattr(plain, "is_active"))


class TestAuxiliaryOutputFlow(CustomTestCase):
    """The documented device -> host -> consume pipeline."""

    def test_copy_to_host_returns_consumable_host_output(self):
        """Device output must copy to a host output that exposes consume."""
        device = _DeviceOutput(token_ids=(5, 6))
        host = device.copy_to_host(copy_tensor=torch.clone)
        self.assertTrue(hasattr(host, "consume"))
        self.assertEqual(host.token_ids, (5, 6))

    def test_consume_receives_commits_aligned_with_batch(self):
        """HostAuxiliaryOutput.consume gets commits aligned with batch.reqs."""
        batch = MagicMock()
        batch.reqs = [MagicMock(), MagicMock()]
        commits = [
            CommittedTokens(output_index=0, token_ids=(1,)),
            CommittedTokens(output_index=1, token_ids=(2,)),
        ]

        host = _HostOutput()
        host.consume(batch, commits)

        self.assertIs(host.seen_batch, batch)
        self.assertEqual(host.seen_commits, commits)
        self.assertEqual(len(host.seen_commits), len(batch.reqs))

    def test_consume_accepts_none_entries(self):
        """Commits may contain None entries (not all requests produce tokens)."""
        batch = MagicMock()
        batch.reqs = [MagicMock()]
        host = _HostOutput()
        host.consume(batch, [None])
        self.assertEqual(host.seen_commits, [None])


if __name__ == "__main__":
    unittest.main()
