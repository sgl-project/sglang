"""Unit tests for ``BreakableCudaGraphBackend`` output handling of
``LogitsProcessorOutput`` — CPU-only.

Black-box statement of the bug these guard: BCG sized its replay buffer and its
row count from the graph's batch size, on the assumption that a captured body
never emits more leading rows than there are requests. A speculative-verify
decode body breaks that assumption — its row unit is tokens, so it emits
``size * num_draft_tokens`` rows. Two failures followed:

  * ``_output_rows`` clamped the count down to ``size``, so the verify step saw
    only ``1 / num_draft_tokens`` of the logits and indexed ``accept_index``
    past the end of the tensor (a device-side assert, surfacing as a hang).
  * ``_alloc_full_buffer`` reserved only ``size`` rows, so the copy-back had
    nowhere to put the remaining tokens.

``LogitsProcessorOutput`` was also not an accepted output structure at all
(``TypeError`` from the alloc / slice / copy helpers), which is why the decode
path had to run eager whenever speculative decoding was on.

The capture path itself needs CUDA, so the graph and capture-context objects are
mocked; the logic under test (row counting, buffer sizing, structured copy and
slice) is pure tensor bookkeeping and runs on CPU.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MODULE = "sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend"
_VOCAB = 8
_HIDDEN = 4


class _FakeCapture:
    """Stand-in for ``BreakableCUDAGraphCapture`` — a no-op context manager."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_backend():
    """Build the backend without ``__init__`` (which would touch CUDA), wiring
    only the attributes ``capture_one`` reads."""
    backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
    backend._graphs = {}
    backend._outputs = {}
    backend._capture_inputs = {}
    backend._pool = None
    backend._capture_stream = None
    backend._shared_output_buffer = None
    backend._debug_eager = False
    backend.deduped_cuda_graph = None
    backend._device_module = SimpleNamespace(synchronize=mock.Mock())
    backend._tp_group = SimpleNamespace(barrier=mock.Mock())
    return backend


def _lpo(rows, *, with_hidden=True, fill=0.0):
    return LogitsProcessorOutput(
        next_token_logits=torch.full((rows, _VOCAB), fill),
        hidden_states=torch.full((rows, _HIDDEN), fill) if with_hidden else None,
    )


class TestOutputRows(CustomTestCase):
    def test_reports_token_rows_for_speculative_verify(self):
        # size=3 requests x num_draft_tokens=4 => 12 token rows. Reporting the
        # request count here is what truncated the verify logits.
        backend = _make_backend()
        self.assertEqual(backend._output_rows(_lpo(12), 3), 12)

    def test_no_tensor_fields_falls_back_to_cap(self):
        backend = _make_backend()
        empty = LogitsProcessorOutput(next_token_logits=None, hidden_states=None)
        self.assertEqual(backend._output_rows(empty, 5), 5)

    def test_tensor_output_still_clamps_to_cap(self):
        # Guards the pre-existing contract for every other output structure: a
        # body that prunes rows must still report at most ``cap``.
        backend = _make_backend()
        self.assertEqual(backend._output_rows(torch.zeros(8, _VOCAB), 4), 4)
        self.assertEqual(backend._output_rows(torch.zeros(2, _VOCAB), 4), 2)

    def test_field_row_mismatch_raises(self):
        backend = _make_backend()
        mismatched = LogitsProcessorOutput(
            next_token_logits=torch.zeros(12, _VOCAB),
            hidden_states=torch.zeros(3, _HIDDEN),
        )
        with self.assertRaises(ValueError):
            backend._output_rows(mismatched, 3)

    def test_unexpected_populated_field_raises(self):
        # A capture point that also filled a Sampler field would otherwise have
        # it silently dropped from the replay buffer.
        backend = _make_backend()
        output = _lpo(4)
        output.next_token_logprobs = torch.zeros(4)
        with self.assertRaises(TypeError):
            backend._output_rows(output, 4)


class TestSliceAndCopy(CustomTestCase):
    def test_alloc_slice_copy_roundtrip(self):
        backend = _make_backend()
        buffer = backend._alloc_full_buffer(_lpo(12), 12)
        self.assertEqual(buffer.next_token_logits.shape, (12, _VOCAB))
        self.assertEqual(buffer.hidden_states.shape, (12, _HIDDEN))

        produced = _lpo(12, fill=7.0)
        backend._copy_output_to_buffer(produced, buffer, 12)
        stored = backend._slice_output(buffer, 12)
        self.assertTrue(
            torch.equal(stored.next_token_logits, produced.next_token_logits)
        )
        self.assertTrue(torch.equal(stored.hidden_states, produced.hidden_states))

    def test_none_field_is_preserved_across_alloc_and_slice(self):
        backend = _make_backend()
        buffer = backend._alloc_full_buffer(_lpo(6, with_hidden=False), 6)
        self.assertIsNone(buffer.hidden_states)
        stored = backend._slice_output(buffer, 6)
        self.assertIsNone(stored.hidden_states)
        self.assertEqual(stored.next_token_logits.shape, (6, _VOCAB))


class TestCaptureOne(CustomTestCase):
    def _capture(self, backend, *, size, forward_fn):
        with mock.patch(
            f"{_MODULE}.BreakableCUDAGraph", return_value="GRAPH"
        ), mock.patch(f"{_MODULE}.BreakableCUDAGraphCapture", _FakeCapture):
            backend.capture_one(ShapeKey(size=size), forward_fn)

    def test_buffer_holds_token_rows_not_request_rows(self):
        backend = _make_backend()
        self._capture(backend, size=3, forward_fn=lambda: _lpo(12, fill=2.0))

        buffer = backend._shared_output_buffer
        self.assertEqual(buffer.next_token_logits.shape, (12, _VOCAB))

        stored = backend._outputs[ShapeKey(size=3)]
        self.assertEqual(stored.next_token_logits.shape, (12, _VOCAB))
        self.assertTrue(
            torch.equal(stored.next_token_logits, torch.full((12, _VOCAB), 2.0))
        )

    def test_tensor_body_keeps_full_size_buffer(self):
        # A pruning tensor body must keep its full-``size`` buffer: later
        # captures of the same shared buffer may not prune.
        backend = _make_backend()
        self._capture(backend, size=4, forward_fn=lambda: torch.full((2, _VOCAB), 3.0))

        self.assertEqual(backend._shared_output_buffer.shape, (4, _VOCAB))
        stored = backend._outputs[ShapeKey(size=4)]
        self.assertEqual(stored.shape, (2, _VOCAB))


if __name__ == "__main__":
    unittest.main()
