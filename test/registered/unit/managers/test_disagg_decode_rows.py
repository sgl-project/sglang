"""PD decode joins the prebuilt rows into the running rows at the plan,
through a copy: the prebuilt batch never becomes the rows."""

import copy
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import (  # noqa: E402
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.managers.scheduler_components.decoding_batch import (  # noqa: E402
    DecodingBatch,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.done = False

    def finished(self):
        return self.done


class _Batch:
    """Keeps the row order and the filter and merge rules of ScheduleBatch."""

    def __init__(self, reqs, mode=ForwardMode.DECODE):
        self.reqs = list(reqs)
        self.forward_mode = mode
        self.num_decode_rows = 0
        self.sampling_info = SimpleNamespace(penalizer_orchestrator=SimpleNamespace())

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, reqs_to_exclude=()):
        self.reqs = [
            r for r in self.reqs if not r.finished() and r not in reqs_to_exclude
        ]

    def merge_batch(self, other):
        self.reqs = self.reqs + other.reqs

    def shallow_copy(self):
        batch = copy.copy(self)
        batch.reqs = self.reqs[:]
        return batch


def _rids(reqs):
    return [r.rid for r in reqs]


def _plan_scheduler(prebuilt_reqs):
    """A scheduler whose PD decode plan finds prebuilt_reqs ready to join and
    selects every running row for decode."""
    s = MagicMock()
    s.chunked_req = None
    s.enable_hisparse = False
    s.last_batch = None
    s.disagg_decode_launches_copy = False
    s.get_next_disagg_decode_batch_to_run = MethodType(
        SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run, s
    )
    s.prebuilt = None

    def get_new_prebuilt_batch(running_batch):
        if not prebuilt_reqs:
            return None
        s.prebuilt = _Batch(prebuilt_reqs, ForwardMode.PREBUILT)
        return s.prebuilt

    s.get_new_prebuilt_batch.side_effect = get_new_prebuilt_batch
    s.update_running_batch.side_effect = lambda decoding_batch: decoding_batch.rows
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = lambda b: b
    return s


class TestPrebuiltJoin(CustomTestCase):
    def _plan(self, s, decoding_batch):
        with patch("sglang.srt.disaggregation.decode.set_schedule_time_batch"):
            return s.get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch)

    def test_join_into_empty_rows_takes_a_copy(self):
        s = _plan_scheduler([_Req("p")])
        decoding_batch = DecodingBatch(_Batch([]))

        self._plan(s, decoding_batch)

        rows = decoding_batch.rows
        self.assertIsNot(rows, s.prebuilt)
        self.assertEqual(_rids(rows.reqs), ["p"])
        # The penalizers read their requests from the rows.
        self.assertIs(rows.sampling_info.penalizer_orchestrator.batch, rows)

    def test_join_into_running_rows_appends(self):
        s = _plan_scheduler([_Req("p")])
        decoding_batch = DecodingBatch(_Batch([_Req("a"), _Req("b")]))
        rows = decoding_batch.rows

        self._plan(s, decoding_batch)

        self.assertIs(decoding_batch.rows, rows)
        self.assertEqual(_rids(rows.reqs), ["a", "b", "p"])


if __name__ == "__main__":
    unittest.main()
