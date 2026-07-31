import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _batch(*, extend=False, decode=False):
    forward_mode = MagicMock()
    forward_mode.is_extend.return_value = extend
    forward_mode.is_decode.return_value = decode
    return SimpleNamespace(forward_mode=forward_mode)


def _running_batch(*, empty=False, prefill_only=False):
    batch = MagicMock()
    batch.is_empty.return_value = empty
    batch.is_prefill_only = prefill_only
    return batch


class TestMixedChunkDecodeInterleave(CustomTestCase):
    def setUp(self):
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.mixed_chunk_decode_interleave_steps = 2
        self.scheduler.mixed_chunk_decode_steps_remaining = 0
        self.scheduler.is_mixed_chunk = True
        self.scheduler.chunked_req = None

    def test_runs_configured_decode_steps_after_extend(self):
        self.scheduler._update_mixed_chunk_decode_interleave_budget(_batch(extend=True))
        self.assertTrue(
            self.scheduler._should_interleave_mixed_chunk_decode(_running_batch())
        )

        self.scheduler._update_mixed_chunk_decode_interleave_budget(_batch(decode=True))
        self.assertTrue(
            self.scheduler._should_interleave_mixed_chunk_decode(_running_batch())
        )

        self.scheduler._update_mixed_chunk_decode_interleave_budget(_batch(decode=True))
        self.assertFalse(
            self.scheduler._should_interleave_mixed_chunk_decode(_running_batch())
        )

    def test_zero_steps_disables_interleave(self):
        self.scheduler.mixed_chunk_decode_interleave_steps = 0
        self.scheduler._update_mixed_chunk_decode_interleave_budget(_batch(extend=True))

        self.assertFalse(
            self.scheduler._should_interleave_mixed_chunk_decode(_running_batch())
        )

    def test_active_chunked_request_continues_prefill(self):
        self.scheduler.mixed_chunk_decode_steps_remaining = 2
        self.scheduler.chunked_req = MagicMock()

        self.assertFalse(
            self.scheduler._should_interleave_mixed_chunk_decode(_running_batch())
        )

    def test_empty_or_prefill_only_batch_cannot_decode(self):
        self.scheduler.mixed_chunk_decode_steps_remaining = 2

        self.assertFalse(
            self.scheduler._should_interleave_mixed_chunk_decode(
                _running_batch(empty=True)
            )
        )
        self.assertFalse(
            self.scheduler._should_interleave_mixed_chunk_decode(
                _running_batch(prefill_only=True)
            )
        )


if __name__ == "__main__":
    unittest.main()
