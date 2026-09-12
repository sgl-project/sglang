"""Lifetime of the DSA multi-CTAs KV counter buffer across CUDA graph capture.

The decode graphs record this buffer's device address, so it must be sized for
every query-row count capture will use, and must never be replaced afterwards.
Two black-box behaviors must not come back:

- capture growing the counter, which happens when the persistent buffer is sized
  from ``max_running_requests`` while capture indexes by expanded query rows
  (target verify / draft extend multiply each request by the draft-token count);
- an oversized eager call replacing the captured allocation, which frees the
  buffer the graphs replay against.
"""

import unittest

import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLM_MLA_MAX_BATCH_SIZE,
    grow_multi_ctas_kv_counter_buffer_if_needed,
    make_persistent_multi_ctas_kv_counter_buffer,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")

_NUM_Q_HEADS = 128
# 1024 captured requests x 9 draft tokens: a supported speculative capture whose
# query-row count exceeds TRTLLM_MLA_MAX_BATCH_SIZE.
_CAPTURED_BS = 1024
_DRAFT_TOKENS = 9
_CAPTURED_ROWS = _CAPTURED_BS * _DRAFT_TOKENS


def _make_backend(max_running_requests: int):
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.device = "cuda"
    backend.num_q_heads = _NUM_Q_HEADS
    backend._multi_ctas_kv_counter_buffer = (
        make_persistent_multi_ctas_kv_counter_buffer(
            torch.device("cuda"),
            _NUM_Q_HEADS,
            max_batch_size=max_running_requests,
        )
    )
    return backend


def _would_grow(backend, rows: int) -> bool:
    return (
        grow_multi_ctas_kv_counter_buffer_if_needed(
            backend._multi_ctas_kv_counter_buffer,
            torch.device("cuda"),
            backend.num_q_heads,
            rows,
        )
        is not backend._multi_ctas_kv_counter_buffer
    )


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestMultiCtasKvCounterLifetime(CustomTestCase):
    def test_capture_row_count_alone_would_grow_the_counter(self):
        """The premise: sizing by requests undercounts captured query rows."""
        self.assertGreater(_CAPTURED_ROWS, TRTLLM_MLA_MAX_BATCH_SIZE)
        backend = _make_backend(max_running_requests=48)
        self.assertTrue(_would_grow(backend, _CAPTURED_ROWS))

    def test_sizing_for_query_rows_stops_capture_from_growing(self):
        """After sizing, capture at that row count must not reallocate."""
        backend = _make_backend(max_running_requests=48)
        backend._ensure_multi_ctas_kv_counter_capacity(_CAPTURED_ROWS)
        self.assertFalse(_would_grow(backend, _CAPTURED_ROWS))

    def test_sizing_is_grow_only(self):
        """A later, smaller request must not discard an earlier allocation."""
        backend = _make_backend(max_running_requests=48)
        backend._ensure_multi_ctas_kv_counter_capacity(_CAPTURED_ROWS)
        sized = backend._multi_ctas_kv_counter_buffer
        backend._ensure_multi_ctas_kv_counter_capacity(64)
        self.assertIs(backend._multi_ctas_kv_counter_buffer, sized)

    def test_oversized_eager_call_keeps_the_captured_allocation(self):
        """capture -> oversized eager call -> replay: the buffer must survive.

        A prefill batch is indexed by query rows too, so it can exceed even the
        captured capacity. It has to take a temporary rather than replace the
        allocation the graphs recorded.
        """
        backend = _make_backend(max_running_requests=48)
        backend._ensure_multi_ctas_kv_counter_capacity(_CAPTURED_ROWS)
        captured = backend._multi_ctas_kv_counter_buffer
        captured_ptr = captured.data_ptr()

        eager_rows = _CAPTURED_ROWS * 2
        temporary = grow_multi_ctas_kv_counter_buffer_if_needed(
            backend._multi_ctas_kv_counter_buffer,
            torch.device("cuda"),
            backend.num_q_heads,
            eager_rows,
        )

        self.assertIsNot(temporary, captured)
        self.assertIs(backend._multi_ctas_kv_counter_buffer, captured)
        self.assertEqual(backend._multi_ctas_kv_counter_buffer.data_ptr(), captured_ptr)


if __name__ == "__main__":
    unittest.main()
