"""Lifetime of the DSA multi-CTAs KV counter across CUDA graph capture.

The decode graphs record this buffer's device address, so three behaviors must
not come back:

- capture growing the counter, which happens when it is sized from
  ``max_running_requests`` while capture indexes by expanded query rows (target
  verify / draft extend multiply each request by the draft-token count);
- an oversized eager call replacing the captured allocation, freeing the buffer
  the graphs replay against;
- a backend branch that never defines the field, which the sizing hook then
  reads.

Each test drives the production entry point whose regression it guards:
``init_cuda_graph_state`` for sizing and ``_multi_ctas_kv_counter_for`` for the
eager path. The full ``_forward_trtllm`` call around the latter needs a live
FlashInfer kernel and is not covered here.
"""

import unittest
import weakref
from types import SimpleNamespace

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
_MAX_CTX_LEN = 64
# 1024 captured requests x 9 draft tokens: a supported speculative capture whose
# query-row count exceeds TRTLLM_MLA_MAX_BATCH_SIZE.
_CAPTURED_BS = 1024
_DRAFT_TOKENS = 9
_CAPTURED_ROWS = _CAPTURED_BS * _DRAFT_TOKENS


def _make_backend(*, allocate_counter: bool = True):
    """Minimal backend carrying only what the counter paths read."""
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.device = "cuda"
    backend.num_q_heads = _NUM_Q_HEADS
    backend.real_page_size = 64
    backend.hisparse_coordinator = None
    backend.speculative_num_draft_tokens = _DRAFT_TOKENS
    backend.dsa_index_kpool = 1
    backend.use_fused_topk = False
    backend.dsa_topk_backend = SimpleNamespace(should_use_topk_v2=lambda: False)
    backend.dsa_index_topk = 2048
    backend.dsa_decode_impl = "trtllm"
    backend.req_to_token = torch.zeros(
        8, _MAX_CTX_LEN, dtype=torch.int32, device="cuda"
    )
    backend._multi_ctas_kv_counter_buffer = (
        make_persistent_multi_ctas_kv_counter_buffer(
            torch.device("cuda"), _NUM_Q_HEADS, max_batch_size=48
        )
        if allocate_counter
        else None
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
    def test_request_sized_counter_would_grow_at_capture(self):
        """The premise: sizing by requests undercounts captured query rows."""
        self.assertGreater(_CAPTURED_ROWS, TRTLLM_MLA_MAX_BATCH_SIZE)
        self.assertTrue(_would_grow(_make_backend(), _CAPTURED_ROWS))

    def test_init_cuda_graph_state_sizes_for_query_rows(self):
        """Capture must not reallocate after graph state is initialized."""
        backend = _make_backend()
        backend.init_cuda_graph_state(_CAPTURED_BS, _CAPTURED_ROWS)
        self.assertFalse(_would_grow(backend, _CAPTURED_ROWS))

    def test_init_cuda_graph_state_is_grow_only(self):
        """A later, smaller graph must not discard an earlier allocation."""
        backend = _make_backend()
        backend.init_cuda_graph_state(_CAPTURED_BS, _CAPTURED_ROWS)
        sized = backend._multi_ctas_kv_counter_buffer
        backend.init_cuda_graph_state(8, 64)
        self.assertIs(backend._multi_ctas_kv_counter_buffer, sized)

    def test_init_cuda_graph_state_tolerates_backends_without_a_counter(self):
        """Non-TRT-LLM branches leave the counter None; sizing must not read it."""
        backend = _make_backend(allocate_counter=False)
        backend.init_cuda_graph_state(_CAPTURED_BS, _CAPTURED_ROWS)
        self.assertIsNone(backend._multi_ctas_kv_counter_buffer)

    def test_counter_field_defaults_to_none_on_the_class(self):
        """Only one __init__ branch allocates a counter; the rest must still
        leave the attribute readable, since the sizing hook is unconditional."""
        self.assertIsNone(DeepseekSparseAttnBackend._multi_ctas_kv_counter_buffer)

    def test_oversized_eager_call_keeps_the_captured_allocation(self):
        """capture -> oversized eager call -> replay: the buffer must survive.

        Holds only a weakref and an address, so a rebinding implementation drops
        the last strong reference and the assertions see it.
        """
        backend = _make_backend()
        backend.init_cuda_graph_state(_CAPTURED_BS, _CAPTURED_ROWS)
        captured_ref = weakref.ref(backend._multi_ctas_kv_counter_buffer)
        captured_ptr = backend._multi_ctas_kv_counter_buffer.data_ptr()

        counter = backend._multi_ctas_kv_counter_for(_CAPTURED_ROWS * 2)
        self.assertIsNot(counter, backend._multi_ctas_kv_counter_buffer)
        del counter

        self.assertIsNotNone(captured_ref())
        self.assertIs(backend._multi_ctas_kv_counter_buffer, captured_ref())
        self.assertEqual(
            backend._multi_ctas_kv_counter_buffer.data_ptr(), captured_ptr
        )

    def test_within_capacity_eager_call_reuses_the_captured_allocation(self):
        backend = _make_backend()
        backend.init_cuda_graph_state(_CAPTURED_BS, _CAPTURED_ROWS)
        self.assertIs(
            backend._multi_ctas_kv_counter_for(_CAPTURED_ROWS),
            backend._multi_ctas_kv_counter_buffer,
        )


if __name__ == "__main__":
    unittest.main()
