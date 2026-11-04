"""The runtime reserve covers what graph capture cannot measure, and the decode
graph running cap backs off only when that reserve would crowd the KV pool."""

import unittest

from sglang.srt.mem_cache.runtime_reserve import (
    graph_bounded_running_cap,
    transient_reserve,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_GIB = 1 << 30
_QWEN3_VOCAB = 151936
_QWEN3_CONTEXT = 40960
# The decode buckets the planner captures up to 512.
_CAPTURE_BS = (
    [1, 2, 4, 8, 12] + list(range(16, 257, 8)) + list(range(272, 512, 16)) + [512]
)


def _cap(kv_budget_gib, *, vocab=_QWEN3_VOCAB):
    return graph_bounded_running_cap(
        ceiling=512,
        capture_bs=_CAPTURE_BS,
        kv_budget_bytes=int(kv_budget_gib * _GIB),
        vocab_size=vocab,
        context_len=_QWEN3_CONTEXT,
        page_size=1,
    )


class TestGraphBoundedRunningCap(CustomTestCase):
    def test_qwen3_pools_keep_the_ceiling_on_h100_and_back_off_on_l40s(self):
        """KV budgets left after Qwen3 weights, net of the post-capture floor. H100
        pools keep all 512 graphs; the L40S 30B-A3B pool backs off, yet stays above
        the 128 requests it can hold at 1k-token contexts."""
        self.assertEqual(_cap(61.66), 512)  # H100, Qwen3-8B
        self.assertEqual(_cap(47.77), 512)  # H100, Qwen3-30B-A3B-FP8
        l40s_moe = _cap(13.14)
        self.assertLess(l40s_moe, 512)
        self.assertGreaterEqual(l40s_moe, 128)

    def test_cap_is_a_captured_size_and_moves_with_the_budget(self):
        """A captured size needs no extra graph at the request-pool size."""
        caps = [_cap(gib) for gib in (0.001, 1, 4, 8, 16, 32, 64, 1024)]
        self.assertTrue(all(cap in _CAPTURE_BS for cap in caps), caps)
        self.assertEqual(caps, sorted(caps))
        # A tiny pool still serves one request; an ample one keeps the ceiling.
        self.assertEqual((caps[0], caps[-1]), (1, 512))
        # A larger vocabulary means a larger per-request reserve, never a larger cap.
        self.assertLessEqual(_cap(16, vocab=4 * _QWEN3_VOCAB), _cap(16))


class TestTransientReserve(CustomTestCase):
    def _reserve(self, *, requests=512, chunk=8192, deterministic=False, graph_gib=0):
        return transient_reserve(
            vocab_size=_QWEN3_VOCAB,
            max_running_requests=requests,
            chunked_prefill_size=chunk,
            enable_deterministic_inference=deterministic,
            prefill_graph_bytes=int(graph_gib * _GIB),
        )

    def test_both_tails_stay_reserved_beside_the_sampling_state(self):
        """512 penalized decodes beside input-logprob prefill chunks made the caching
        allocator release its cache and retry: blocks freed by one tail were split
        for the other's tensors. The sampling state, both tails and a stranded block
        must all fit at once."""
        slack = self._reserve(requests=0, chunk=0).total_bytes
        reserve = self._reserve(requests=512, chunk=8192)
        self.assertGreaterEqual(
            reserve.total_bytes,
            slack
            + reserve.persistent_bytes
            + reserve.decode_bytes
            + reserve.prefill_bytes
            + reserve.fragmentation_bytes,
        )

    def test_a_logprob_chunk_still_fits_beside_a_stranded_one(self):
        """An input-logprob prefill under a full penalized decode batch ran out of
        memory with a whole chunk's worth reserved-but-unallocated: freed chunk
        blocks were split by smaller per-step tensors. One more chunk must fit."""
        reserve = self._reserve(requests=208, chunk=4096)
        chunk_block = 4 * _QWEN3_VOCAB * 2048
        self.assertGreaterEqual(reserve.fragmentation_bytes, chunk_block)
        self.assertGreaterEqual(
            reserve.total_bytes - reserve.persistent_bytes,
            reserve.prefill_bytes + chunk_block,
        )

    def test_prefill_tail_covers_an_eager_chunk_and_the_slow_logprob_path(self):
        """The measured prefill graph pool bounds an eager forward of one chunk, and
        deterministic inference's full log-softmax holds more than the fast path."""
        self.assertEqual(self._reserve(graph_gib=64).prefill_bytes, 64 * _GIB)
        self.assertGreater(
            self._reserve(deterministic=True).prefill_bytes,
            self._reserve(deterministic=False).prefill_bytes,
        )


if __name__ == "__main__":
    unittest.main()
