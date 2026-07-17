import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_queue(
    *, page_size: int, num_reserved_decode_tokens: int
) -> DecodePreallocQueue:
    queue = object.__new__(DecodePreallocQueue)
    queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=page_size)
    queue.num_reserved_decode_tokens = num_reserved_decode_tokens
    return queue


def _make_req(*, allocated_len: int = 0) -> SimpleNamespace:
    if allocated_len == 0:
        return SimpleNamespace(kv=None)
    return SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=allocated_len))


class TestAdmissionRequiredTokens(CustomTestCase):
    def test_page_tail_is_not_charged_twice_against_the_reserve(self):
        """The rounded-up tail already serves the first decode steps; charging the full reserve on top permanently rejected a request that fits exactly (capacity 48 = ceil(14) + 34 - 2)."""
        queue = _make_queue(page_size=16, num_reserved_decode_tokens=34)

        required = queue._admission_required_tokens(
            req=_make_req(), fill_len=14, total_prefix_len=0
        )

        self.assertEqual(required, 48)

    def test_a_watermark_past_the_reserve_needs_no_new_tokens(self):
        """A rebootstrapped request whose old allocation already covers fill + reserve must not be charged again."""
        queue = _make_queue(page_size=16, num_reserved_decode_tokens=34)

        required = queue._admission_required_tokens(
            req=_make_req(allocated_len=64), fill_len=14, total_prefix_len=0
        )

        self.assertEqual(required, 0)

    def test_an_aligned_fill_still_pays_the_full_reserve(self):
        """With no tail past fill_len the reserve discount must be zero, or admission would under-budget decode headroom."""
        queue = _make_queue(page_size=16, num_reserved_decode_tokens=34)

        required = queue._admission_required_tokens(
            req=_make_req(), fill_len=32, total_prefix_len=0
        )

        self.assertEqual(required, 66)

    def test_a_radix_prefix_only_charges_the_delta_pages(self):
        """Tokens up to total_prefix_len are cache-owned; only [total_prefix_len, ceil(fill)) plus the uncovered reserve is new."""
        queue = _make_queue(page_size=16, num_reserved_decode_tokens=34)

        required = queue._admission_required_tokens(
            req=_make_req(), fill_len=2064, total_prefix_len=2048
        )

        self.assertEqual(required, 16 + 34)


if __name__ == "__main__":
    unittest.main()
