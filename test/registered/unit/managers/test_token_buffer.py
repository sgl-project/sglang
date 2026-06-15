import unittest
from array import array

from sglang.srt.managers.token_buffer import TokenBuffer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-b-test-cpu")


class TestTokenBuffer(CustomTestCase):
    def test_init_from_list_array_and_empty_match_length_and_content(self):
        """Construction from list/array/None yields the expected length and values."""
        self.assertEqual(len(TokenBuffer()), 0)
        self.assertEqual(len(TokenBuffer([])), 0)
        buf = TokenBuffer([1, 2, 3])
        self.assertEqual(list(buf.readonly_view()), [1, 2, 3])
        buf_from_array = TokenBuffer(array("q", [4, 5]))
        self.assertEqual(list(buf_from_array.readonly_view()), [4, 5])

    def test_append_grows_past_initial_capacity_and_preserves_order(self):
        """Appending many tokens past capacity keeps all values in order."""
        buf = TokenBuffer([0])
        expected = [0] + list(range(1, 500))
        for token_id in range(1, 500):
            buf.append(token_id)
        self.assertEqual(len(buf), 500)
        self.assertEqual(list(buf.readonly_view()), expected)

    def test_extend_with_list_and_array_appends_in_bulk(self):
        """Extending with a list and an array appends every value contiguously."""
        buf = TokenBuffer([1])
        buf.extend([2, 3])
        buf.extend(array("q", [4, 5]))
        buf.extend([])
        self.assertEqual(list(buf.readonly_view()), [1, 2, 3, 4, 5])

    def test_truncate_lowers_length_without_touching_prefix(self):
        """Truncating drops the tail while keeping the retained prefix intact."""
        buf = TokenBuffer([1, 2, 3, 4, 5])
        buf.truncate(2)
        self.assertEqual(list(buf.readonly_view()), [1, 2])
        buf.append(9)
        self.assertEqual(list(buf.readonly_view()), [1, 2, 9])

    def test_readonly_view_bounds_default_to_full_logical_range(self):
        """readonly_view(None, None) covers the full logical buffer, never spare capacity."""
        buf = TokenBuffer([10, 20, 30, 40])
        self.assertEqual(list(buf.readonly_view(None, 2)), [10, 20])
        self.assertEqual(list(buf.readonly_view(2, None)), [30, 40])
        self.assertEqual(len(buf.readonly_view()), 4)

    def test_readonly_view_is_readonly(self):
        """A handed-out view cannot be written through."""
        buf = TokenBuffer([1, 2, 3])
        view = buf.readonly_view()
        with self.assertRaises(TypeError):
            view[0] = 99

    def test_append_does_not_raise_while_a_view_is_alive(self):
        """Appending is safe even while a prior readonly view is still held (no BufferError)."""
        buf = TokenBuffer([1, 2])
        held = buf.readonly_view()
        for token_id in range(100):
            buf.append(token_id)
        self.assertEqual(len(buf), 102)
        self.assertEqual(list(held), [1, 2])

    def test_materialize_returns_detached_array_copy(self):
        """materialize returns an array snapshot unaffected by later appends."""
        buf = TokenBuffer([1, 2, 3])
        snapshot = buf.materialize(None, 2)
        self.assertIsInstance(snapshot, array)
        self.assertEqual(list(snapshot), [1, 2])
        buf.append(4)
        self.assertEqual(list(snapshot), [1, 2])


if __name__ == "__main__":
    unittest.main()
