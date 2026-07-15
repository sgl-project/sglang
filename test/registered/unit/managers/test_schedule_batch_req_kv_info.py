from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import copy
import unittest

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ReqKvInfo  # noqa: E402
from sglang.srt.runtime_context import get_flags  # noqa: E402


class TestReqKvInfoAlignmentGuardrail(CustomTestCase):
    def test_unaligned_values_are_accepted_when_no_page_size_is_published(self):
        """Before pools publish a modulus, the default of 1 leaves every value legal."""
        self.assertEqual(get_flags().kv_bookkeeping_page_size, 1)

        kv = ReqKvInfo(kv_allocated_len=7, swa_evicted_seqlen=3)

        self.assertEqual(kv.kv_allocated_len, 7)
        self.assertEqual(kv.swa_evicted_seqlen, 3)

        kv.kv_allocated_len = 13
        self.assertEqual(kv.kv_allocated_len, 13)

    def test_construction_rejects_unaligned_kv_allocated_len(self):
        """__init__ routes through the setter, so an unaligned constructor arg fails."""
        with get_flags().override(kv_bookkeeping_page_size=4):
            ReqKvInfo(kv_allocated_len=8, swa_evicted_seqlen=0)

            with self.assertRaises(AssertionError):
                ReqKvInfo(kv_allocated_len=6, swa_evicted_seqlen=0)

    def test_construction_rejects_unaligned_swa_evicted_seqlen(self):
        """Both fields are guarded, not just the first one wired up."""
        with get_flags().override(kv_bookkeeping_page_size=4):
            ReqKvInfo(kv_allocated_len=0, swa_evicted_seqlen=8)

            with self.assertRaises(AssertionError):
                ReqKvInfo(kv_allocated_len=0, swa_evicted_seqlen=6)

    def test_assignment_rejects_unaligned_kv_allocated_len(self):
        """Post-construction assignment is checked, not only construction."""
        with get_flags().override(kv_bookkeeping_page_size=4):
            kv = ReqKvInfo(kv_allocated_len=4, swa_evicted_seqlen=0)

            with self.assertRaises(AssertionError):
                kv.kv_allocated_len = 6

            kv.kv_allocated_len = 8
            self.assertEqual(kv.kv_allocated_len, 8)

    def test_assignment_rejects_unaligned_swa_evicted_seqlen(self):
        """The swa_evicted_seqlen setter mirrors the kv_allocated_len setter."""
        with get_flags().override(kv_bookkeeping_page_size=4):
            kv = ReqKvInfo(kv_allocated_len=0, swa_evicted_seqlen=4)

            with self.assertRaises(AssertionError):
                kv.swa_evicted_seqlen = 6

            kv.swa_evicted_seqlen = 8
            self.assertEqual(kv.swa_evicted_seqlen, 8)

    def test_augmented_assignment_goes_through_the_setter(self):
        """`+=` on the field is a get followed by a set, so the guardrail still fires."""
        with get_flags().override(kv_bookkeeping_page_size=4):
            kv = ReqKvInfo(kv_allocated_len=4, swa_evicted_seqlen=0)

            with self.assertRaises(AssertionError):
                kv.kv_allocated_len += 2

            kv.kv_allocated_len += 4
            self.assertEqual(kv.kv_allocated_len, 8)

    def test_fields_are_slotted_properties(self):
        """A slotted dataclass would silently swallow the setter; pin the class shape."""
        kv = ReqKvInfo(kv_allocated_len=0, swa_evicted_seqlen=0)

        self.assertFalse(hasattr(kv, "__dict__"))
        self.assertIsInstance(ReqKvInfo.kv_allocated_len, property)
        self.assertIsInstance(ReqKvInfo.swa_evicted_seqlen, property)

    def test_override_restores_the_previous_page_size(self):
        """The flag leaf participates in the tier's transactional override."""
        with get_flags().override(kv_bookkeeping_page_size=4):
            self.assertEqual(get_flags().kv_bookkeeping_page_size, 4)

        self.assertEqual(get_flags().kv_bookkeeping_page_size, 1)


class TestReqKvInfoCopySemantics(CustomTestCase):
    def test_copy_revalidates_against_the_current_page_size(self):
        """copy.copy must not restore raw slots behind the setter's back."""
        kv = ReqKvInfo(kv_allocated_len=7, swa_evicted_seqlen=3)

        with get_flags().override(kv_bookkeeping_page_size=4):
            with self.assertRaises(AssertionError):
                copy.copy(kv)

    def test_deepcopy_revalidates_against_the_current_page_size(self):
        """__copy__ alone does not cover deepcopy, which falls back to __reduce_ex__."""
        kv = ReqKvInfo(kv_allocated_len=7, swa_evicted_seqlen=3)

        with get_flags().override(kv_bookkeeping_page_size=4):
            with self.assertRaises(AssertionError):
                copy.deepcopy(kv)

    def test_copy_returns_an_equal_but_distinct_object(self):
        """Guarding copies must not break what copying is for."""
        kv = ReqKvInfo(kv_allocated_len=8, swa_evicted_seqlen=4)

        with get_flags().override(kv_bookkeeping_page_size=4):
            duplicate = copy.copy(kv)

        self.assertIsNot(duplicate, kv)
        self.assertEqual(duplicate.kv_allocated_len, 8)
        self.assertEqual(duplicate.swa_evicted_seqlen, 4)

        duplicate.kv_allocated_len = 12
        self.assertEqual(kv.kv_allocated_len, 8)

    def test_no_unguarded_serialization_channel_is_defined(self):
        """Pickle-style hooks would reopen the bypass __copy__ closes; fail if one appears."""
        self.assertIn("__copy__", ReqKvInfo.__dict__)
        self.assertIn("__deepcopy__", ReqKvInfo.__dict__)

        for channel in ("__reduce__", "__reduce_ex__", "__setstate__", "__getstate__"):
            self.assertNotIn(
                channel,
                ReqKvInfo.__dict__,
                f"ReqKvInfo defines {channel}; it must validate through the setter "
                "or the alignment guardrail has a new side door.",
            )


if __name__ == "__main__":
    unittest.main()
