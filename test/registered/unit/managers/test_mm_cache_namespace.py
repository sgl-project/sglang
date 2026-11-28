import hashlib
import unittest

from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.managers.schedule_batch import media_spans_carry_identity
from sglang.srt.multimodal.modality import Modality
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _item(name: str, offsets):
    item = MultimodalDataItem(modality=Modality.IMAGE, offsets=offsets)
    item.set_identity(hashlib.sha256(name.encode()).digest())
    return item


class MediaSpansCarryIdentityTest(CustomTestCase):
    def test_long_span_carries_identity(self):
        self.assertTrue(media_spans_carry_identity([_item("a", [(100, 726)])], 800))

    def test_single_token_span_needs_namespace(self):
        self.assertFalse(media_spans_carry_identity([_item("a", [(100, 100)])], 800))

    def test_span_shorter_than_pad_values_needs_namespace(self):
        item = _item("a", [(100, 100 + len(_item("a", None).pad_values) - 2)])
        self.assertFalse(media_spans_carry_identity([item], 800))

    def test_span_exactly_pad_values_long_carries_identity(self):
        n = len(_item("a", None).pad_values)
        self.assertTrue(media_spans_carry_identity([_item("a", [(10, 10 + n - 1)])], 800))

    def test_any_short_span_needs_namespace(self):
        items = [_item("a", [(10, 600)]), _item("b", [(700, 700)])]
        self.assertFalse(media_spans_carry_identity(items, 800))

    def test_truncated_span_is_measured_inside_prompt(self):
        self.assertFalse(media_spans_carry_identity([_item("a", [(798, 900)])], 800))

    def test_missing_offsets_need_namespace(self):
        self.assertFalse(media_spans_carry_identity([_item("a", None)], 800))

    def test_different_media_spell_different_keys(self):
        a, b = _item("a", [(0, 7)]), _item("b", [(0, 7)])
        self.assertNotEqual(a.padding_sequence(8), b.padding_sequence(8))
        self.assertEqual(a.padding_sequence(8), _item("a", [(0, 7)]).padding_sequence(8))


if __name__ == "__main__":
    unittest.main()
