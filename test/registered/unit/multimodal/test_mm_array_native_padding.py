"""array('q') fast path for scheduler-side multimodal padding (CPU)."""

import random
import unittest
from array import array

from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    pad_input_ids_array,
)
from sglang.srt.managers.schedule_batch import (
    MM_PAD_SHIFT_VALUE,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

IM_TOKEN_ID = 900


def _item(modality=Modality.IMAGE, pad_value=MM_PAD_SHIFT_VALUE, offsets=()):
    return MultimodalDataItem(
        modality=modality,
        hash=pad_value - MM_PAD_SHIFT_VALUE,
        pad_value=pad_value,
        offsets=list(offsets),
    )


def _mm(items, **kwargs):
    return MultimodalInputs(mm_items=items, im_token_id=IM_TOKEN_ID, **kwargs)


def _reference(ids_arr, mm):
    # Force the loop implementation by passing a list; array('q') input now
    # dispatches to pad_input_ids_array, which is the path under test.
    return array(
        "q",
        MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            list(ids_arr), mm
        ),
    )


class TestPadInputIdsArray(CustomTestCase):
    def _check(self, ids, mm):
        ids_arr = array("q", ids)
        snapshot = array("q", ids_arr)
        want = _reference(ids_arr, mm)

        got = pad_input_ids_array(ids_arr, mm)

        self.assertIsInstance(got, array)
        self.assertEqual(got.typecode, "q")
        self.assertEqual(got, want)
        # Input must not be mutated.
        self.assertEqual(ids_arr, snapshot)
        return got

    def test_no_mm_items_returns_distinct_copy(self):
        ids = [1, 2, 3, 4]
        got = self._check(ids, _mm([]))
        self.assertEqual(got, array("q", ids))
        self.assertIsNot(got, None)
        ids_arr = array("q", ids)
        self.assertIsNot(pad_input_ids_array(ids_arr, _mm([])), ids_arr)

    def test_overlapping_cross_modality_offsets_match_loop_write_order(self):
        # Pathological: image and audio items with overlapping offsets, with a
        # second image item after the audio one. The loop path groups by
        # modality (all image writes before any audio write); the array path
        # must produce the same last-writer, not the flat item order's.
        ids = list(range(20))
        mm = _mm(
            [
                _item(modality=Modality.IMAGE, pad_value=101, offsets=[(3, 9)]),
                _item(modality=Modality.AUDIO, pad_value=202, offsets=[(6, 12)]),
                _item(modality=Modality.IMAGE, pad_value=103, offsets=[(6, 9)]),
            ],
            audio_token_id=77,
        )
        self._check(ids, mm)

    def test_empty_input_with_items_returns_empty(self):
        got = self._check([], _mm([_item(offsets=[(0, 5)])]))
        self.assertEqual(got, array("q"))

    def test_random_non_overlapping_images(self):
        for seed in range(200):
            rng = random.Random(seed)
            n = rng.randint(1, 5000)
            ids = rng.choices(range(-100, 2000), k=n)

            items = []
            cursor = 0
            for _ in range(rng.randint(0, 6)):
                length = rng.randint(1, 64)
                if cursor + length > n:
                    break
                start = cursor + rng.randint(0, max(0, n - cursor - length))
                items.append(
                    _item(
                        pad_value=rng.randrange(
                            MM_PAD_SHIFT_VALUE, MM_PAD_SHIFT_VALUE + 2**30
                        ),
                        offsets=[(start, start + length - 1)],
                    )
                )
                cursor = start + length

            with self.subTest(seed=seed):
                self._check(ids, _mm(items))

    def test_image_at_offset_zero(self):
        ids = list(range(10))
        item = _item(pad_value=MM_PAD_SHIFT_VALUE + 7, offsets=[(0, 4)])
        got = self._check(ids, _mm([item]))
        self.assertEqual(got[:5], array("q", [MM_PAD_SHIFT_VALUE + 7] * 5))

    def test_image_ending_at_last_token(self):
        ids = list(range(10))
        item = _item(pad_value=MM_PAD_SHIFT_VALUE + 8, offsets=[(5, 9)])
        got = self._check(ids, _mm([item]))
        self.assertEqual(got[5:], array("q", [MM_PAD_SHIFT_VALUE + 8] * 5))

    def test_adjacent_images_distinct_pad_values(self):
        ids = list(range(10))
        pv1, pv2 = MM_PAD_SHIFT_VALUE + 11, MM_PAD_SHIFT_VALUE + 22
        items = [
            _item(pad_value=pv1, offsets=[(2, 4)]),
            _item(pad_value=pv2, offsets=[(5, 7)]),
        ]
        got = self._check(ids, _mm(items))
        self.assertEqual(got[2:5], array("q", [pv1] * 3))
        self.assertEqual(got[5:8], array("q", [pv2] * 3))

    def test_truncated_last_image_clamps_like_reference(self):
        ids = list(range(10))
        item = _item(pad_value=MM_PAD_SHIFT_VALUE + 33, offsets=[(7, 20)])
        got = self._check(ids, _mm([item]))
        self.assertEqual(got[7:], array("q", [MM_PAD_SHIFT_VALUE + 33] * 3))

    def test_item_with_unmapped_modality_is_skipped(self):
        ids = list(range(10))
        audio = _item(
            modality=Modality.AUDIO,
            pad_value=MM_PAD_SHIFT_VALUE + 44,
            offsets=[(2, 5)],
        )
        # audio_token_id left as None -> both implementations skip the item.
        got = self._check(ids, _mm([audio]))
        self.assertEqual(got, array("q", ids))


class TestGeneralPatternPadInputTokensDispatch(CustomTestCase):
    def test_array_input_takes_array_path(self):
        ids = array("q", range(20))
        mm = _mm([_item(pad_value=MM_PAD_SHIFT_VALUE + 55, offsets=[(3, 9)])])
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        got = pattern.pad_input_tokens(ids, mm)
        self.assertIsInstance(got, array)
        self.assertEqual(got, _reference(ids, mm))

    def test_list_input_keeps_list_behavior(self):
        ids = list(range(20))
        mm = _mm([_item(pad_value=MM_PAD_SHIFT_VALUE + 55, offsets=[(3, 9)])])
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        got = pattern.pad_input_tokens(ids, mm)
        self.assertIsInstance(got, list)
        self.assertEqual(array("q", got), _reference(array("q", ids), mm))


if __name__ == "__main__":
    unittest.main()
