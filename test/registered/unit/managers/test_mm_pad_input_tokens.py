"""Unit tests for padding multimodal token spans with wide pad sentinels."""

import unittest
from array import array

from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

IMAGE_TOKEN_ID = 9


class TestMultimodalTokensPadding(CustomTestCase):
    def test_truncated_prompt_pads_only_the_kept_part_of_each_span(self):
        # 2 text tokens, image A at [2, 7], image B at [8, 11], 1 text token;
        # auto-truncation cut the prompt at 5 tokens, inside A and before B.
        full = array("q", [1, 2] + [IMAGE_TOKEN_ID] * 10 + [3])
        cut = full[:5]
        a = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(2, 7)])
        a.set_hash(0xAAAA)
        b = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(8, 11)])
        b.set_hash(0xBBBB)
        mm_inputs = MultimodalInputs(mm_items=[a, b], im_token_id=IMAGE_TOKEN_ID)
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()

        padded = pattern.pad_input_tokens(cut, mm_inputs)

        # The kept prefix matches the untruncated padding, so the radix key
        # of the kept tokens does not depend on where the prompt was cut.
        self.assertEqual(
            list(padded), list(pattern.pad_input_tokens(full, mm_inputs)[:5])
        )
        self.assertEqual(list(padded[2:]), a.padding_sequence(3))
        self.assertEqual(list(cut), [1, 2] + [IMAGE_TOKEN_ID] * 3)


if __name__ == "__main__":
    unittest.main()
