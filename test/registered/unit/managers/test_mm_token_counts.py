import unittest
from array import array
from unittest.mock import MagicMock

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _item(modality, *offsets):
    return MultimodalDataItem(modality=modality, offsets=list(offsets) or None)


def _req(rid):
    return Req(rid, "", array("q"), SamplingParams(max_new_tokens=1))


class TestMultimodalTokenCounts(unittest.TestCase):
    def test_request_memoizes_empty_counts(self):
        """All-zero counts must be cached too; a zero result is not a cache miss."""
        req = _req("empty")
        req.multimodal_inputs = mm = MagicMock()
        mm.compute_mm_token_counts.return_value = (0, 0, 0)

        self.assertEqual(req.get_mm_token_counts(), (0, 0, 0))
        self.assertEqual(req.get_mm_token_counts(), (0, 0, 0))
        mm.compute_mm_token_counts.assert_called_once_with()

    def test_session_request_inherits_and_extends_counts(self):
        """A session turn reuses the prior turn's counts and adds only new media."""
        parent = _req("parent")
        audio = _item(Modality.AUDIO, (0, 3))
        parent.multimodal_inputs = history = MultimodalInputs(mm_items=[audio])

        child = _req("child")
        child.session = object()
        child.origin_input_ids = array("q", range(8))
        child.multimodal_inputs = history
        child.inherit_mm_token_counts(parent)
        child.extend_image_inputs(
            MultimodalInputs(mm_items=[_item(Modality.IMAGE, (4, 7))])
        )

        self.assertEqual(parent.get_mm_token_counts(), (0, 4, 0))
        self.assertEqual(child.get_mm_token_counts(), (4, 4, 0))
        self.assertIsNot(child.multimodal_inputs, history)


if __name__ == "__main__":
    unittest.main()
