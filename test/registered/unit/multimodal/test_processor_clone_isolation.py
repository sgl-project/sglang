"""A preprocessing worker's processor clone must not alias the original.

The worker pool hands each thread its own ``copy.deepcopy`` of the HF processor.
That isolation is only real if a processor's customizations both survive the
copy and rebind to the clone. An instance-level patch that closes over the
original object copies as itself, so every worker thread calls back into the one
shared object -- exactly what the per-thread clone exists to prevent -- and a
patch installed after ``super().__init__()`` is missing from the clone entirely.

Sarashina2Vision is the processor that patches its image processor this way.
"""

import copy
import unittest

from sglang.srt.multimodal.processors.sarashina2_vision import (
    _install_preprocess_kwarg_filter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _NarrowImageProcessor:
    """Stands in for Sarashina2Vision's remote-code image processor."""

    def __init__(self):
        self.owner = "original"

    def _preprocess(self, images, do_resize=None, do_rescale=None):
        return self.owner


class TestSarashina2PreprocessFilterSurvivesCloning(CustomTestCase):
    def test_unfiltered_preprocess_rejects_what_transformers_forwards(self):
        """Why the filter exists: the raw method cannot take the full kwarg set."""
        with self.assertRaises(TypeError):
            _NarrowImageProcessor()._preprocess(["img"], do_resize=True, do_pad=False)

    def test_filter_applies_to_the_patched_processor(self):
        image_processor = _NarrowImageProcessor()
        _install_preprocess_kwarg_filter(image_processor)

        self.assertEqual(
            image_processor._preprocess(["img"], do_resize=True, do_pad=False),
            "original",
        )

    def test_clone_runs_the_filter_against_itself(self):
        image_processor = _NarrowImageProcessor()
        _install_preprocess_kwarg_filter(image_processor)

        clone = copy.deepcopy(image_processor)
        clone.owner = "clone"

        self.assertIs(clone._preprocess.__self__, clone)
        self.assertEqual(
            clone._preprocess(["img"], do_resize=True, do_pad=False), "clone"
        )
        self.assertEqual(
            image_processor._preprocess(["img"], do_resize=True, do_pad=False),
            "original",
        )


if __name__ == "__main__":
    unittest.main()
