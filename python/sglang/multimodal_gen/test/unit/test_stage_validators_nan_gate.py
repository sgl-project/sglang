"""Unit tests for the StageValidators NaN-scan gate.

SGLANG_DIFFUSION_DISABLE_STAGE_NAN_CHECK skips the torch.isnan() scan (a
full-size kernel plus a host sync) while keeping shape/type checks; the
default (unset) must keep today's behavior.
"""

import unittest

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)


class TestStageValidatorsNanGate(unittest.TestCase):
    def setUp(self):
        self._prior = envs.SGLANG_DIFFUSION_DISABLE_STAGE_NAN_CHECK

    def tearDown(self):
        envs.SGLANG_DIFFUSION_DISABLE_STAGE_NAN_CHECK = self._prior

    def test_default_still_rejects_nan(self):
        envs.SGLANG_DIFFUSION_DISABLE_STAGE_NAN_CHECK = False
        nan_tensor = torch.tensor([1.0, float("nan")])
        self.assertFalse(V.is_tensor(nan_tensor))
        self.assertFalse(V.tensor_with_dims(nan_tensor, 1))
        self.assertFalse(V.list_of_tensors([nan_tensor]))
        self.assertFalse(V.none_or_tensor(nan_tensor))

    def test_disabling_the_scan_accepts_nan_but_keeps_shape_checks(self):
        envs.SGLANG_DIFFUSION_DISABLE_STAGE_NAN_CHECK = True
        nan_tensor = torch.tensor([1.0, float("nan")])
        self.assertTrue(V.is_tensor(nan_tensor))
        self.assertTrue(V.tensor_with_dims(nan_tensor, 1))
        self.assertTrue(V.list_of_tensors([nan_tensor]))
        self.assertTrue(V.none_or_tensor(nan_tensor))
        # Shape/type checks are unaffected by the gate.
        self.assertFalse(V.tensor_with_dims(nan_tensor, 2))
        self.assertFalse(V.is_tensor("not a tensor"))

    def test_clean_tensor_passes_either_way(self):
        clean = torch.ones(3)
        for disabled in (False, True):
            envs.SGLANG_DIFFUSION_DISABLE_STAGE_NAN_CHECK = disabled
            self.assertTrue(V.is_tensor(clean))


if __name__ == "__main__":
    unittest.main()
