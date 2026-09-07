import unittest

import torch
from dsv41_norm import RMSNorm
from ref_loader import RefTestCase, assert_equal, randomize_, requires_ref


@requires_ref
class TestRMSNorm(RefTestCase):
    def test_matches_reference(self):
        ref = self.model.RMSNorm(256, 1e-20)
        randomize_(ref)
        ours = RMSNorm(256, 1e-20)
        ours.load_state_dict(ref.state_dict())
        x = torch.randn(3, 5, 256) * 4
        assert_equal(ours(x), ref(x))


if __name__ == "__main__":
    unittest.main()
