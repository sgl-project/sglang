import unittest

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)


def _old_loop_fixed_point(
    h: torch.Tensor, a21: torch.Tensor, x_mid: torch.Tensor, denoised: torch.Tensor
) -> torch.Tensor:
    """The pre-fix iteration this closed form replaces: anchor converges to the
    same fixed point after ~25 iterations in fp64, regardless of eps1's start."""
    eps1 = torch.zeros_like(x_mid)  # arbitrary start: the recurrence is contractive
    anchor = x_mid
    for _ in range(100):
        anchor = x_mid - h * a21 * eps1
        eps1 = denoised - anchor
    return anchor


class TestLTX2BongmathAnchorRefine(unittest.TestCase):
    def test_closed_form_matches_the_100_iteration_fixed_point(self):
        torch.manual_seed(0)
        h = torch.tensor(0.42, dtype=torch.float64)
        a21 = torch.tensor(0.4506, dtype=torch.float64)  # c = h*a21 ~ 0.189, |c|<1
        x_mid = torch.randn(4, 4, dtype=torch.float64)
        denoised = torch.randn(4, 4, dtype=torch.float64)

        anchor, eps1 = LTX2DenoisingStage._ltx2_bongmath_anchor_refine(
            h, a21, x_mid, denoised
        )

        expected_anchor = _old_loop_fixed_point(h, a21, x_mid, denoised)
        torch.testing.assert_close(anchor, expected_anchor, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(eps1, denoised - anchor, rtol=1e-12, atol=1e-12)

    def test_result_is_independent_of_the_prior_eps1_value(self):
        """The closed form takes no eps1 input, so it must agree with the loop
        started from any prior eps1 -- the fixed point the loop converges to
        does not depend on where it started."""
        h = torch.tensor(0.3, dtype=torch.float64)
        a21 = torch.tensor(0.46, dtype=torch.float64)
        x_mid = torch.full((2, 2), 1.5, dtype=torch.float64)
        denoised = torch.full((2, 2), -0.5, dtype=torch.float64)

        anchor, _ = LTX2DenoisingStage._ltx2_bongmath_anchor_refine(
            h, a21, x_mid, denoised
        )

        for eps1_start in (torch.zeros(2, 2), torch.full((2, 2), 10.0)):
            eps1 = eps1_start.double()
            loop_anchor = x_mid
            for _ in range(100):
                loop_anchor = x_mid - h * a21 * eps1
                eps1 = denoised - loop_anchor
            torch.testing.assert_close(anchor, loop_anchor, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
