import unittest

import sgl_kernel  # noqa: F401
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="extra-test-cpu-intel")

torch.manual_seed(1234)


class TestRouterLogitsPrecision(CustomTestCase):
    """Document the precision of the CPU AMX router-logit path in NemotronHMoE.

    torch.ops.sgl_kernel.weight_packed_linear accumulates in fp32 but stores
    BF16 output for BF16 input, so ``.float()`` afterwards cannot recover the
    fp32 logits. Experts whose fp32 logits differ by less than one BF16 ulp
    collapse to a tie, and the top-k selection may differ from an fp32 router.
    """

    def _make_near_tie(self, hidden=256, num_experts=16, topk=6):
        x = torch.randn(1, hidden, dtype=torch.bfloat16)
        w = torch.randn(num_experts, hidden, dtype=torch.bfloat16) * 0.05
        # Make expert `topk` a near-copy of expert `topk - 1` so the k-th and
        # (k+1)-th logits are separated by much less than one BF16 ulp.
        ref = torch.mm(x.float(), w.float().t())[0]
        order = ref.argsort(descending=True)
        a, b = order[topk - 1].item(), order[topk].item()
        w[b] = w[a]
        j = x[0].float().abs().argmax().item()
        delta = torch.zeros(hidden, dtype=torch.bfloat16)
        delta[j] = torch.finfo(torch.bfloat16).eps * w[a, j].abs().clamp_min(1e-3)
        w[b] = (w[b].float() - delta.float()).bfloat16()
        return x, w, topk, a, b

    def test_bf16_output_and_near_tie_topk(self):
        x, w, topk, a, b = self._make_near_tie()
        packed_w = torch.ops.sgl_kernel.convert_weight_packed(w)

        out = torch.ops.sgl_kernel.weight_packed_linear(x, packed_w, None, True)
        # The kernel output dtype follows the input: BF16, not fp32.
        self.assertEqual(out.dtype, torch.bfloat16)

        cpu_logits = out.float()[0]
        fp32_logits = torch.mm(x.float(), w.float().t())[0]

        # Error is bounded by BF16 rounding of the logits (~1 ulp + accum).
        ulp = torch.finfo(torch.bfloat16).eps
        bound = 2 * ulp * fp32_logits.abs() + 1e-3
        self.assertTrue(torch.all((cpu_logits - fp32_logits).abs() <= bound))

        # The near-tie pair is resolved in fp32 but indistinguishable (or
        # within one ulp) after BF16 output rounding.
        fp32_gap = (fp32_logits[a] - fp32_logits[b]).abs().item()
        bf16_gap = (cpu_logits[a] - cpu_logits[b]).abs().item()
        self.assertGreater(fp32_gap, 0.0)
        self.assertLessEqual(bf16_gap, ulp * cpu_logits[a].abs().item() + 1e-6)

        # Outside the near-tie pair the selected expert set must agree; only
        # membership of the tied boundary experts {a, b} may differ.
        cpu_topk = set(torch.sigmoid(cpu_logits).topk(topk).indices.tolist())
        ref_topk = set(torch.sigmoid(fp32_logits).topk(topk).indices.tolist())
        self.assertEqual(cpu_topk - {a, b}, ref_topk - {a, b})
        self.assertLessEqual(len(cpu_topk ^ ref_topk), 2)


if __name__ == "__main__":
    unittest.main()
