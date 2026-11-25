"""The verify renorm fuses top-p and top-k only when a top-k bound exists.

The fused kernel keeps just the leading max_keep columns, which is exact only
when max_keep covers every row. Without a bound, when only one filter applies,
or on a platform without the kernel, the two single-filter kernels must run in
the original order: top-p on the original distribution, then top-k.
"""

from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.sampling import renorm as platform
from sglang.srt.layers import sampling_renorm
from sglang.srt.speculative.eagle_utils import _apply_joint_top_k_top_p
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sampling_info(*, top_p=True, top_k=True, max_top_k=20):
    return SimpleNamespace(
        need_top_p_sampling=top_p,
        need_top_k_sampling=top_k,
        max_top_k=max_top_k,
        top_ks=torch.tensor([20, 5]),
        top_ps=torch.tensor([0.95, 0.9]),
    )


def _identity_top_k(probs, top_ks):
    return probs


def _identity_top_p(probs, top_ps):
    return probs


class TestJointRenormDispatch(CustomTestCase):
    def _apply(self, probs, info, top_k=None, top_p=None):
        return _apply_joint_top_k_top_p(
            probs,
            info,
            2,
            top_k or _identity_top_k,
            top_p or _identity_top_p,
        )

    def test_fused_kernel_runs_once_with_bound(self):
        probs = torch.rand(2, 8)
        calls = []

        def fused(probs, top_ks, top_ps, max_keep):
            calls.append((top_ks.tolist(), top_ps.tolist(), max_keep))
            return probs

        with mock.patch.object(platform, "top_k_top_p_renorm_prob", fused):
            out = self._apply(probs, _sampling_info())

        self.assertIs(out, probs)
        # One row per draft token, in request order.
        expected_ps = torch.tensor([0.95, 0.95, 0.9, 0.9]).tolist()
        self.assertEqual(calls, [([20, 20, 5, 5], expected_ps, 20)])

    def test_two_step_when_no_bound(self):
        probs = torch.rand(2, 8)
        order = []

        def top_p(probs, top_ps):
            order.append("top_p")
            return probs

        def top_k(probs, top_ks):
            order.append("top_k")
            return probs

        fused = mock.Mock(return_value=probs)
        with mock.patch.object(platform, "top_k_top_p_renorm_prob", fused):
            out = self._apply(
                probs, _sampling_info(max_top_k=None), top_k=top_k, top_p=top_p
            )

        self.assertIs(out, probs)
        self.assertEqual(order, ["top_p", "top_k"])
        fused.assert_not_called()

    def test_two_step_without_platform_kernel(self):
        # A sgl-kernel-npu wheel that predates the fused operator: the fused
        # branch is taken but the wrapper falls back to the two kernels.
        probs = torch.rand(2, 8)
        order = []

        def top_p(probs, top_ps):
            order.append("top_p")
            return probs

        def top_k(probs, top_ks):
            order.append("top_k")
            return probs

        with mock.patch.object(platform, "top_k_top_p_renorm_prob", None):
            with mock.patch.object(
                sampling_renorm, "_top_p_renorm_prob", side_effect=top_p
            ):
                with mock.patch.object(
                    sampling_renorm, "_top_k_renorm_prob", side_effect=top_k
                ):
                    out = self._apply(probs, _sampling_info())

        self.assertIs(out, probs)
        self.assertEqual(order, ["top_p", "top_k"])

    def test_two_step_when_only_top_k_applies(self):
        probs = torch.rand(2, 8)
        order = []

        def top_p(probs, top_ps):
            order.append("top_p")
            return probs

        def top_k(probs, top_ks):
            order.append("top_k")
            return probs

        fused = mock.Mock(return_value=probs)
        with mock.patch.object(platform, "top_k_top_p_renorm_prob", fused):
            out = self._apply(
                probs, _sampling_info(top_p=False), top_k=top_k, top_p=top_p
            )

        self.assertIs(out, probs)
        self.assertEqual(order, ["top_k"])
        fused.assert_not_called()
