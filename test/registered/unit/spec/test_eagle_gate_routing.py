"""Gate-routing tests for EAGLE verify: sampling vs. greedy (argmax).

Guards the correctness contract of ``eagle_utils._verify_uses_greedy``: on CUDA,
CPU and NPU the gate must reduce byte-for-byte to the original predicate
(``is_all_greedy or is_cpu or is_hip or is_xpu``); HIP may take the sampling path
only when rejection sampling is on; XPU only for a top-1 chain (its target-only
sampler) or with rejection sampling. A regression that forced greedy on CUDA, or
let HIP/XPU sample through a path they cannot run, would turn a case here red.
Pure-boolean logic, so it runs on CPU CI.
"""

import itertools
import unittest

from sglang.srt.arg_groups.speculative_hook import (
    _should_auto_enable_hip_rejection_sampling,
)
from sglang.srt.speculative.eagle_utils import _verify_uses_greedy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _pre_patch_gate(is_all_greedy, is_cpu, is_hip, is_xpu):
    # eagle_utils.py gate before this PR. Non-HIP behavior must match this exactly.
    return is_all_greedy or is_cpu or is_hip or is_xpu


# (name, is_cpu, is_npu, is_hip, is_xpu); CUDA == no platform flag set.
_PLATFORMS = {
    "cuda": (False, False, False, False),
    "hip": (False, False, True, False),
    "cpu": (True, False, False, False),
    "npu": (False, True, False, False),
    "xpu": (False, False, False, True),
}


class TestEagleGateRouting(CustomTestCase):
    def _gate(self, is_all_greedy, platform, use_rej, tree_topk=1):
        is_cpu, _, is_hip, is_xpu = _PLATFORMS[platform]
        return _verify_uses_greedy(
            is_all_greedy=is_all_greedy,
            is_cpu=is_cpu,
            is_hip=is_hip,
            is_xpu=is_xpu,
            use_rejection_sampling=use_rej,
            tree_topk=tree_topk,
        )

    def test_non_hip_is_byte_identical_to_pre_patch(self):
        for platform, is_all_greedy, use_rej, tree_topk in itertools.product(
            _PLATFORMS, (False, True), (False, True), (1, 4)
        ):
            is_cpu, _, is_hip, is_xpu = _PLATFORMS[platform]
            if is_hip or is_xpu:
                continue
            got = self._gate(is_all_greedy, platform, use_rej, tree_topk)
            expected = _pre_patch_gate(is_all_greedy, is_cpu, is_hip, is_xpu)
            self.assertEqual(
                got,
                expected,
                f"{platform} greedy={is_all_greedy} rej={use_rej}: gate diverged "
                f"from pre-patch ({got} != {expected})",
            )

    def test_hip_samples_only_with_rejection_and_non_greedy(self):
        # The single new sampling entry: HIP + rejection sampling + not all-greedy.
        self.assertFalse(self._gate(False, "hip", True))
        # Every other HIP combination still commits greedy (argmax).
        self.assertTrue(self._gate(True, "hip", True))
        self.assertTrue(self._gate(True, "hip", False))
        self.assertTrue(self._gate(False, "hip", False))

    def test_xpu_samples_top1_chains_or_with_rejection(self):
        # Target-only Triton chain sampler: top-1 chains sample without rejection sampling.
        self.assertFalse(self._gate(False, "xpu", False, tree_topk=1))
        self.assertFalse(self._gate(False, "xpu", True, tree_topk=1))
        # Tree drafts have no XPU target-only kernel; only rejection sampling samples.
        self.assertTrue(self._gate(False, "xpu", False, tree_topk=4))
        self.assertFalse(self._gate(False, "xpu", True, tree_topk=4))
        for use_rej, tree_topk in itertools.product((False, True), (1, 4)):
            self.assertTrue(self._gate(True, "xpu", use_rej, tree_topk))

    def test_cuda_and_npu_keyed_on_all_greedy_only(self):
        # Both sample whenever the batch isn't all-greedy, regardless of the flag.
        for platform in ("cuda", "npu"):
            self.assertFalse(self._gate(False, platform, False))
            self.assertFalse(self._gate(False, platform, True))
            self.assertTrue(self._gate(True, platform, False))
            self.assertTrue(self._gate(True, platform, True))


def _hip_auto_enable(**overrides):
    kwargs = dict(
        is_hip=True,
        use_rejection_sampling=False,
        algorithm="EAGLE",
        token_map=None,
        eagle_topk=1,
        accept_threshold_single=1.0,
        accept_threshold_acc=1.0,
        enable_deterministic_inference=False,
    )
    kwargs.update(overrides)
    return _should_auto_enable_hip_rejection_sampling(**kwargs)


class TestHipAutoEnableRejectionSampling(CustomTestCase):
    def test_same_vocab_eagle_on_hip(self):
        self.assertTrue(_hip_auto_enable())

    def test_eagle3_stays_off(self):
        # Reduced hot-token vocab; stage-a test_basic_sanity_eagle3.
        self.assertFalse(_hip_auto_enable(algorithm="EAGLE3"))

    def test_token_map_stays_off(self):
        self.assertFalse(_hip_auto_enable(token_map="d2t.pt"))

    def test_cuda_never_flips(self):
        self.assertFalse(_hip_auto_enable(is_hip=False))

    def test_already_on_is_a_no_op(self):
        self.assertFalse(_hip_auto_enable(use_rejection_sampling=True))

    def test_simulate_acc_len_stays_off(self):
        # AgentX throughput overwrites accept after real verify; skip the
        # Triton chain sampler so HIP stays on the greedy argmax path.
        self.assertFalse(_hip_auto_enable(simulate_acc_len=3.39))
        self.assertTrue(_hip_auto_enable(simulate_acc_len=-1.0))


if __name__ == "__main__":
    unittest.main()
