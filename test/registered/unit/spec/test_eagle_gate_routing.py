"""Gate-routing tests for EAGLE verify: sampling vs. greedy (argmax).

Guards the correctness contract of the ROCm fix in
``eagle_utils._verify_uses_greedy``: on every non-HIP platform the gate must
reduce byte-for-byte to the pre-patch predicate
(``is_all_greedy or is_cpu or is_npu or is_hip or is_xpu``), and HIP may take the
sampling path only when rejection sampling is on and the batch isn't all-greedy.
A regression that forced greedy on CUDA, or that let HIP sample without rejection
sampling, would turn a case here red. Pure-boolean logic, so it runs on CPU CI.
"""

import itertools
import unittest

from sglang.srt.speculative.eagle_utils import _verify_uses_greedy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _pre_patch_gate(is_all_greedy, is_cpu, is_npu, is_hip, is_xpu):
    # eagle_utils.py gate before this PR. Non-HIP behavior must match this exactly.
    return is_all_greedy or is_cpu or is_npu or is_hip or is_xpu


# (name, is_cpu, is_npu, is_hip, is_xpu); CUDA == no platform flag set.
_PLATFORMS = {
    "cuda": (False, False, False, False),
    "hip": (False, False, True, False),
    "cpu": (True, False, False, False),
    "npu": (False, True, False, False),
    "xpu": (False, False, False, True),
}


class TestEagleGateRouting(CustomTestCase):
    def _gate(self, is_all_greedy, platform, use_rej):
        is_cpu, is_npu, is_hip, is_xpu = _PLATFORMS[platform]
        return _verify_uses_greedy(
            is_all_greedy=is_all_greedy,
            is_cpu=is_cpu,
            is_npu=is_npu,
            is_hip=is_hip,
            is_xpu=is_xpu,
            use_rejection_sampling=use_rej,
        )

    def test_non_hip_is_byte_identical_to_pre_patch(self):
        for platform, is_all_greedy, use_rej in itertools.product(
            _PLATFORMS, (False, True), (False, True)
        ):
            is_cpu, is_npu, is_hip, is_xpu = _PLATFORMS[platform]
            if is_hip:
                continue
            got = self._gate(is_all_greedy, platform, use_rej)
            expected = _pre_patch_gate(is_all_greedy, is_cpu, is_npu, is_hip, is_xpu)
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

    def test_cuda_keyed_on_all_greedy_only(self):
        # CUDA samples whenever the batch isn't all-greedy, regardless of the flag.
        self.assertFalse(self._gate(False, "cuda", False))
        self.assertFalse(self._gate(False, "cuda", True))
        self.assertTrue(self._gate(True, "cuda", False))
        self.assertTrue(self._gate(True, "cuda", True))

    def test_cpu_npu_xpu_always_greedy(self):
        for platform in ("cpu", "npu", "xpu"):
            for is_all_greedy in (False, True):
                for use_rej in (False, True):
                    self.assertTrue(
                        self._gate(is_all_greedy, platform, use_rej),
                        f"{platform} must force greedy",
                    )


if __name__ == "__main__":
    unittest.main()
