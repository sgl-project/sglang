import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.tune.candidates import candidate_backends
from sglang.tune.device import DeviceInfo
from sglang.tune.shapes import AttnProfile

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MHA = AttnProfile(32, 8, 128, "bfloat16")
MLA = AttnProfile(128, 128, 576, "bfloat16", is_mla=True)


class TestCandidates(unittest.TestCase):
    def test_fa3_hopper_only(self):
        hop, _ = candidate_backends(DeviceInfo("H100", 90), MHA, "decode")
        self.assertIn("fa3", hop)
        amp_elig, amp_pruned = candidate_backends(DeviceInfo("A100", 80), MHA, "decode")
        self.assertNotIn("fa3", amp_elig)
        self.assertIn("fa3", amp_pruned)  # pruned WITH a reason

    def test_blackwell_cutlass_mla(self):
        bw, _ = candidate_backends(DeviceInfo("B200", 100), MLA, "decode")
        self.assertIn("cutlass_mla", bw)
        hop, pruned = candidate_backends(DeviceInfo("H100", 90), MLA, "decode")
        self.assertNotIn("cutlass_mla", hop)
        self.assertIn("cutlass_mla", pruned)

    def test_mla_backends_excluded_for_mha(self):
        # MLA-only kernels are simply not in the MHA candidate pool.
        elig, _ = candidate_backends(DeviceInfo("H100", 90), MHA, "decode")
        self.assertNotIn("flashmla", elig)
        self.assertNotIn("cutlass_mla", elig)
        # ...and are eligible for an MLA profile on the right arch
        mla_elig, _ = candidate_backends(DeviceInfo("H100", 90), MLA, "decode")
        self.assertIn("flashmla", mla_elig)

    def test_always_fail_open_to_triton(self):
        # Even an exotic SM with nothing else eligible must yield a runnable candidate.
        elig, _ = candidate_backends(DeviceInfo("weird", 70), MHA, "decode")
        self.assertTrue(elig)  # never empty

    def test_trtllm_mha_phase_specific(self):
        # prefill requires sm100; decode allows sm90/100/120
        _, pruned_prefill = candidate_backends(DeviceInfo("H100", 90), MHA, "prefill")
        self.assertIn("trtllm_mha", pruned_prefill)
        elig_decode, _ = candidate_backends(DeviceInfo("H100", 90), MHA, "decode")
        self.assertIn("trtllm_mha", elig_decode)


if __name__ == "__main__":
    unittest.main()
