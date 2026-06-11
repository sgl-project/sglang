import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.spec_server_kits import SpecParityKit
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_cpu_ci(est_time=900, suite="base-b-test-cpu")


class TestEagle3ParityCPU(SpecParityKit, Eagle3Base):
    """EAGLE3 spec (intel_amx) greedy output == non-spec reference.
    """

    attention_backend = "intel_amx"
    disable_overlap = True
    mem_fraction_static = 0.3
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


if __name__ == "__main__":
    unittest.main()
