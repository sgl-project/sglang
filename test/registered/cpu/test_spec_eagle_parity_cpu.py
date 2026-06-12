"""Lossless output parity: spec-decode greedy output == a non-spec reference.

CPU (intel_amx) mirror of test/registered/spec/eagle/test_spec_eagle_parity.py.
The reference is a separate non-spec server, launched and torn down BEFORE the
spec server (sequential -- one model resident at a time; see SpecParityKit).

popen_launch_server auto-appends ``--device cpu`` on a CPU-only host, and the
CPU CI environment provides SGLANG_USE_CPU_ENGINE=1 (inherited by both the
reference and spec server subprocesses), so neither needs to be set here.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.spec_server_kits import SpecParityKit
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_cpu_ci(est_time=900, suite="base-b-test-cpu")


class TestEagle3ParityCPU(SpecParityKit, Eagle3Base):
    """EAGLE3 spec (intel_amx) greedy output == non-spec reference.

    SpecParityKit is first so its setUpClass runs the reference server (and tears
    it down) before the fixture launches the spec server -- sequential, one model
    at a time.
    """

    attention_backend = "intel_amx"
    # The overlap spec path is not supported on CPU; the speculative hook would
    # force this anyway, but keep it explicit.
    disable_overlap = True
    # CPU sizes the static pool from host RAM; 0.3 matches the other CPU server
    # tests (the CUDA preset's 0.75 would grab most of the host's RAM).
    mem_fraction_static = 0.3
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


if __name__ == "__main__":
    unittest.main()
