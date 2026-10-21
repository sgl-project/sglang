"""PPU nightly serving performance entry for Kimi-K2.6 (ZW-M890P, eight devices).

  configs/kimi2.6/mxfp4-fp8-144g-prefill.json   the only config

Earlier revisions also carried a W4A8-INT8 config, ported from the plan's INT4
cases.  It is removed rather than kept red: the format's W-INT4 kernels route
through ACEXT, which the ZW-M890P stack does not support, so the entry could
not serve on this board line at all.

The config carries two measurements, a 4k-token and a 64k-token prefill of ten
prompts at concurrency one, which share one launch with the KV cache flushed
between them.

The workflow names the config in `SGLANG_PPU_PERF_TEST_CONFIG` and the class
falls back to the default when it is unset.

Nothing here judges a number.  The suite is red only when a measurement produced
no numbers at all, never for being slow.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.perf_suite_kit import PerfSuiteMixin

DATA_ROOT = Path(__file__).parent

# A suite of its own, not a shared one: a suite is what `--suite` selects, and
# every file in one runs in the same job off the same config variable, so two
# models under one name could not be given different weights.  The estimate
# covers a cold load of the larger of the two checkpoints plus both prefill
# sweeps, and the first measured run replaces it.
register_ppu_ci(est_time=5400, suite="nightly-perf-8-kimi26-ppu", nightly=True)


class TestPPUKimiK26Perf(PerfSuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "kimi2.6" / "mxfp4-fp8-144g-prefill.json"
    )


if __name__ == "__main__":
    unittest.main()
