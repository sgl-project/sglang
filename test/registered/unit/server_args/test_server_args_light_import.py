"""Importing ``sglang.srt.server_args`` must not pull in ``transformers``.

The per-model override modules are imported eagerly by ``server_args``, so one
of them reaching a transformers-backed ``configs/*`` module costs every CLI
start ~1.5s. See https://github.com/sgl-project/sglang/issues/10492.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import subprocess
import sys
import unittest

from sglang.test.test_utils import CustomTestCase


class TestServerArgsLightImport(CustomTestCase):
    def test_server_args_does_not_import_transformers(self):
        # A fresh interpreter: this test process may already have transformers.
        code = (
            "import sys\n"
            "import sglang.srt.server_args\n"
            "sys.exit(1 if 'transformers' in sys.modules else 0)\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True)
        self.assertEqual(
            result.returncode,
            0,
            "importing sglang.srt.server_args loaded transformers; keep "
            "arg_groups/model_overrides free of transformers-backed imports.\n"
            + result.stderr.decode(errors="replace"),
        )

    def test_ernie55_arch_name_matches_config(self):
        from sglang.srt.arg_groups.model_overrides import ernie55 as override
        from sglang.srt.configs import ernie55 as config

        self.assertEqual(override.ERNIE5_5_MOE_ARCH, config.ERNIE5_5_MOE_ARCH)


if __name__ == "__main__":
    unittest.main()
