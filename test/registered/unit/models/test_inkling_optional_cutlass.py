import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestInklingOptionalCutlass(CustomTestCase):
    def test_inkling_import_succeeds_without_cutlass(self):
        """Inkling remains registered when the optional CUTE dependency is absent."""
        script = textwrap.dedent("""
            import importlib.abc
            import sys

            class BlockCutlass(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path, target=None):
                    if fullname == "cutlass" or fullname.startswith("cutlass."):
                        raise ModuleNotFoundError("blocked optional cutlass dependency")
                    return None

            for module_name in tuple(sys.modules):
                if module_name == "cutlass" or module_name.startswith("cutlass."):
                    del sys.modules[module_name]
            sys.meta_path.insert(0, BlockCutlass())

            import sglang.srt.models.inkling
            """)

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
