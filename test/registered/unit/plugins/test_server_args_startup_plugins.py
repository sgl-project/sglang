"""Unit tests for server_args startup plugin import ordering."""

import os
import subprocess
import sys
import tempfile
import textwrap

try:
    import torch

    try:
        torch.xpu.get_device_name = lambda device=None: ""
    except Exception:
        pass
except Exception:
    pass

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestServerArgsStartupPlugins(CustomTestCase):
    def test_server_args_import_loads_startup_plugins_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "fake_startup_plugin.py"), "w") as f:
                f.write(textwrap.dedent("""
                        def startup():
                            import os

                            os.environ["SGLANG_TEST_STARTUP_PLUGIN"] = "loaded"
                            try:
                                import torch

                                if hasattr(torch, "xpu"):
                                    torch.xpu.get_device_name = lambda *a, **kw: ""
                            except ImportError:
                                pass
                        """))

            dist_info = os.path.join(tmpdir, "fake_startup_plugin-0.0.dist-info")
            os.makedirs(dist_info)
            with open(os.path.join(dist_info, "METADATA"), "w") as f:
                f.write("Name: fake-startup-plugin\nVersion: 0.0\n")
            with open(os.path.join(dist_info, "entry_points.txt"), "w") as f:
                f.write(textwrap.dedent("""
                        [sglang.srt.startup_plugins]
                        fake_startup = fake_startup_plugin:startup
                        """))

            env = os.environ.copy()
            pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(p for p in [tmpdir, pythonpath] if p)
            env["SGLANG_PLUGINS"] = "fake_startup"

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent("""
                        import os
                        from sglang.srt.server_args import ServerArgs
                        assert ServerArgs is not None
                        print(os.environ.get("SGLANG_TEST_STARTUP_PLUGIN"))
                        """),
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            self.assertIn("loaded", result.stdout)


if __name__ == "__main__":
    import unittest

    unittest.main()
