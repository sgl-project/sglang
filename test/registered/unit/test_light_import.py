import json
import os
import pathlib
import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PYTHON_ROOT = _REPO_ROOT / "python"
_HEAVY_IMPORTS = (
    "numpy",
    "torch",
    "tqdm",
    "sglang.utils",
    "sglang.srt.utils",
    "sglang.srt.utils.common",
    "sglang.lang.backend.runtime_endpoint",
)


class TestLightImport(unittest.TestCase):
    def test_import_sglang_does_not_load_runtime_dependencies(self):
        script = textwrap.dedent(f"""
            import json
            import sys

            import sglang  # noqa: F401

            loaded_modules = {{
                name: name in sys.modules
                for name in {_HEAVY_IMPORTS!r}
            }}
            print(json.dumps(loaded_modules))
            """)
        env = os.environ.copy()
        pythonpath = str(_PYTHON_ROOT)
        if env.get("PYTHONPATH"):
            pythonpath += os.pathsep + env["PYTHONPATH"]
        env["PYTHONPATH"] = pythonpath

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=_REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        loaded_modules = json.loads(result.stdout)
        self.assertFalse(
            any(loaded_modules.values()),
            msg=f"`import sglang` loaded unexpected modules: {loaded_modules}",
        )


if __name__ == "__main__":
    unittest.main()
