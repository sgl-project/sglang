import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
GENERATOR = ROOT / "scripts/kvbit/generate_dsv4_layout.py"


class TestDSV4INT4ABI(unittest.TestCase):
    def test_generated_constants_are_current(self):
        result = subprocess.run(
            [sys.executable, str(GENERATOR), "--check"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_schema_rejects_unimplemented_codec_changes(self):
        spec = importlib.util.spec_from_file_location(
            "_kvbit_layout_generator", GENERATOR
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        schema = json.loads(module.SCHEMA.read_text())
        self.assertEqual(len(module.render(schema)), 2)
        for change in ({"group_size": 64}, {"transform": "hadamard"}, {"version": 2}):
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "Unsupported codec"),
            ):
                module.render({**schema, **change})

    def test_cpp_header_and_sink_math(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        source = Path(__file__).with_name("kvbit_attention_math_test.cc")
        include = ROOT / "python/sglang/kernels/aot/csrc/kvbit/flashmla"
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory) / "kvbit_attention_math_test")
            build = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Werror",
                    "-I",
                    str(include),
                    str(source),
                    "-o",
                    binary,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            result = subprocess.run(
                [binary], text=True, capture_output=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.strip(), "120 sink/LSE cases passed")


if __name__ == "__main__":
    unittest.main()
