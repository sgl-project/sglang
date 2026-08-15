import json
import os
import platform
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_sgl_deepep.sh"


class TestBuildSglDeepEP(CustomTestCase):
    def test_packaging_overlay_does_not_shadow_python_packaging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source = temp_root / "deepep"
            overlay = temp_root / "sgl_deep_ep"
            fake_bin = temp_root / "bin"
            docker_log = temp_root / "docker.jsonl"

            (source / "deep_ep").mkdir(parents=True)
            overlay.mkdir()
            fake_bin.mkdir()
            for path in (source / "setup.py", source / "deep_ep" / "__init__.py"):
                path.touch()
            for filename in ("build_sgl_deep_ep.sh", "setup.py", "VERSION"):
                (overlay / filename).touch()

            fake_docker = fake_bin / "docker"
            fake_docker.write_text("""#!/usr/bin/env python3
import json
import os
import sys

with open(os.environ["DOCKER_LOG"], "a") as output:
    output.write(json.dumps(sys.argv[1:]) + "\\n")
""")
            fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IXUSR)

            architecture = platform.machine()
            if architecture == "arm64":
                architecture = "aarch64"

            env = os.environ.copy()
            env["DOCKER_LOG"] = str(docker_log)
            env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
            subprocess.run(
                [
                    str(BUILD_SCRIPT),
                    "3.11",
                    "13.0",
                    str(source),
                    str(overlay),
                    architecture,
                ],
                check=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            calls = [json.loads(line) for line in docker_log.read_text().splitlines()]
            run_args = next(args for args in calls if args[0] == "run")
            volumes = [
                run_args[index + 1]
                for index, argument in enumerate(run_args[:-1])
                if argument == "--volume"
            ]
            self.assertIn(
                f"{overlay}:/sgl-deep-ep-packaging:ro",
                volumes,
            )
            self.assertIn(
                "bash /sgl-deep-ep-packaging/build_sgl_deep_ep.sh",
                run_args[-1],
            )
            self.assertIn(
                "/deepep /sgl-deep-ep-packaging",
                run_args[-1],
            )


if __name__ == "__main__":
    unittest.main()
