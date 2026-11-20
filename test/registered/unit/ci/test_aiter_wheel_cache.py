import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "aiter_wheel_cache", ROOT / "scripts/ci/amd/aiter_wheel_cache.py"
)
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


class TestAiterWheelCache(CustomTestCase):
    def test_host_lock_is_used_only_for_the_actual_container_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            host = Path(directory) / "host"
            other = Path(directory) / "other"
            for shared in [True, False]:
                remote = host if shared else other

                def read_mount(command, **kwargs):
                    path = remote / "aiter-wheels/.locks" / Path(command[-1]).name
                    return SimpleNamespace(
                        returncode=0 if path.exists() else 1,
                        stdout=path.read_text() if path.exists() else "",
                    )

                with patch.object(cache.subprocess, "run", side_effect=read_mount):
                    self.assertEqual(
                        cache.visible_mount(host_root=host, container="ci"), shared
                    )
                self.assertEqual(list((host / "aiter-wheels/.locks").iterdir()), [])

    def _make_wheel(self, directory, complete=True):
        wheel = Path(directory) / "amd_aiter-1-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("aiter/__init__.py", "")
            if complete:
                archive.writestr("aiter_meta/csrc/kernel.cpp", "kernel")
                archive.writestr("aiter_meta/hsa/kernel.co", "binary")
        return wheel

    def test_cache_key_covers_build_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            patch = Path(directory) / "patch"
            patch.write_text("first")
            arguments = dict(
                commit="a" * 40,
                image="sha256:image",
                arch="gfx942",
                namespace="pr",
                inputs=[patch],
                abi="torch-abi",
            )
            original = cache.cache_key(**arguments)
            for name, value in [
                ("commit", "b" * 40),
                ("image", "sha256:new"),
                ("arch", "gfx950"),
                ("namespace", "main"),
                ("abi", "new-abi"),
            ]:
                self.assertNotEqual(
                    original, cache.cache_key(**{**arguments, name: value})
                )
            patch.write_text("second")
            self.assertNotEqual(original, cache.cache_key(**arguments))
            with self.assertRaises(ValueError):
                cache.cache_key(**{**arguments, "commit": "main"})

    def test_atomic_publication_reuse_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = self._make_wheel(directory)
            root = Path(directory) / "cache"
            cache.publish(wheel=wheel, root=root, key="key")
            cache.publish(wheel=wheel, root=root, key="key")
            cached = cache.cached_wheel(directory=root / "key", key="key")
            self.assertEqual(cached.read_bytes(), wheel.read_bytes())
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["key"])
            cached.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                cache.cached_wheel(directory=root / "key", key="key")

    def test_wheel_without_runtime_assets_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            with self.assertRaises(ValueError):
                cache.publish(
                    wheel=self._make_wheel(directory, complete=False),
                    root=root,
                    key="key",
                )
            self.assertFalse(root.exists())

            directories_only = Path(directory) / "amd_aiter-2-py3-none-any.whl"
            with zipfile.ZipFile(directories_only, "w") as archive:
                for name in ("aiter/", "aiter_meta/csrc/", "aiter_meta/hsa/"):
                    archive.writestr(name, "")
                archive.writestr("aiter/__init__.py", "")
            with self.assertRaises(ValueError):
                cache.publish(wheel=directories_only, root=root, key="key")
            self.assertFalse(root.exists())

    def test_jobs_that_run_from_the_aiter_checkout_skip_the_cache(self):
        """A cache hit deletes /sgl-workspace/aiter, so a job that runs from it must opt out."""
        checked = 0
        for workflow in sorted((ROOT / ".github/workflows").glob("*.yml")):
            jobs = yaml.safe_load(workflow.read_text()).get("jobs") or {}
            for name, job in jobs.items():
                runs = [str(step.get("run", "")) for step in job.get("steps", [])]
                if not any("/sgl-workspace/aiter" in run for run in runs):
                    continue
                checked += 1
                for line in "\n".join(runs).splitlines():
                    if "amd_ci_install_dependency.sh" in line:
                        self.assertIn(
                            "--skip-aiter-cache", line, f"{workflow.name}:{name}"
                        )
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
