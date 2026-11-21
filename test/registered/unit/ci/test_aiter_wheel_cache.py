import importlib.util
import re
import subprocess
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

    def _evict(self, docker_status):
        installer = (ROOT / "scripts/ci/amd/amd_ci_install_dependency.sh").read_text()
        function = re.search(
            r"^aiter_evict_cache_entry\(\) \{.*?^\}$", installer, re.M | re.S
        ).group(0)
        script = f"""set -e
docker() {{ return {docker_status}; }}
CI_CONTAINER_NAME=ci AITER_CACHE_ROOT=/cache AITER_CACHE_KEY=key
{function}
aiter_evict_cache_entry
echo "[$AITER_CACHE_KEY]"
"""
        return subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True
        ).stdout

    def test_failed_eviction_disables_the_cache_instead_of_aborting_the_job(self):
        """Under `set -e` a stale entry that cannot be removed must fall back to the source rebuild."""
        self.assertIn("[]", self._evict(docker_status=1))
        self.assertIn("[key]", self._evict(docker_status=0))

    def _cacheable(self, *, environment, packages):
        installer = (ROOT / "scripts/ci/amd/amd_ci_install_dependency.sh").read_text()
        function = re.search(
            r"^aiter_rebuild_is_cacheable\(\) \{.*?^\}$", installer, re.M | re.S
        ).group(0)
        script = f"""
docker() {{
    case "${{@: -1}}" in
        environment) echo "{environment}" ;;
        packages) echo "{packages}" ;;
    esac
}}
CI_CONTAINER_NAME=ci AITER_CACHE_TOOL=tool AITER_ENVIRONMENT=abi AITER_PACKAGES=pkgs
{function}
aiter_rebuild_is_cacheable
"""
        return subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True
        ).returncode

    def test_rebuild_is_published_only_if_nothing_but_aiter_changed(self):
        """A cache hit installs only the wheel, so a rebuild that changed the ABI or pulled packages must not publish."""
        self.assertEqual(self._cacheable(environment="abi", packages="pkgs"), 0)
        self.assertNotEqual(self._cacheable(environment="abi2", packages="pkgs"), 0)
        self.assertNotEqual(
            self._cacheable(environment="abi", packages="pkgs+ninja"), 0
        )

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

    def test_package_snapshot_excludes_only_aiter(self):
        """A rebuild that installs any package but AITER must change the snapshot that gates publication."""

        def distributions(*names):
            return [
                SimpleNamespace(metadata={"Name": name}, version=version)
                for name, version in names
            ]

        before = [("torch", "2"), ("amd-aiter", "1"), ("Requests", "3")]
        after = [("torch", "2"), ("amd-aiter", "2"), ("Requests", "3")]
        added = [*after, ("tqdm", "4")]
        upgraded = [("torch", "2"), ("amd-aiter", "2"), ("Requests", "4")]
        snapshots = []
        for installed in (before, after, added, upgraded):
            with patch.object(
                cache.importlib.metadata,
                "distributions",
                return_value=distributions(*installed),
            ):
                snapshots.append(cache.installed_packages())
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertNotIn("amd-aiter", snapshots[0])
        self.assertNotEqual(snapshots[1], snapshots[2])
        self.assertNotEqual(snapshots[1], snapshots[3])

    def test_entry_is_invisible_until_publication_completes(self):
        """A job that looks up the key while another copies its wheel must see a miss, not a partial entry."""
        with tempfile.TemporaryDirectory() as directory:
            wheel = self._make_wheel(directory)
            root = Path(directory) / "cache"
            visible_while_copying = []
            real_copyfile = cache.shutil.copyfile

            def copyfile(source, destination):
                visible_while_copying.append((root / "key").exists())
                return real_copyfile(source, destination)

            with patch.object(cache.shutil, "copyfile", copyfile):
                cache.publish(wheel=wheel, root=root, key="key")
            self.assertEqual(visible_while_copying, [False])
            self.assertTrue((root / "key").is_dir())

    def test_wheel_without_runtime_assets_is_not_published(self):
        """Wheels without runtime files must be rejected before a cache entry is published."""
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
