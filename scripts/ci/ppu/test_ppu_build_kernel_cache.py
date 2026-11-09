import hashlib
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = REPO_ROOT / "scripts/ci/ppu/ppu_build_kernel.sh"
AOT_TREE = "aot-tree-object"
TOOLCHAIN = "2.11.0|13.0|cp312"
SDK_2_2_IMAGE = (
    "pkg.flytiger-eco.com/docker_release/llm:"
    "sdk2.2.0-pytorch2.11.0-ubuntu24.04-cuda13.0-sglang0.5.17-py312-20261001"
)


class KernelCacheHarness:
    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.repo = self.root / "repo"
        self.aot = self.repo / "python/sglang/kernels/aot"
        self.aot.mkdir(parents=True)
        (self.aot / "setup_ppu.py").write_text("# test fixture\n")
        self.cache = self.root / "cache"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self._write_executable(
            "git",
            """#!/bin/sh
case "$*" in
  *HEAD:python/sglang/kernels/aot*) printf '%s\\n' 'aot-tree-object' ;;
  *--short*) printf '%s\\n' 'deadbee' ;;
  *) exit 1 ;;
esac
""",
        )
        self._write_executable(
            "python3",
            """#!/bin/sh
if [ "${1:-}" = "-c" ]; then
  case "${2:-}" in
    *"import sys"*) printf '%s\\n' '2.11.0|13.0|cp312' ;;
    *) exit 0 ;;
  esac
elif [ "${1:-}" = "-m" ]; then
  exit 0
fi
exit 1
""",
        )

    def close(self):
        self._tmp.cleanup()

    def _write_executable(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    @staticmethod
    def cache_key(image_identity, sdk_version):
        material = "|".join((AOT_TREE, TOOLCHAIN, image_identity, sdk_version))
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def run(self, *, image="", digest="", sdk_version="", cached_key=None):
        if cached_key:
            key_dir = self.cache / cached_key
            key_dir.mkdir(parents=True)
            (key_dir / "sglang_kernel-0.0.0-py3-none-any.whl").write_bytes(b"wheel")
        env = os.environ.copy()
        env.update(
            {
                "GITHUB_WORKSPACE": str(self.repo),
                "PATH": f"{self.bin}:{env['PATH']}",
                "SGL_KERNEL_WHEEL_CACHE_DIR": str(self.cache),
                "PPU_BASE_IMAGE": image,
                "PPU_BASE_IMAGE_DIGEST": digest,
                "PPU_SDK_VERSION": sdk_version,
            }
        )
        return subprocess.run(
            ["bash", str(BUILD_SCRIPT)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )


class TestKernelCacheIdentity(unittest.TestCase):
    def setUp(self):
        self.harness = KernelCacheHarness()

    def tearDown(self):
        self.harness.close()

    def assert_cache_hit(self, result, key):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"Cache HIT (key {key})", result.stdout)
        self.assertIn(f"SGL_KERNEL_LOCAL_VERSION: ppu.src.{key}", result.stdout)

    def test_image_tag_changes_cache_key(self):
        for image in ("registry/llm:sdk2.1.1", "registry/llm:sdk2.2.0"):
            with self.subTest(image=image):
                key = self.harness.cache_key(image, "2.2.0")
                result = self.harness.run(
                    image=image, sdk_version="2.2.0", cached_key=key
                )
                self.assert_cache_hit(result, key)

    def test_digest_takes_precedence_over_tag(self):
        digest = "sha256:" + "a" * 64
        key = self.harness.cache_key(digest, "2.2.0")
        result = self.harness.run(
            image="registry/llm:mutable",
            digest=digest,
            sdk_version="2.2.0",
            cached_key=key,
        )
        self.assert_cache_hit(result, key)
        self.assertIn("Cache image identity (digest)", result.stdout)

    def test_sdk_version_changes_cache_key(self):
        image = "registry/llm:stable"
        for sdk_version in ("2.1.1", "2.2.0"):
            with self.subTest(sdk_version=sdk_version):
                key = self.harness.cache_key(image, sdk_version)
                result = self.harness.run(
                    image=image, sdk_version=sdk_version, cached_key=key
                )
                self.assert_cache_hit(result, key)

    def test_missing_image_identity_bypasses_cache(self):
        result = self.harness.run(sdk_version="2.2.0")
        self.assertNotIn("Cache HIT", result.stdout)
        self.assertIn("Wheel cache unavailable (missing image identity)", result.stdout)

    def test_missing_sdk_version_bypasses_cache(self):
        result = self.harness.run(image="registry/llm:sdk2.2.0")
        self.assertNotIn("Cache HIT", result.stdout)
        self.assertIn(
            "Wheel cache unavailable (missing PPU_SDK_VERSION)", result.stdout
        )


class TestWorkflowCacheIdentity(unittest.TestCase):
    WORKFLOW_SDK_VERSIONS = {
        "pr-test-ppu.yml": "2.2.0",
        "nightly-test-ppu.yml": "2.2.0",
        "test-ppu-answer.yml": "2.2.0",
        "test-ppu-answer-16.yml": "2.2.0",
        "test-ppu-answer-32.yml": "2.2.0",
        "test-ppu-accuracy.yml": "2.2.0",
        "test-ppu-perf.yml": "2.2.0",
        "test-ppu-perf-16.yml": "2.2.0",
        "test-ppu-perf-32.yml": "2.2.0",
        "test-ppu-pd-perf-glm52.yml": "2.2.0",
        "test-ppu-pd-perf-qwen35.yml": "2.2.0",
    }

    def test_workflows_declare_expected_image_and_sdk_version(self):
        workflow_dir = REPO_ROOT / ".github/workflows"
        for filename, sdk_version in self.WORKFLOW_SDK_VERSIONS.items():
            with self.subTest(filename=filename):
                text = (workflow_dir / filename).read_text()
                self.assertIn(f"  PPU_BASE_IMAGE: {SDK_2_2_IMAGE}\n", text)
                self.assertIn(f"  PPU_SDK_VERSION: {sdk_version}\n", text)

    def test_k8s_extra_env_forwards_image_and_sdk_identity(self):
        workflow_dir = REPO_ROOT / ".github/workflows"
        for filename in self.WORKFLOW_SDK_VERSIONS:
            text = (workflow_dir / filename).read_text()
            extra_env_lines = [
                line for line in text.splitlines() if "extra_env:" in line
            ]
            self.assertTrue(extra_env_lines, filename)
            for line in extra_env_lines:
                with self.subTest(filename=filename):
                    self.assertIn("PPU_BASE_IMAGE=${{ env.PPU_BASE_IMAGE }}", line)
                    self.assertIn("PPU_SDK_VERSION=${{ env.PPU_SDK_VERSION }}", line)
                    if filename == "nightly-test-ppu.yml":
                        self.assertIn("PIP_INDEX_URL=https://pypi.org/simple", line)
                    else:
                        self.assertIn(
                            "PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/", line
                        )
                        self.assertIn("PIP_TRUSTED_HOST=mirrors.aliyun.com", line)


if __name__ == "__main__":
    unittest.main()
