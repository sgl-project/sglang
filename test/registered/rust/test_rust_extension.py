import importlib.abc
import importlib.machinery
import multiprocessing
import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest import mock

from sglang.srt.rust_extensions import load_rust_extension
from sglang.srt.rust_extensions import loader as rust_extension
from sglang.srt.rust_extensions.torch_build import torch_build_configuration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _hold_filesystem_lock(path: str, ready, release) -> None:
    with rust_extension._filesystem_lock(Path(path)):
        ready.set()
        release.wait(timeout=10)


class _FailingExtensionLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise RuntimeError("broken extension")


class TestRustExtension(CustomTestCase):
    def _workspace(self, root: Path) -> Path:
        workspace = root / "rust"
        crate = workspace / "demo"
        crate.mkdir(parents=True)
        (workspace / "Cargo.toml").write_text(
            '[workspace]\nmembers = ["demo"]\n', encoding="utf-8"
        )
        (workspace / "Cargo.lock").write_text(
            "# generated lockfile\n", encoding="utf-8"
        )
        (crate / "Cargo.toml").write_text(
            """
[package]
name = "demo-extension"
version = "0.1.0"

[package.metadata.sglang]
python-module = "demo._core"
features = ["python"]

[lib]
name = "demo_extension"
crate-type = ["cdylib"]
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (crate / "lib.rs").write_text("fn input() {}\n", encoding="utf-8")
        return workspace

    def test_bundled_wheel_extension_never_touches_source_or_cargo(self):
        bundled = ModuleType("demo._core")
        with (
            mock.patch.object(
                rust_extension.importlib, "import_module", return_value=bundled
            ),
            mock.patch.object(rust_extension, "_discover_crate") as discover,
            mock.patch.object(rust_extension, "_build_context") as fingerprint,
            mock.patch.object(rust_extension, "_cargo_build") as cargo_build,
        ):
            self.assertIs(
                load_rust_extension(
                    "demo._core", mode="auto", workspace=Path("/workspace/not-present")
                ),
                bundled,
            )
        discover.assert_not_called()
        fingerprint.assert_not_called()
        cargo_build.assert_not_called()

    def test_bundled_named_variant_never_touches_source_or_cargo(self):
        bundled = ModuleType("demo._inspection")
        with (
            mock.patch.object(
                rust_extension.importlib, "import_module", return_value=bundled
            ) as import_module,
            mock.patch.object(rust_extension, "_discover_crate") as discover,
            mock.patch.object(rust_extension, "_build_context") as fingerprint,
            mock.patch.object(rust_extension, "_cargo_build") as cargo_build,
        ):
            self.assertIs(
                load_rust_extension(
                    "demo._core",
                    mode="never",
                    workspace=Path("/workspace/not-present"),
                    additional_features=("inspection",),
                    extension_module="demo._inspection",
                ),
                bundled,
            )
        import_module.assert_called_once_with("demo._inspection")
        discover.assert_not_called()
        fingerprint.assert_not_called()
        cargo_build.assert_not_called()

    def test_auto_ignores_a_stale_bundled_extension_in_a_source_tree(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._workspace(root)
            (workspace / "demo/lib.rs").write_text(
                "fn source_changed() {}\n", encoding="utf-8"
            )
            stale = ModuleType("demo._core")
            built = ModuleType("demo._core")
            artifact = root / "libdemo_extension.so"
            artifact.write_bytes(b"fresh extension")
            context = rust_extension._BuildContext(
                "changed-source", "fingerprint", "target"
            )
            with (
                mock.patch.object(
                    rust_extension, "_import_bundled_extension", return_value=stale
                ) as bundled_import,
                mock.patch.object(
                    rust_extension, "_build_context", return_value=context
                ),
                mock.patch.object(
                    rust_extension, "_source_digest", return_value="changed-source"
                ),
                mock.patch.object(
                    rust_extension, "_cargo_build", return_value=artifact
                ) as cargo_build,
                mock.patch.object(
                    rust_extension, "_load_extension_from_path", return_value=built
                ),
            ):
                self.assertIs(
                    load_rust_extension(
                        "demo._core",
                        mode="auto",
                        workspace=workspace,
                        cache_dir=root / "cache",
                    ),
                    built,
                )
            bundled_import.assert_not_called()
            cargo_build.assert_called_once()

    def test_discovery_reads_crate_manifest_metadata(self):
        with TemporaryDirectory() as directory:
            workspace = self._workspace(Path(directory))
            crate = rust_extension._discover_crate(workspace, "demo._core")
            self.assertEqual(crate.package, "demo-extension")
            self.assertEqual(crate.library, "demo_extension")
            self.assertEqual(crate.python_module, "demo._core")
            self.assertEqual(crate.features, ("python",))

            with self.assertRaisesRegex(
                ModuleNotFoundError, r"declared modules: \['demo\._core'\]"
            ):
                rust_extension._discover_crate(workspace, "demo._missing")

    def test_fingerprint_is_content_based_and_covers_build_inputs(self):
        with TemporaryDirectory() as directory:
            workspace = self._workspace(Path(directory))
            crate = rust_extension._discover_crate(workspace, "demo._core")
            with mock.patch.object(
                rust_extension,
                "_command_version",
                side_effect=lambda command, *args, **kwargs: f"{command} 1.0",
            ):
                first = rust_extension._build_context(crate)
                source = workspace / "demo" / "lib.rs"
                os.utime(source, (1, 1))
                self.assertEqual(first, rust_extension._build_context(crate))

                source.write_text("fn changed() {}\n", encoding="utf-8")
                changed_source = rust_extension._build_context(crate)
                self.assertNotEqual(first.fingerprint, changed_source.fingerprint)

                with mock.patch.dict(os.environ, {"RUSTFLAGS": "-Ctarget-cpu=native"}):
                    changed_flags = rust_extension._build_context(crate)
                self.assertNotEqual(
                    changed_source.fingerprint, changed_flags.fingerprint
                )
                self.assertNotEqual(
                    changed_source.target_fingerprint,
                    changed_flags.target_fingerprint,
                )

                inspection = rust_extension._build_context(
                    crate,
                    features=(*crate.features, "inspection"),
                    extension_module="demo._inspection",
                    build_fingerprint={"torch": "2.13"},
                )
                self.assertNotEqual(changed_source.fingerprint, inspection.fingerprint)
                self.assertNotEqual(
                    changed_source.target_fingerprint,
                    inspection.target_fingerprint,
                )

    def test_auto_builds_once_then_uses_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._workspace(root)
            artifact = root / "libdemo_extension.so"
            artifact.write_bytes(b"extension")
            context = rust_extension._BuildContext("source", "fingerprint", "target")
            loaded = ModuleType("demo._core")
            with (
                mock.patch.object(
                    rust_extension, "_import_bundled_extension", return_value=None
                ),
                mock.patch.object(
                    rust_extension, "_build_context", return_value=context
                ),
                mock.patch.object(
                    rust_extension, "_source_digest", return_value="source"
                ),
                mock.patch.object(
                    rust_extension, "_cargo_build", return_value=artifact
                ) as cargo_build,
                mock.patch.object(
                    rust_extension,
                    "_load_extension_from_path",
                    return_value=loaded,
                ),
            ):
                self.assertIs(
                    rust_extension.load_rust_extension(
                        "demo._core",
                        mode="auto",
                        workspace=workspace,
                        cache_dir=root / "cache",
                    ),
                    loaded,
                )
                self.assertIs(
                    rust_extension.load_rust_extension(
                        "demo._core",
                        mode="auto",
                        workspace=workspace,
                        cache_dir=root / "cache",
                    ),
                    loaded,
                )
            cargo_build.assert_called_once()

    def test_never_rejects_missing_cache_without_building(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._workspace(root)
            context = rust_extension._BuildContext("source", "fingerprint", "target")
            with (
                mock.patch.object(
                    rust_extension, "_import_bundled_extension", return_value=None
                ),
                mock.patch.object(
                    rust_extension, "_build_context", return_value=context
                ),
                mock.patch.object(rust_extension, "_cargo_build") as cargo_build,
            ):
                with self.assertRaisesRegex(
                    ModuleNotFoundError, "build mode is 'never'"
                ):
                    rust_extension.load_rust_extension(
                        "demo._core",
                        mode="never",
                        workspace=workspace,
                        cache_dir=root / "cache",
                    )
            cargo_build.assert_not_called()

    def test_force_skips_bundled_import_and_rebuilds_cached_artifact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._workspace(root)
            crate = rust_extension._discover_crate(workspace, "demo._core")
            artifact = root / "libdemo_extension.so"
            artifact.write_bytes(b"new extension")
            context = rust_extension._BuildContext("source", "fingerprint", "target")
            cached = rust_extension._cached_extension_path(
                root / "cache", crate, context.fingerprint
            )
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"old extension")
            with (
                mock.patch.object(
                    rust_extension, "_import_bundled_extension"
                ) as bundled_import,
                mock.patch.object(
                    rust_extension, "_build_context", return_value=context
                ),
                mock.patch.object(
                    rust_extension, "_source_digest", return_value="source"
                ),
                mock.patch.object(
                    rust_extension, "_cargo_build", return_value=artifact
                ) as cargo_build,
                mock.patch.object(
                    rust_extension,
                    "_load_extension_from_path",
                    return_value=ModuleType("demo._core"),
                ),
            ):
                rust_extension.load_rust_extension(
                    "demo._core",
                    mode="force",
                    workspace=workspace,
                    cache_dir=root / "cache",
                )
            bundled_import.assert_not_called()
            cargo_build.assert_called_once()
            self.assertEqual(cached.read_bytes(), b"new extension")

    def test_cargo_build_uses_locked_release_and_declared_features(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._workspace(root)
            crate = rust_extension._discover_crate(workspace, "demo._core")
            target_dir = root / "target"

            def run(command, *, cwd, env, check):
                self.assertTrue(check)
                self.assertEqual(cwd, crate.workspace)
                self.assertEqual(env["PYO3_PYTHON"], sys.executable)
                artifact = Path(env["CARGO_TARGET_DIR"]) / "release"
                artifact.mkdir(parents=True)
                (artifact / "libdemo_extension.so").write_bytes(b"extension")
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(
                rust_extension.subprocess, "run", side_effect=run
            ) as cargo:
                artifact = rust_extension._cargo_build(crate, target_dir)

            self.assertEqual(artifact, target_dir / "release/libdemo_extension.so")
            self.assertEqual(
                cargo.call_args.args[0],
                [
                    "cargo",
                    "build",
                    "--release",
                    "--locked",
                    "--package",
                    "demo-extension",
                    "--features",
                    "python",
                ],
            )

    def test_variant_uses_its_own_module_name_features_and_environment(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._workspace(root)
            artifact = root / "libdemo_extension.so"
            artifact.write_bytes(b"extension")
            context = rust_extension._BuildContext("source", "fingerprint", "target")
            loaded = ModuleType("demo._inspection")
            environment = {"CUSTOM_BUILD_INPUT": "value"}
            with (
                mock.patch.object(
                    rust_extension, "_import_bundled_extension", return_value=None
                ) as bundled_import,
                mock.patch.object(
                    rust_extension, "_build_context", return_value=context
                ) as build_context,
                mock.patch.object(
                    rust_extension, "_source_digest", return_value="source"
                ),
                mock.patch.object(
                    rust_extension, "_cargo_build", return_value=artifact
                ) as cargo_build,
                mock.patch.object(
                    rust_extension,
                    "_load_extension_from_path",
                    return_value=loaded,
                ) as load_from_path,
            ):
                self.assertIs(
                    load_rust_extension(
                        "demo._core",
                        mode="auto",
                        workspace=workspace,
                        cache_dir=root / "cache",
                        additional_features=("inspection",),
                        extension_module="demo._inspection",
                        build_environment=environment,
                        build_fingerprint={"native": "abi"},
                    ),
                    loaded,
                )
            bundled_import.assert_not_called()
            self.assertEqual(
                build_context.call_args.kwargs,
                {
                    "features": ("python", "inspection"),
                    "build_fingerprint": {"native": "abi"},
                    "extension_module": "demo._inspection",
                },
            )
            self.assertEqual(
                cargo_build.call_args.kwargs,
                {
                    "features": ("python", "inspection"),
                    "build_environment": environment,
                },
            )
            self.assertEqual(load_from_path.call_args.args[0], "demo._inspection")

    def test_torch_build_configuration_is_versioned_and_relocatable(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            torch_root = root / "torch"
            (torch_root / "lib").mkdir(parents=True)
            torch_init = torch_root / "__init__.py"
            torch_init.write_text("", encoding="utf-8")
            compat_header = root / "compat.h"
            compat_header.write_text("// compatibility\n", encoding="utf-8")
            fake_torch = SimpleNamespace(
                __version__="2.13.0+cu130",
                __file__=str(torch_init),
                compiled_with_cxx11_abi=lambda: True,
                version=SimpleNamespace(cuda="13.0", hip=None),
            )

            build = torch_build_configuration(
                compat_header=compat_header,
                python_module="sglang.srt.mem_cache.rust_tree_core.mem_cache",
                torch_module=fake_torch,
                base_environment={
                    "PATH": "/usr/bin",
                    "CXXFLAGS": "-O2",
                    "RUSTFLAGS": "-Ctarget-cpu=x86-64",
                    "LIBTORCH_USE_PYTORCH": "1",
                },
            )

            self.assertNotIn("LIBTORCH_USE_PYTORCH", build.environment)
            self.assertEqual(build.environment["LIBTORCH"], str(torch_root))
            self.assertEqual(build.environment["LIBTORCH_INCLUDE"], str(torch_root))
            self.assertEqual(build.environment["LIBTORCH_LIB"], str(torch_root))
            self.assertEqual(build.environment["LIBTORCH_CXX11_ABI"], "1")
            self.assertEqual(build.environment["LIBTORCH_BYPASS_VERSION_CHECK"], "1")
            self.assertIn(str(compat_header), build.environment["CXXFLAGS"])
            self.assertIn(
                "$ORIGIN/../../../../torch/lib", build.environment["RUSTFLAGS"]
            )
            self.assertIn(str(torch_root / "lib"), build.environment["RUSTFLAGS"])
            self.assertEqual(build.fingerprint["torch_version"], "2.13.0+cu130")
            self.assertTrue(build.fingerprint["torch_cxx11_abi"])

            wheel_build = torch_build_configuration(
                compat_header=compat_header,
                python_module="sglang.srt.mem_cache.rust_tree_core.mem_cache",
                torch_module=fake_torch,
                base_environment={},
                include_absolute_rpath=False,
            )
            self.assertNotIn(
                str(torch_root / "lib"), wheel_build.environment["RUSTFLAGS"]
            )
            self.assertFalse(wheel_build.fingerprint["include_absolute_rpath"])

            fake_torch.__version__ = "2.14.0"
            with self.assertRaisesRegex(RuntimeError, "PyTorch 2.11 through 2.13"):
                torch_build_configuration(
                    compat_header=compat_header,
                    python_module="sglang.srt.mem_cache.rust_tree_core.mem_cache",
                    torch_module=fake_torch,
                )

    def test_filesystem_lock_serializes_processes(self):
        with TemporaryDirectory() as directory:
            lock_path = Path(directory) / "build.lock"
            context = multiprocessing.get_context("fork")
            ready = context.Event()
            release = context.Event()
            process = context.Process(
                target=_hold_filesystem_lock,
                args=(os.fspath(lock_path), ready, release),
            )
            process.start()
            self.assertTrue(ready.wait(timeout=5))

            acquired = threading.Event()

            def acquire_in_parent():
                with rust_extension._filesystem_lock(lock_path):
                    acquired.set()

            thread = threading.Thread(target=acquire_in_parent)
            thread.start()
            try:
                time.sleep(0.1)
                self.assertFalse(acquired.is_set())
            finally:
                release.set()
                process.join(timeout=5)
                thread.join(timeout=5)
            self.assertEqual(process.exitcode, 0)
            self.assertTrue(acquired.is_set())

    def test_failed_import_does_not_poison_sys_modules(self):
        module_name = "demo._broken_core"
        module_spec = importlib.machinery.ModuleSpec(
            module_name, _FailingExtensionLoader()
        )
        with mock.patch.object(
            rust_extension.importlib.util,
            "spec_from_file_location",
            return_value=module_spec,
        ):
            with self.assertRaisesRegex(RuntimeError, "broken extension"):
                rust_extension._load_extension_from_path(
                    module_name, Path("/cache/_broken_core.so")
                )
        self.assertNotIn(module_name, sys.modules)

    def test_checked_in_crates_are_discovered_from_wheel_metadata(self):
        for python_module, package, library, features in (
            (
                "sglang.srt.rust_extensions._server",
                "sglang-server",
                "sglang_server",
                (),
            ),
            (
                "sglang.srt.rust_extensions._grpc",
                "sglang-grpc",
                "sglang_grpc_core",
                (),
            ),
            (
                "sglang.srt.rust_extensions._multimodal",
                "sglang-mm",
                "sglang_mm_core",
                ("python", "parallel"),
            ),
            (
                "sglang.srt.mem_cache.rust_tree_core.mem_cache",
                "sglang-radix-tree",
                "mem_cache",
                ("python-extension",),
            ),
        ):
            crate = rust_extension._discover_crate(
                rust_extension._RUST_WORKSPACE, python_module
            )
            self.assertEqual(crate.package, package)
            self.assertEqual(crate.library, library)
            self.assertEqual(crate.features, features)


if __name__ == "__main__":
    unittest.main()
