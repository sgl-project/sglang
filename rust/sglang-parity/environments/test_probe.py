"""CPU-only tests for environment identity and probe failure boundaries."""

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

spec = importlib.util.spec_from_file_location(
    "probe", Path(__file__).with_name("probe.py")
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def test_library_paths_use_interpreter_sites_without_importing_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            purelib, platlib = root / "purelib", root / "platlib"
            libraries = [
                purelib / "torch/lib",
                purelib / "nvidia/cublas/lib",
                platlib / "nvidia/nested/cudnn/lib",
            ]
            for path in libraries:
                path.mkdir(parents=True)
            (platlib / "torch").mkdir()
            (platlib / "torch/lib").symlink_to(libraries[0])
            (purelib / "nvidia/lib").touch()  # A file named lib is not a search path.
            output = root / "paths.json"
            with (
                patch.object(
                    probe.sysconfig,
                    "get_path",
                    side_effect={
                        "purelib": str(purelib),
                        "platlib": str(platlib),
                    }.__getitem__,
                ),
                patch.object(probe, "verify_python") as verify_python,
                patch.object(probe, "verify_packages") as verify_packages,
                patch.object(probe.importlib, "import_module") as imports,
            ):
                self.assertEqual(probe.main(["--library-paths", str(output)]), 0)
                self.assertEqual(
                    json.loads(output.read_text()),
                    sorted(str(path.resolve()) for path in libraries),
                )
                verify_python.assert_not_called()
                verify_packages.assert_not_called()
                imports.assert_not_called()
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    probe.main(["--library-paths", str(output), "--backend", "cuda"])

    def test_python_version_and_selected_environment(self):
        with patch.object(sys, "version_info", (3, 12, 8)):
            probe.verify_python(Path(sys.executable), "3.12.8")
            with self.assertRaisesRegex(RuntimeError, "Python executable"):
                probe.verify_python(Path("/another/environment/bin/python"), "3.12.8")
        with patch.object(sys, "version_info", (3, 12, 9)):
            with self.assertRaisesRegex(RuntimeError, "Python 3.12.8 is required"):
                probe.verify_python(Path(sys.executable), "3.12.8")
            probe.verify_python(Path(sys.executable), "3.12.9")

    def test_lock_versions_are_required_and_extra_packages_are_reported(self):
        def distribution(name, version):
            return SimpleNamespace(metadata={"Name": name}, version=version)

        with patch.object(probe, "locked_packages", return_value={"torch": "2.8.0"}):
            for installed, error in [
                ([], "torch==2.8.0 is missing"),
                ([distribution("torch", "2.7.0")], "installed 2.7.0"),
            ]:
                with self.subTest(error=error):
                    with patch.object(
                        probe.importlib.metadata,
                        "distributions",
                        return_value=installed,
                    ):
                        with self.assertRaisesRegex(RuntimeError, error):
                            probe.verify_packages(Path("unused.lock"))
            installed = [
                distribution("torch", "2.8.0"),
                distribution("extra_package", "1.0"),
                distribution("sglang", "0.0.0"),
            ]
            with patch.object(
                probe.importlib.metadata, "distributions", return_value=installed
            ):
                self.assertEqual(
                    probe.verify_packages(Path("unused.lock")),
                    {"extra-package": "1.0", "sglang": "0.0.0", "torch": "2.8.0"},
                )

    def test_import_origin_rejects_other_checkouts_and_escaping_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            python = source / "python"
            python.mkdir(parents=True)
            local = python / "module.py"
            local.touch()
            module = SimpleNamespace(__file__=str(local))
            self.assertEqual(probe.source_origin(module, source), str(local.resolve()))
            outside = Path(directory) / "other.py"
            outside.touch()
            alias = python / "alias.py"
            alias.symlink_to(outside)
            for filename in [outside, alias]:
                with self.subTest(filename=filename):
                    module.__file__ = str(filename)
                    with self.assertRaisesRegex(RuntimeError, "outside"):
                        probe.source_origin(module, source)

    def test_rust_extension_must_match_loader_current_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            artifact = source / "cache" / "current" / "_server.so"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"verified artifact")
            crate = SimpleNamespace(workspace=source / "rust")
            context = SimpleNamespace(
                fingerprint="current", source_digest="source-digest"
            )
            module = SimpleNamespace(__file__=str(artifact))
            loader = SimpleNamespace(
                _discover_crate=Mock(return_value=crate),
                _build_context=Mock(return_value=context),
                _cache_root=Mock(return_value=source / "cache"),
                _cached_extension_path=Mock(return_value=artifact),
                load_rust_extension=Mock(return_value=module),
                _command_version=Mock(return_value="test toolchain"),
            )
            extension, _ = probe.verify_rust(source, loader)
            self.assertEqual(extension["fingerprint"], "current")
            self.assertEqual(
                extension["sha256"], hashlib.sha256(b"verified artifact").hexdigest()
            )
            loader.load_rust_extension.assert_called_once_with(
                probe.RUST_MODULE, mode="auto", workspace=source / "rust"
            )
            module.__file__ = str(source / "cache" / "stale" / "_server.so")
            with self.assertRaisesRegex(
                RuntimeError, "expected current source artifact"
            ):
                probe.verify_rust(source, loader)

    def test_backend_requires_a_working_device_and_real_companion_imports(self):
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.tensor.return_value.sum.return_value.item.return_value = 3.0
        torch.cuda.get_device_name.return_value = "test GPU"
        torch.cuda.get_device_capability.return_value = (8, 0)
        torch.version.cuda = "12.8"
        mx = Mock()
        mx.metal.is_available.return_value = True
        mx.add.return_value.item.return_value = 3.0
        mx.metal.device_info.return_value = {"device_name": "test Metal"}
        modules = {
            "torch": torch,
            "mlx.core": mx,
            "torchvision": Mock(),
            "torchaudio": Mock(),
            "sgl_kernel": Mock(),
        }
        with patch.object(
            probe.importlib, "import_module", side_effect=modules.__getitem__
        ) as imports:
            self.assertEqual(probe.verify_backend("cuda")["device"], "cuda")
            torch.tensor.assert_called_once_with([1.0, 2.0], device="cuda")
            self.assertIn(call("sgl_kernel"), imports.call_args_list)
            self.assertEqual(probe.verify_backend("mlx")["device"], "metal")
            self.assertIs(mx.add.call_args.kwargs["stream"], mx.gpu)
            mx.eval.assert_called_once_with(mx.add.return_value)
            for backend, available in [
                ("cuda", torch.cuda.is_available),
                ("mlx", mx.metal.is_available),
            ]:
                available.return_value = False
                with self.subTest(backend=backend):
                    with self.assertRaisesRegex(RuntimeError, "device is unavailable"):
                        probe.verify_backend(backend)
            for companion in ["torchvision", "torchaudio"]:
                self.assertIn(call(companion), imports.call_args_list)

    def test_dependencies_only_avoids_source_and_device_imports_and_writes_failures(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probe.json"
            args = [
                "--source",
                directory,
                "--lock",
                str(Path(directory) / "requirements.lock"),
                "--backend",
                "mlx",
                "--python",
                sys.executable,
                "--output",
                str(output),
                "--python-version",
                "3.12.8",
                "--dependencies-only",
            ]
            with (
                patch.object(probe, "verify_python"),
                patch.object(
                    probe, "verify_packages", return_value={"torch": "2.8.0"}
                ) as packages,
                patch.object(probe.importlib, "import_module") as imports,
            ):
                self.assertEqual(probe.main(args), 0)
                self.assertEqual(
                    json.loads(output.read_text())["packages"], {"torch": "2.8.0"}
                )
                imports.assert_not_called()
                packages.side_effect = RuntimeError("missing dependency")
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(probe.main(args), 1)
                result = json.loads(output.read_text())
                self.assertEqual(result["status"], "failed")
                self.assertIn("missing dependency", result["error"])
                self.assertNotIn("rust_extension", result)


if __name__ == "__main__":
    unittest.main()
