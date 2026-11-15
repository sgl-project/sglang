"""sglang build hooks.

Rust extensions are auto-discovered from the Cargo workspace in ../rust and
the extension manifests declared by its workspace metadata. Every crate whose
Cargo.toml declares

    [package.metadata.sglang]
    python-module = "sglang.srt.<pkg>._core"   # import path inside the wheel
    debug = false                              # optional RustExtension knob
    features = ["python"]                      # optional cargo features to enable

is built as a PyO3 extension module at that import path. Adding a new extension
crate therefore needs no pyproject changes — declare the metadata in the crate.

Two filters can narrow the discovered set:

- [tool.sglang] rust-extensions in the active pyproject.toml: a list of
  case-insensitive substrings of the target module. Platform pyprojects use
  this to build a subset (e.g. pyproject_other.toml builds only "multimodal";
  grpc needs proto/tonic and is intentionally CUDA-only).
- SGLANG_BUILD_RUST_EXTS env var, applied at build time on top of the above:
  unset or "all" builds everything, "none" builds nothing, and a
  comma-separated list matches substrings, e.g. "grpc" matches
  "sglang.srt.rust_extensions._grpc". It is read directly from os.environ
  instead of sglang.srt.environ, which is not importable until the package is
  built.
"""

import os
import runpy
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.sdist import sdist

try:
    from setuptools_rust import Binding, RustExtension, build_rust
except ModuleNotFoundError as exc:
    if exc.name != "setuptools_rust":
        raise
    # Alternate platform pyprojects that build no Rust extensions do not
    # install setuptools-rust.
    build_rust = None

_PYTHON_DIR = Path(__file__).resolve().parent
_RUST_EXTENSIONS_DIR = _PYTHON_DIR / "sglang" / "srt" / "rust_extensions"
_RUST_BUILD_HELPERS = runpy.run_path(os.fspath(_RUST_EXTENSIONS_DIR / "torch_build.py"))
_torch_build_configuration = _RUST_BUILD_HELPERS["torch_build_configuration"]
_RUST_DISCOVERY = runpy.run_path(os.fspath(_RUST_EXTENSIONS_DIR / "discovery.py"))
_selected_rust_extensions = _RUST_DISCOVERY["selected_rust_extensions"]


def _declared_rust_extensions():
    """One RustExtension per declared crate the pyproject allowlist keeps."""
    extensions = []
    for declared in _RUST_DISCOVERY["declared_rust_extensions"](
        rust_dir=_PYTHON_DIR.parent / "rust", pyproject=_PYTHON_DIR / "pyproject.toml"
    ):
        extension = RustExtension(
            target=declared.name,
            path=declared.manifest_path,
            binding=Binding.PyO3,
            debug=declared.metadata.get("debug"),
            # Crates that gate their PyO3 bindings behind a non-default
            # feature (so the pure-Rust core stays pyo3-free) declare it here.
            features=declared.metadata.get("features"),
            cargo_manifest_args=["--locked"],
        )
        # Preserve Cargo metadata until the selected extension is actually
        # built. Alternate platform pyprojects filter the Rust TreeCore extension
        # out before this point and therefore do not need torch as a build dependency.
        extension._sglang_metadata = declared.metadata
        extension._sglang_manifest_path = declared.manifest_path
        extensions.append(extension)
    return extensions


class Sdist(sdist):
    """Materialize shared GGUF headers in the standalone source distribution."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        relative = Path("sglang/kernels/jit/include/sgl_kernel/quantization/gguf")
        canonical = (
            _PYTHON_DIR / "sglang/kernels/aot/include/sgl_kernel/quantization/gguf"
        )
        included = {Path(name) for name in files}
        for header in sorted((_PYTHON_DIR / relative).glob("*")):
            name = header.relative_to(_PYTHON_DIR)
            if name not in included or not header.is_symlink():
                continue
            target = header.resolve(strict=True)
            if target.parent != canonical.resolve():
                raise ValueError(f"Shared GGUF header escapes its owner: {header}")
            destination = Path(base_dir) / name
            # make_release_tree may hard-link the source symlink itself.
            destination.unlink()
            shutil.copyfile(target, destination)


if build_rust is not None:

    class BuildRust(build_rust):
        """Build only the Rust extensions selected by SGLANG_BUILD_RUST_EXTS."""

        def run_for_extension(self, extension) -> None:
            metadata = extension._sglang_metadata
            compat_header = metadata.get("torch-compat-header")
            if compat_header is not None:
                manifest = Path(extension._sglang_manifest_path)
                build = _torch_build_configuration(
                    compat_header=manifest.parent / compat_header,
                    python_module=extension.name,
                    include_absolute_rpath=False,
                )
                extension.env.env = build.environment
            super().run_for_extension(extension)

        def run(self) -> None:
            rust_extensions = _selected_rust_extensions(self.extensions or [])
            self.extensions = rust_extensions
            self.distribution.rust_extensions = rust_extensions
            if not rust_extensions:
                return
            super().run()

    setup(
        cmdclass={"build_rust": BuildRust, "sdist": Sdist},
        rust_extensions=_declared_rust_extensions(),
    )
else:
    setup(cmdclass={"sdist": Sdist})
