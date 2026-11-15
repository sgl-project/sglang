"""Build the Rust extension modules in place for CI, compiling shared deps once.

setup.py build_rust runs one cargo invocation per extension, and cargo resolves
features per invocation, so a dependency shared by several extensions compiles
once per extension. Here the root-workspace crates that declare no features
and no torch-compat-header build in one cargo invocation instead; the rest go
through setup.py, a crate with features alone so they do not leak into other
crates' copy of it (rust/sglang-mm/Cargo.toml), and a torch crate with the
environment setup.py gives it.

The shared invocation is not byte-identical to setup.py's builds: cargo unifies
the dependency features of the crates it builds together, and it cannot LTO a
lib that also builds as an rlib, which setup.py avoids with --crate-type cdylib.
Wheels and local installs keep setup.py.

Reads SGLANG_BUILD_RUST_EXTS and SETUPTOOLS_RUST_CARGO_PROFILE as setup.py does.
"""

import json
import os
import runpy
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[3]
PYTHON_DIR = REPO_DIR / "python"
RUST_DIR = REPO_DIR / "rust"
DISCOVERY = runpy.run_path(
    os.fspath(PYTHON_DIR / "sglang" / "srt" / "rust_extensions" / "discovery.py")
)


def build_shared(extensions):
    command = [
        "cargo",
        "build",
        "--lib",
        "--message-format=json-render-diagnostics",
        "--manifest-path",
        os.fspath(RUST_DIR / "Cargo.toml"),
        "--profile",
        os.environ["SETUPTOOLS_RUST_CARGO_PROFILE"],
        "--locked",
        "--features",
        "pyo3/extension-module",
    ]
    for extension in extensions:
        command += ["-p", extension.package_name]
    # What setuptools-rust sets for a PyO3 extension; pyo3's build script
    # reruns when they change, so they also keep its units shared with the
    # setup.py builds that follow.
    env = dict(os.environ)
    env.setdefault("PYO3_PYTHON", sys.executable)
    env.setdefault("PYO3_BUILD_EXTENSION_MODULE", "1")
    print(" ".join(command), file=sys.stderr, flush=True)
    stdout = subprocess.run(
        command, env=env, stdout=subprocess.PIPE, text=True, check=True
    ).stdout

    libraries = {}
    for line in stdout.splitlines():
        message = json.loads(line)
        if (
            message["reason"] == "compiler-artifact"
            and "cdylib" in message["target"]["kind"]
        ):
            libraries[message["manifest_path"]] = [
                name for name in message["filenames"] if name.endswith(".so")
            ]

    # Where setuptools-rust's build_rust --inplace puts the module, copied the
    # way it copies it: unstripped, with the read bits mirrored to execute.
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    for extension in extensions:
        (library,) = libraries[extension.manifest_path]
        module_path = PYTHON_DIR / (extension.name.replace(".", "/") + suffix)
        shutil.copyfile(library, module_path)
        mode = module_path.stat().st_mode
        module_path.chmod(mode | (mode & 0o444) >> 2)
        print(f"copied {library} -> {module_path}", file=sys.stderr)


def main():
    selected = DISCOVERY["selected_rust_extensions"](
        DISCOVERY["declared_rust_extensions"](
            rust_dir=RUST_DIR, pyproject=PYTHON_DIR / "pyproject.toml"
        )
    )
    shared = [
        extension
        for extension in selected
        if Path(extension.workspace_root) == RUST_DIR
        and not extension.metadata.get("features")
        and "torch-compat-header" not in extension.metadata
    ]
    rest = [extension for extension in selected if extension not in shared]
    if shared:
        build_shared(shared)
    if rest:
        subprocess.run(
            [sys.executable, "setup.py", "build_rust", "--inplace"],
            cwd=PYTHON_DIR,
            env={
                **os.environ,
                DISCOVERY["BUILD_RUST_EXTS_ENV"]: ",".join(e.name for e in rest),
            },
            check=True,
        )


if __name__ == "__main__":
    main()
