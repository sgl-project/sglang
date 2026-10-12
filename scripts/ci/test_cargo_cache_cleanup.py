import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

WORKSPACE = '[workspace]\nresolver = "3"\nmembers = ["types", "consumer"]\n'
TYPES_TOML = '[package]\nname = "types"\nversion = "0.1.0"\nedition = "2021"\n'
CONSUMER_TOML = (
    '[package]\nname = "consumer"\nversion = "0.1.0"\nedition = "2021"\n'
    '[dependencies]\ntypes = { path = "../types" }\n'
)


def _write_checkout(rust_dir, with_text):
    field = ", pub text: String" if with_text else ""
    init = ", text: String::new()" if with_text else ""
    files = {
        "Cargo.toml": WORKSPACE,
        "types/Cargo.toml": TYPES_TOML,
        "types/src/lib.rs": f"pub struct R {{ pub id: u32{field} }}\n",
        "consumer/Cargo.toml": CONSUMER_TOML,
        "consumer/src/lib.rs": f"pub fn f() -> types::R {{ types::R {{ id: 1{init} }} }}\n",
    }
    for relative, text in files.items():
        path = rust_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(
        ["cargo", "generate-lockfile", "--manifest-path", rust_dir / "Cargo.toml"],
        check=True,
        capture_output=True,
    )


@unittest.skipIf(shutil.which("cargo") is None, "needs a Rust toolchain")
class TestCargoCacheCleanup(unittest.TestCase):
    def _setup_cargo_cache(self, repo_root, env):
        source = (ROOT / "scripts/ci/cuda/ci_install_dependency.sh").read_text()
        function = re.search(r"^setup_cargo_cache\(\) \{\n.*?^\}", source, re.M | re.S)[
            0
        ]
        script = (
            function
            + '\nmark_step_done() { :; }\nsetup_cargo_cache\necho "$CARGO_TARGET_DIR"\n'
        )
        result = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", script],
            env=dict(env, REPO_ROOT=str(repo_root)),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().splitlines()[-1]

    def test_rebuilds_local_crates_built_by_another_checkout(self):
        """A local crate another checkout built must not pass as Fresh, with or without --target."""
        host = re.search(
            r"^host: (\S+)$",
            subprocess.run(
                ["rustc", "-vV"], capture_output=True, text=True, check=True
            ).stdout,
            re.M,
        )[1]
        for build_target in (None, host):
            with (
                self.subTest(build_target=build_target),
                tempfile.TemporaryDirectory() as tmp,
            ):
                tmp = Path(tmp)
                env = {
                    k: v for k, v in os.environ.items() if not k.startswith("CARGO_")
                }
                env.pop("SGLANG_BUILD_RUST_EXTS", None)
                env["HOME"] = str(tmp / "home")
                env["CARGO_HOME"] = os.environ.get(
                    "CARGO_HOME", str(Path.home() / ".cargo")
                )
                env.setdefault("RUSTUP_HOME", str(Path.home() / ".rustup"))
                if build_target:
                    env["CARGO_BUILD_TARGET"] = build_target
                build = ["cargo", "build", "--release", "--manifest-path"]

                # This checkout's crates were written before another PR built into the cache.
                checkout = tmp / "checkout"
                _write_checkout(checkout / "rust", with_text=False)
                for path in (checkout / "rust/types").rglob("*"):
                    os.utime(path, (0, 0))

                other = tmp / "other"
                _write_checkout(other / "rust", with_text=True)
                target_dir = self._setup_cargo_cache(other, env)
                subprocess.run(
                    [*build, other / "rust/Cargo.toml"],
                    env=dict(env, CARGO_TARGET_DIR=target_dir),
                    check=True,
                    capture_output=True,
                )
                (checkout / "rust/consumer/src/lib.rs").touch()

                target_dir = self._setup_cargo_cache(checkout, env)
                result = subprocess.run(
                    [*build, checkout / "rust/Cargo.toml"],
                    env=dict(env, CARGO_TARGET_DIR=target_dir),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
