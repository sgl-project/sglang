"""Install Dynamo 1.6.0 ready to serve Kimi-K3.1 with SGLang, no per-job setup script needed.

`ai-dynamo` 1.6.0 comes from PyPI. `ai-dynamo-runtime` (the Rust bindings, module dynamo._core)
is rebuilt from ai-dynamo/dynamo v1.6.0 with the two patches in dynamo/:
  0001  ai-dynamo/dynamo#15333 (rebase of #12191): keep the session id across the Python
        chat-processor hop (--dyn-chat-processor sglang)
  0002  carry the parent session id the same way and start a subagent on its parent's worker
Both touch only lib/bindings/python and lib/llm. The features match the K3.1 GB300 image.

The `ai-dynamo` Python package then gets the K3.1 edits of dynamo/dynamo_sglang_k31_patch.py
(applied unmodified): DSA decode context parallel is accepted for
KimiK31ForConditionalGeneration, and the model card is built from a staged copy whose config.json
says model_type "kimi_k3", which Dynamo's tokenizer maps to the Kimi regex.
dynamo/register_stage_model_card.py is appended to register.py so that copy is staged on first use
instead of by a setup script. The setup script still runs cleanly on this image: it finds every
edit already applied.

Idempotent: a marker records this recipe's key and the sha256 of the installed dynamo/_core, so a
rerun is a no-op unless the recipe changed or a reinstall replaced the runtime. The Rust toolchain
and the source tree live in a temporary directory that is removed after the build.
"""

import hashlib
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DYNAMO_VERSION = "1.6.0"
DYNAMO_REPO = "https://github.com/ai-dynamo/dynamo"
DYNAMO_COMMIT = "701da25292ecca8f64d770cc0b17d389200019a1"  # tag v1.6.0
FEATURES = "kv-indexer,slot-tracker,select-service,mm-routing,ais-forward-pass,request-trace-s3,nvtx"
HERE = Path(__file__).resolve().parent
PATCHES = sorted((HERE / "dynamo").glob("*.patch"))
SGLANG_PATCHER = HERE / "dynamo/dynamo_sglang_k31_patch.py"
STAGE_HOOK = HERE / "dynamo/register_stage_model_card.py"
STAGE_HOOK_MARKER = "# K3.1 image: stage the K3.1 model card on first use"
# libclang-dev: bindgen in the nixl-sys crate needs libclang.
APT_PACKAGES = ["protobuf-compiler", "pkg-config", "libssl-dev", "libclang-dev"]

SITE = Path(sysconfig.get_paths()["purelib"])
CORE = SITE / "dynamo/_core.abi3.so"
DYNAMO_SGLANG = SITE / "dynamo/sglang"
MARKER = SITE / "dynamo/k31-dynamo.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*cmd: str, cwd: Path, env=None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def installed(name: str):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    inputs = [Path(__file__), *PATCHES, SGLANG_PATCHER, STAGE_HOOK]
    key = hashlib.sha256(b"".join(p.read_bytes() for p in inputs)).hexdigest()[:16]
    register = DYNAMO_SGLANG / "register.py"
    if (
        MARKER.exists()
        and CORE.exists()
        and register.exists()
        and STAGE_HOOK_MARKER in register.read_text()
    ):
        marker = json.loads(MARKER.read_text())
        if (
            marker.get("key") == key
            and marker.get("core_sha256") == sha256(CORE)
            and installed("ai-dynamo") == DYNAMO_VERSION
        ):
            print(
                f"Dynamo {DYNAMO_VERSION} with the runtime patches and K3.1 edits already installed (key {key})"
            )
            return

    run("apt-get", "update", "-qq", cwd=Path("/"))
    run(
        "apt-get",
        "install",
        "-y",
        "-qq",
        "--no-install-recommends",
        *APT_PACKAGES,
        cwd=Path("/"),
    )
    run("bash", "-c", "rm -rf /var/lib/apt/lists/*", cwd=Path("/"))
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        f"ai-dynamo=={DYNAMO_VERSION}",
        cwd=Path("/"),
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        env = dict(
            os.environ, RUSTUP_HOME=str(tmp / "rustup"), CARGO_HOME=str(tmp / "cargo")
        )
        env["PATH"] = f"{tmp / 'cargo/bin'}:{env['PATH']}"
        run(
            "bash",
            "-c",
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs"
            " | sh -s -- -y --profile minimal --default-toolchain none --no-modify-path",
            cwd=tmp,
            env=env,
        )
        src = tmp / "dynamo"
        run("git", "clone", "--quiet", DYNAMO_REPO, str(src), cwd=tmp)
        run("git", "checkout", "--quiet", DYNAMO_COMMIT, cwd=src)
        for patch in PATCHES:
            run("git", "apply", str(patch), cwd=src)
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "maturin[patchelf]>=1.0,<2.0",
            cwd=tmp,
        )
        # rust-toolchain.toml pins the compiler; cargo installs it on first use.
        bindings = src / "lib/bindings/python"
        run(
            sys.executable,
            "-m",
            "maturin",
            "build",
            "--release",
            "--locked",
            "--features",
            FEATURES,
            "--out",
            str(tmp / "wheels"),
            cwd=bindings,
            env=env,
        )
        wheels = sorted((tmp / "wheels").glob("ai_dynamo_runtime-*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"expected one ai-dynamo-runtime wheel, got {wheels}")
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(wheels[0]),
            cwd=tmp,
        )

    # The edits fail closed on a missing anchor; a model dir without config.json skips the staging
    # step, which the appended hook does at runtime.
    with tempfile.TemporaryDirectory() as tmp:
        run(
            sys.executable,
            "-I",
            "-B",
            str(SGLANG_PATCHER),
            str(DYNAMO_SGLANG),
            str(Path(tmp) / "no-model"),
            str(Path(tmp) / "staged"),
            cwd=Path("/"),
        )
    register = DYNAMO_SGLANG / "register.py"
    if STAGE_HOOK_MARKER not in register.read_text():
        with open(register, "a", encoding="utf-8") as f:
            f.write(STAGE_HOOK.read_text())
    for pyc in (DYNAMO_SGLANG / "__pycache__").glob("*.pyc"):
        pyc.unlink()

    if installed("ai-dynamo-runtime") != DYNAMO_VERSION:
        raise SystemExit(
            f"ai-dynamo-runtime {installed('ai-dynamo-runtime')} installed, expected {DYNAMO_VERSION}"
        )
    MARKER.write_text(
        json.dumps({"key": key, "commit": DYNAMO_COMMIT, "core_sha256": sha256(CORE)})
    )
    print(
        f"installed Dynamo {DYNAMO_VERSION} with the runtime patches and K3.1 edits (key {key})"
    )


if __name__ == "__main__":
    main()
