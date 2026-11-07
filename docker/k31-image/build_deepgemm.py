"""Build sgl-deep-gemm 0.2.3.post2 with the MegaMoE decode block_m heuristic.

python/pyproject.toml pins sgl-deep-gemm 0.2.3.post2. MegaMoE wide-EP decode wants
sgl-project/DeepGEMM#130 (block_m 64 for decode-sized loads), which is host-side heuristic code
compiled into deep_gemm/_C.so, so the PyPI wheel cannot be patched in place. This builds the
v0.2.3.post2 tag with that one change and replaces the installed wheel. It fails if another
sgl-deep-gemm version is installed, so a pin bump cannot be silently rebuilt back to 0.2.3.post2.

Idempotent: a marker next to the installed _C.so records this recipe's key and the module's
sha256, so a rerun is a no-op unless the recipe changed or a reinstall replaced the wheel.
"""

import hashlib
import json
import subprocess
import sys
import sysconfig
import tempfile
from importlib.metadata import version
from pathlib import Path

DEEPGEMM_REPO = "https://github.com/sgl-project/DeepGEMM"
DEEPGEMM_VERSION = "0.2.3.post2"
DEEPGEMM_COMMIT = "3bd6c364e49d5c34f2f0819b6f632bd6f7a3503a"  # tag v0.2.3.post2
HERE = Path(__file__).resolve().parent
PATCH = HERE / "deepgemm-mega-moe-decode-block-m.patch"
# deep_jit includes elfutils/libdwfl.h, which the nightly image does not ship.
APT_PACKAGES = ["libdw-dev"]

SITE = Path(sysconfig.get_paths()["purelib"])
MARKER = SITE / "deep_gemm/k31-deepgemm.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module() -> Path:
    found = sorted((SITE / "deep_gemm").glob("_C*.so"))
    if not found:
        raise SystemExit("no deep_gemm/_C*.so is installed")
    return found[0]


def run(*cmd: str, cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    installed = version("sgl-deep-gemm")
    if installed != DEEPGEMM_VERSION:
        raise SystemExit(
            f"sgl-deep-gemm {installed} is installed; the MegaMoE decode patch is for "
            f"{DEEPGEMM_VERSION}: update DEEPGEMM_VERSION, DEEPGEMM_COMMIT and the patch"
        )
    key = hashlib.sha256(Path(__file__).read_bytes() + PATCH.read_bytes()).hexdigest()[
        :16
    ]
    if MARKER.exists():
        marker = json.loads(MARKER.read_text())
        if marker.get("key") == key and marker.get("module_sha256") == sha256(module()):
            print(
                f"sgl-deep-gemm already built with the MegaMoE decode patch (key {key})"
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

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "DeepGEMM"
        run("git", "clone", "--quiet", DEEPGEMM_REPO, str(src), cwd=Path(tmp))
        run("git", "checkout", "--quiet", DEEPGEMM_COMMIT, cwd=src)
        run("git", "submodule", "update", "--init", "--recursive", "--quiet", cwd=src)
        run("git", "apply", str(PATCH), cwd=src)
        run("bash", "build_sgl_deep_gemm.sh", cwd=src)
        wheels = sorted((src / "dist").glob("*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"expected one sgl-deep-gemm wheel, got {wheels}")
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(wheels[0]),
            cwd=src,
        )

    MARKER.write_text(
        json.dumps(
            {"key": key, "commit": DEEPGEMM_COMMIT, "module_sha256": sha256(module())}
        )
    )
    print(
        f"installed sgl-deep-gemm {DEEPGEMM_COMMIT[:8]} + MegaMoE decode patch (key {key})"
    )


if __name__ == "__main__":
    main()
