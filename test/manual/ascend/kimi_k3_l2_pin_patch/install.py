"""Install the reference prints into the package selected by the Nightly pod."""

import argparse
import fcntl
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path


def _atomic_write(path, content, mode=0o644):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            os.fchmod(stream.fileno(), mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install(package_root):
    payload = Path(__file__).resolve().parent
    target = package_root.resolve() / "srt/mem_cache/pool_host"
    mamba = target / "mamba.py"
    helper = target / "npu_pin_diagnostics.py"
    helper_content = (payload / helper.name).read_bytes()
    patch = payload / "mamba-pin-diagnostics.patch"

    # The mounted checkout is shared by four pods. Serialize check/apply/write
    # and publish each file atomically; repeat installs must not append hooks.
    with (target / ".k3-pin-diagnostics.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = mamba.read_bytes()
        if helper.exists() and helper.read_bytes() != helper_content:
            raise RuntimeError(
                f"A different diagnostic helper already exists: {helper}"
            )
        with tempfile.TemporaryDirectory(prefix="k3-pin-patch-") as temporary:
            root = Path(temporary)
            staged = root / "python/sglang/srt/mem_cache/pool_host/mamba.py"
            staged.parent.mkdir(parents=True)
            staged.write_bytes(original)
            command = ["git", "apply", "--check", str(patch)]
            check = subprocess.run(
                command, cwd=root, capture_output=True, text=True, check=False
            )
            if check.returncode == 0:
                subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
                updated = staged.read_bytes()
                status = "applied"
            else:
                reverse = subprocess.run(
                    ["git", "apply", "--reverse", "--check", str(patch)],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if reverse.returncode:
                    raise RuntimeError(f"Patch does not match {mamba}: {check.stderr}")
                updated = original
                status = "already applied"
        compile(updated, str(mamba), "exec")
        compile(helper_content, str(helper), "exec")
        if updated != original:
            backup = mamba.with_name("mamba.py.k3-pin-original")
            if backup.exists():
                raise RuntimeError(f"Refusing to replace an existing backup: {backup}")
            _atomic_write(backup, original, mamba.stat().st_mode & 0o777)
        # Publish the dependency first, so another pod can safely import mamba.
        _atomic_write(helper, helper_content)
        if updated != original:
            _atomic_write(mamba, updated, mamba.stat().st_mode & 0o777)
        print(
            f"K3_PIN_PATCH {status}: {mamba} "
            f"sha256={hashlib.sha256(updated).hexdigest()}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, type=Path)
    install(parser.parse_args().package_root)
