"""Validate the PPU runner environment without importing SGLang."""

import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import traceback
from pathlib import Path


class CheckFailure(RuntimeError):
    """A failed PPU environment contract check."""


def require(condition: bool, message: str) -> None:
    """Raise a concise contract failure when a condition is false."""
    if not condition:
        raise CheckFailure(message)


def print_host_summary() -> None:
    """Print capacity information without hostnames or network identifiers."""
    disk_usage = shutil.disk_usage("/")

    print(f"Runner OS: {platform.system()}")
    print(f"Runner architecture: {platform.machine()}")
    print(f"Python version: {platform.python_version()}")

    # Memory reporting is optional; sysconf support varies by platform.
    try:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        memory_bytes = -1
    if memory_bytes > 0:
        print(f"Memory total (bytes): {memory_bytes}")
    else:
        print("Memory total (bytes): unavailable")
    print(f"Root disk total (bytes): {disk_usage.total}")
    print(f"Root disk available (bytes): {disk_usage.free}")


def check_ppu_inventory() -> int:
    """Return the number of PPUs reported by the management interface."""
    ppu_smi = shutil.which("ppu-smi")
    require(ppu_smi is not None, "ppu-smi is not available on PATH")

    try:
        result = subprocess.run(
            [ppu_smi, "--query-ppu=index", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CheckFailure(
            f"ppu-smi could not query the PPU inventory: {error}"
        ) from error

    if result.returncode != 0:
        stderr = result.stderr.strip()
        detail = f" (rc={result.returncode})"
        if stderr:
            detail += f": {stderr}"
        raise CheckFailure(f"ppu-smi failed to query the PPU inventory{detail}")

    inventory_lines = result.stdout.splitlines()
    indices = [line.strip() for line in inventory_lines if line.strip()]
    require(indices, "ppu-smi did not report any PPU devices")
    require(
        all(re.fullmatch(r"[0-9]+", index) for index in indices),
        "ppu-smi returned an unexpected inventory format",
    )

    print(f"PPU devices reported: {len(indices)}")
    return len(indices)


def check_device_nodes() -> None:
    """Check the character devices used by the current PPU runtime."""
    for device_node in (Path("/dev/alixpu_ctl"), Path("/dev/alixpu")):
        try:
            is_character_device = stat.S_ISCHR(device_node.stat().st_mode)
        except OSError as error:
            raise CheckFailure(
                f"Required PPU character device is unavailable: {device_node}"
            ) from error
        require(
            is_character_device,
            f"Required PPU device is not a character device: {device_node}",
        )

    print("Required PPU device nodes are present.")


def check_ppu_sdk() -> None:
    """Verify that PPU_SDK names an installed SDK directory."""
    ppu_sdk = os.environ.get("PPU_SDK", "")
    require(ppu_sdk, "PPU_SDK is not set")
    require(
        Path(ppu_sdk).is_dir(),
        f"PPU_SDK must name an existing directory, got {ppu_sdk!r}",
    )
    print(f"PPU SDK directory is available: {ppu_sdk}")


def check_torch_compute() -> int:
    """Run a minimal PPU computation using PyTorch's CUDA-compatible API."""
    try:
        import torch
    except Exception as error:
        raise CheckFailure(f"PyTorch cannot be imported: {error}") from error

    require(
        torch.cuda.is_available(),
        "torch.cuda.is_available() returned false; PPU is reached through the "
        "CUDA-compatible device type, so this means no PPU is usable",
    )
    device_count = torch.cuda.device_count()
    require(
        device_count > 0,
        "torch.cuda.device_count() did not report a device",
    )

    # Device-name lookup is optional and must not block the compute check.
    try:
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    except Exception:
        print("torch.cuda.get_device_name(0): unavailable")

    device = torch.device("cuda:0")
    try:
        source = torch.tensor([1.0, 2.0, 3.0], device=device)
        actual = source * 2.0 + 1.0
        torch.cuda.synchronize(device)
        actual_cpu = actual.cpu()
        expected = torch.tensor([3.0, 5.0, 7.0])
        torch.testing.assert_close(actual_cpu, expected)
    except Exception as error:
        raise CheckFailure(f"Tensor compute on cuda:0 failed: {error}") from error

    print(f"PyTorch version: {torch.__version__}")
    print(f"PPU devices visible through torch.cuda: {device_count}")
    print("Tensor compute on cuda:0 passed.")
    return device_count


def check_visibility_consistency(reported: int, visible: int) -> None:
    """Require visible devices not to exceed the ppu-smi inventory.

    Device visibility restrictions may expose fewer devices to PyTorch.
    """
    require(
        visible <= reported,
        f"torch reports {visible} device(s) but ppu-smi reports only {reported}",
    )
    if visible < reported:
        print(
            f"Note: {visible} of {reported} PPU(s) visible to this job "
            f"(device visibility is restricted per runner job)."
        )
    else:
        print(f"ppu-smi and torch agree on {reported} device(s).")


def main() -> int:
    """Run every check and return a process-compatible status code."""
    try:
        print_host_summary()
        reported = check_ppu_inventory()
        check_device_nodes()
        check_ppu_sdk()
        visible = check_torch_compute()
        check_visibility_consistency(reported, visible)
    except CheckFailure as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"ERROR: unexpected preflight failure ({type(error).__name__}): {error}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 1

    print("PPU runner environment preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
