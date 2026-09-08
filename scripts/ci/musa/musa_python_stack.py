"""Build constraints and verify the Python stack used by MUSA CI."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import re
import site
from pathlib import Path

CORE_DISTRIBUTIONS = (
    "torch",
    "torch-musa",
    "torchada",
    "triton",
)

OPTIONAL_VENDOR_DISTRIBUTIONS = (
    "apache-tvm-ffi",
    "deep-gemm",
    "flash-attn-3",
    "mate",
    "mthreads-ml-py",
    "mt-sparse-attention",
    "torchaudio",
    "torchvision",
)

OPTIONAL_STACK_DISTRIBUTIONS = ("setuptools",)

# compressed-tensors 0.16+ requires Torch 2.10+, while the older MUSA runner
# stack uses Torch 2.9. Keep this mapping explicit instead of globally pinning
# one version in pyproject_other.toml for every accelerator stack.
COMPRESSED_TENSORS_BY_TORCH_MINOR = {
    (2, 9): "0.15.0",
    (2, 11): "0.17.0",
}

LOGGER = logging.getLogger(__name__)


class StackError(RuntimeError):
    """Raised when the installed MUSA stack violates the CI contract."""


def distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise StackError(f"required distribution is not installed: {name}") from exc


def torch_minor(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise StackError(f"cannot parse Torch version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def compressed_tensors_version(torch_version: str) -> str:
    minor = torch_minor(torch_version)
    try:
        return COMPRESSED_TENSORS_BY_TORCH_MINOR[minor]
    except KeyError as exc:
        supported = ", ".join(
            f"{major}.{minor}"
            for major, minor in sorted(COMPRESSED_TENSORS_BY_TORCH_MINOR)
        )
        raise StackError(
            f"unsupported MUSA Torch line {minor[0]}.{minor[1]}; "
            f"supported lines: {supported}"
        ) from exc


def build_constraints() -> list[str]:
    versions = {name: distribution_version(name) for name in CORE_DISTRIBUTIONS}
    pins = [f"{name}=={version}" for name, version in versions.items()]

    for name in OPTIONAL_VENDOR_DISTRIBUTIONS + OPTIONAL_STACK_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        pins.append(f"{name}=={version}")

    pins.append(f"compressed-tensors=={compressed_tensors_version(versions['torch'])}")
    return sorted(pins, key=str.casefold)


def write_constraints(output: Path) -> None:
    pins = build_constraints()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(pins) + "\n", encoding="utf-8")
    LOGGER.info("Wrote MUSA constraints to %s", output)
    for pin in pins:
        LOGGER.info("  %s", pin)


def validate_core_versions(versions: dict[str, str]) -> None:
    torch_line = torch_minor(versions["torch"])
    torch_musa_line = torch_minor(versions["torch-musa"])
    if torch_line != torch_musa_line:
        raise StackError(
            "Torch and Torch-MUSA lines do not match: "
            f"torch={versions['torch']}, torch-musa={versions['torch-musa']}"
        )


def triton_metadata() -> dict[str, object]:
    import triton

    distribution = importlib.metadata.distribution("triton")
    summary = distribution.metadata.get("Summary") or ""
    module_file = getattr(triton, "__file__", None)
    if module_file is None:
        raise StackError("cannot locate the imported Triton module")
    module_path = Path(module_file).resolve()
    backend_root = module_path.parent / "backends"
    if not backend_root.is_dir():
        raise StackError(f"Triton backend directory is missing: {backend_root}")
    backends = sorted(path.name for path in backend_root.iterdir() if path.is_dir())
    return {
        "version": distribution.version,
        "summary": summary,
        "module": str(module_path),
        "backends": backends,
        "user_site": site.getusersitepackages(),
    }


def verify_stack(
    *,
    expected_triton_version: str,
    require_driver: bool,
    require_resolved_dependencies: bool,
    require_user_site: bool,
    triton_only: bool,
) -> None:
    if require_driver:
        import torchada  # noqa: F401

    info = triton_metadata()
    if info["version"] != expected_triton_version:
        raise StackError(
            "unexpected Triton version: "
            f"observed={info['version']}, expected={expected_triton_version}"
        )
    if "MUSA" not in str(info["summary"]):
        raise StackError(f"Triton is not the MUSA build: {info['summary']!r}")
    if "mtgpu" not in info["backends"]:
        raise StackError(f"Triton has no mtgpu backend: {info['backends']}")
    if require_user_site:
        module_path = Path(str(info["module"]))
        user_site = Path(str(info["user_site"])).resolve()
        if not module_path.is_relative_to(user_site):
            raise StackError(
                "Triton was not imported from the task-local user site: "
                f"module={module_path}, user_site={user_site}"
            )
    if triton_only:
        LOGGER.info(json.dumps({"triton": info}, indent=2, sort_keys=True))
        return

    versions = {name: distribution_version(name) for name in CORE_DISTRIBUTIONS}
    validate_core_versions(versions)
    result: dict[str, object] = {
        "versions": versions,
        "triton": info,
        "driver_checked": require_driver,
    }

    if require_resolved_dependencies:
        expected_compressed_tensors = compressed_tensors_version(versions["torch"])
        observed_compressed_tensors = distribution_version("compressed-tensors")
        if observed_compressed_tensors != expected_compressed_tensors:
            raise StackError(
                "compressed-tensors does not match the MUSA Torch line: "
                f"observed={observed_compressed_tensors}, "
                f"expected={expected_compressed_tensors}"
            )
        result["compressed_tensors"] = observed_compressed_tensors

    if require_driver:
        import torch
        from triton.runtime import driver

        musa_version = getattr(torch.version, "musa", None)
        if musa_version is None:
            raise StackError(f"Torch is not a MUSA build: {torch.__version__}")
        if not hasattr(torch, "musa"):
            raise StackError("torch.musa is unavailable after importing torchada")
        device_count = torch.musa.device_count()
        if device_count < 1:
            raise StackError(f"no MUSA device is visible: device_count={device_count}")

        target = driver.active.get_current_target()
        if getattr(target, "backend", None) != "musa":
            raise StackError(f"Triton active target is not MUSA: {target}")
        result.update(
            {
                "musa_version": musa_version,
                "device_count": device_count,
                "target": repr(target),
            }
        )

    LOGGER.info(json.dumps(result, indent=2, sort_keys=True, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    constraints = subparsers.add_parser("constraints")
    constraints.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--expected-triton-version", required=True)
    verify.add_argument("--require-driver", action="store_true")
    verify.add_argument("--require-resolved-dependencies", action="store_true")
    verify.add_argument("--require-user-site", action="store_true")
    verify.add_argument("--triton-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    try:
        if args.command == "constraints":
            write_constraints(args.output)
        else:
            verify_stack(
                expected_triton_version=args.expected_triton_version,
                require_driver=args.require_driver,
                require_resolved_dependencies=args.require_resolved_dependencies,
                require_user_site=args.require_user_site,
                triton_only=args.triton_only,
            )
    except StackError as exc:
        raise SystemExit(f"MUSA Python stack error: {exc}") from exc


if __name__ == "__main__":
    main()
