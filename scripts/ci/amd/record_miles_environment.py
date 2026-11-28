import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from pathlib import Path


def package_info(module_name: str, distribution_name: str) -> dict[str, object]:
    spec = importlib.util.find_spec(module_name)
    locations = [] if spec is None else list(spec.submodule_search_locations or [])
    origin = None if spec is None else spec.origin
    if origin:
        directory = Path(origin).parent
    elif locations:
        directory = Path(locations[0])
    else:
        directory = None
    try:
        version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    record: dict[str, object] = {
        "origin": origin,
        "locations": locations,
        "distribution": distribution_name,
        "version": version,
    }
    # Wheels may have no Git checkout; missing provenance must stay explicit.
    for field, args in (
        ("git_commit", ["rev-parse", "HEAD"]),
        ("git_status", ["status", "--porcelain", "--untracked-files=no"]),
    ):
        record[field] = None
        if directory is not None:
            result = subprocess.run(
                ["git", "-C", str(directory.resolve()), *args],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                record[field] = result.stdout.strip()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-sha", required=True)
    args = parser.parse_args()
    rocm_version = Path("/opt/rocm/.info/version")
    print(
        json.dumps(
            {
                "workflow_sha": args.workflow_sha,
                "python": platform.python_version(),
                "rocm": rocm_version.read_text().strip()
                if rocm_version.exists()
                else None,
                "packages": {
                    module: package_info(module, distribution)
                    for module, distribution in (
                        ("miles", "miles"),
                        ("sglang", "sglang"),
                        ("megatron", "megatron-core"),
                        ("torch", "torch"),
                        ("aiter", "amd-aiter"),
                        ("tilelang", "tilelang"),
                        ("mooncake", "mooncake-transfer-engine"),
                        ("sglang_router", "sglang-router"),
                    )
                },
            },
            indent=2,
        )
    )
