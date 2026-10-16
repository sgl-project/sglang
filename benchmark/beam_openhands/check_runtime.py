"""Validate the prepared Beam serving runtime without installing dependencies."""

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path

from curve_config import ENGINE_COMMIT

EXPECTED = {
    "torch": "2.14.1",
    "transformers": "5.19.0",
    "tokenizers": "0.23.2",
    "sglang-kernel": "0.4.9",
    "flashinfer-python": "0.7.1",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    subprocess.run(
        ["git", "-C", str(repo), "diff", "--exit-code", ENGINE_COMMIT, "--", "python"],
        check=True,
    )
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    mismatches = {
        name: version
        for name, version in versions.items()
        if version.split("+")[0] != EXPECTED[name]
    }
    if mismatches:
        raise RuntimeError(
            f"Serving runtime differs from the measured recipe: {mismatches}; expected {EXPECTED}"
        )
    import flashinfer
    import torch

    import sglang

    if Path(sglang.__file__).resolve() != repo / "python/sglang/__init__.py":
        raise RuntimeError(f"Wrong serving source imported: {sglang.__file__}")
    if not torch.version.cuda or not torch.version.cuda.startswith("13."):
        raise RuntimeError(f"This recipe requires CUDA 13; found {torch.version.cuda}")
    result = dict(
        versions=versions,
        cuda=torch.version.cuda,
        sglang=sglang.__file__,
        flashinfer=flashinfer.__file__,
        engine_commit=ENGINE_COMMIT,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
