"""Check the prebuilt environment before launching this text-only benchmark."""

import argparse
import importlib.metadata
import importlib.util
import shutil
from pathlib import Path

from packaging.specifiers import SpecifierSet

# Matches this branch and the upstream v0.5.19 release's pyproject.toml.
# CUDA wheel local suffixes (e.g. 2.13.0+cu130) satisfy ==2.13.0.
REQUIREMENTS = {
    "torch": "==2.13.0",
    "flashinfer-python": "==0.6.18",
    "sglang-kernel": "==0.4.6.post1",
    "apache-tvm-ffi": "==0.1.11",
    "transformers": "==5.12.1",
    "tokenizers": "==0.22.2",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages-only", action="store_true")
    args = parser.parse_args()
    errors = []
    for package, requirement in REQUIREMENTS.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{package} is missing (requires {requirement})")
            continue
        print(f"{package}: {version}", flush=True)
        if version not in SpecifierSet(requirement):
            errors.append(f"{package}{requirement} required; found {version}")
    for command in ("c++", "ninja", "nvcc", "git"):
        if shutil.which(command) is None:
            errors.append(f"Missing tool: {command}")
    if errors:
        raise SystemExit("Incompatible benchmark environment:\n" + "\n".join(errors))
    if args.packages_only:
        return

    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.find_spec("sglang")
    expected = repo / "python/sglang/__init__.py"
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
        raise SystemExit(f"SGLang must load from {expected}; check PYTHONPATH")
    print(f"SGLang source: {expected}", flush=True)

    import torch

    if not torch.version.cuda or torch.version.cuda.split(".")[0] != "13":
        raise SystemExit(f"Expected CUDA 13 PyTorch; found CUDA {torch.version.cuda}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly the one assigned GPU to this container")
    print(f"GPU: {torch.cuda.get_device_name(0)}; CUDA: {torch.version.cuda}")
    print("Environment checks passed. Server startup will compile NGRAM as needed.")


if __name__ == "__main__":
    main()
