# Generates a PEP 503 simple index for sgl-deep-gemm wheels under
# sgl-whl/cu<version>/sgl-deep-gemm/index.html. Mirrors the layout used by
# update_kernel_whl_index.py so consumers can `pip install
# sgl-deep-gemm --extra-index-url https://...whl/cu129`.

import argparse
import hashlib
import pathlib
import re

SUPPORTED_CUDA_VERSIONS = ["129", "130"]


def update_wheel_index(cuda_version, wheel_dir):
    index_dir = pathlib.Path(f"sgl-whl/cu{cuda_version}/sgl-deep-gemm")
    index_dir.mkdir(exist_ok=True, parents=True)
    base_url = "https://github.com/sgl-project/whl/releases/download"

    suffix = f"+cu{cuda_version}"
    for path in sorted(pathlib.Path(wheel_dir).glob("*.whl")):
        if suffix not in path.name:
            continue
        with open(path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        match = re.match(r"sgl_deep_gemm-([0-9][^-+]*)(?:\+cu[0-9]+)?-", path.name)
        if not match:
            continue
        ver = match.group(1)
        full_url = f"{base_url}/v{ver}/{path.name}#sha256={sha256}"
        with (index_dir / "index.html").open("a") as f:
            f.write(f'<a href="{full_url}">{path.name}</a><br>\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", type=str, required=True, choices=SUPPORTED_CUDA_VERSIONS
    )
    parser.add_argument("--wheel-dir", type=str, default="dist")
    args = parser.parse_args()
    update_wheel_index(args.cuda, args.wheel_dir)


if __name__ == "__main__":
    main()
