# Reference: https://github.com/flashinfer-ai/flashinfer/blob/v0.2.0/scripts/update_whl_index.py

import argparse
import hashlib
import pathlib
import re

# All the CUDA versions that the wheels will cover
SUPPORTED_CUDA_VERSIONS = ["129", "130"]
DEFAULT_CUDA_VERSION = "129"


def check_wheel_cuda_version(path_name, target_cuda_version):
    # For other CUDA versions, the wheel path name will contain the cuda version suffix, e.g. sgl_kernel-0.3.16.post5+cu130-cp310-abi3-manylinux2014_x86_64.whl
    if target_cuda_version != DEFAULT_CUDA_VERSION:
        return target_cuda_version in path_name

    # For the default CUDA version, the wheel path name will not contain any cuda version suffix, e.g. sgl_kernel-0.3.16.post5-cp310-abi3-manylinux2014_x86_64.whl
    # So we need to check if the wheel path name contains any other cuda version suffix
    for cuda_version in SUPPORTED_CUDA_VERSIONS:
        if cuda_version != DEFAULT_CUDA_VERSION and cuda_version in path_name:
            return False
    return True


def update_wheel_index(cuda_version=DEFAULT_CUDA_VERSION):
    index_dir = pathlib.Path(f"sgl-whl/cu{cuda_version}/sgl-kernel")
    index_dir.mkdir(exist_ok=True)
    base_url = "https://github.com/sgl-project/whl/releases/download"

    for path in sorted(pathlib.Path("sgl-kernel/dist").glob("*.whl")):
        # Skip the wheel if mismatches the passed in cuda_version
        if not check_wheel_cuda_version(path.name, cuda_version):
            continue
        with open(path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        ver = re.findall(
            r"sgl_kernel-([0-9.]+(?:\.post[0-9]+)?)(?:\+cu[0-9]+)?-", path.name
        )[0]
        full_url = f"{base_url}/v{ver}/{path.name}#sha256={sha256}"
        with (index_dir / "index.html").open("a") as f:
            f.write(f'<a href="{full_url}">{path.name}</a><br>\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="118")
    args = parser.parse_args()
    update_wheel_index(args.cuda)


if __name__ == "__main__":
    main()
