# Reference: https://github.com/flashinfer-ai/flashinfer/blob/v0.2.0/scripts/update_whl_index.py

import argparse
import hashlib
import pathlib
import re

DEFAULT_CUDA_VERSION = "130"
# Local version a CUDA wheel carries, e.g. sglang_kernel-0.4.6.post1+cu130-...whl
CUDA_LOCAL_VERSION_PATTERN = re.compile(r"\+cu(\d+)")


def check_wheel_cuda_version(path_name, target_cuda_version):
    # Skip non-CUDA backend wheels. ROCm/MUSA encode the backend in the
    # local-version tag (for example +rocm720), while XPU uses a dedicated
    # package name (sglang_kernel_xpu-*).
    if re.search(r"\+(rocm|musa)", path_name) or path_name.startswith(
        "sglang_kernel_xpu-"
    ):
        return False

    # Match on the wheel's own +cuNNN tag rather than a list of known versions,
    # so a wheel built for a CUDA version this script has never heard of is
    # rejected instead of landing in the target index.
    match = CUDA_LOCAL_VERSION_PATTERN.search(path_name)
    if match is not None:
        return match.group(1) == target_cuda_version

    # An untagged wheel is the default-CUDA build, e.g.
    # sglang_kernel-0.4.0-cp310-abi3-manylinux2014_x86_64.whl (PyPI rejects
    # local versions, so that upload strips the tag).
    return target_cuda_version == DEFAULT_CUDA_VERSION


def update_wheel_index(cuda_version=DEFAULT_CUDA_VERSION, rocm_version=None):
    index_dir = pathlib.Path(f"sgl-whl/cu{cuda_version}/sglang-kernel")
    index_dir.mkdir(exist_ok=True, parents=True)
    base_url = "https://github.com/sgl-project/whl/releases/download"

    for path in sorted(pathlib.Path("python/sglang/kernels/aot/dist").glob("*.whl")):
        # Skip the wheel if mismatches the passed in cuda_version
        if not check_wheel_cuda_version(path.name, cuda_version):
            continue
        with open(path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        ver = re.findall(
            r"sglang_kernel-([0-9.]+(?:\.post[0-9]+)?)(?:\+cu[0-9]+)?-", path.name
        )[0]
        full_url = f"{base_url}/v{ver}/{path.name}#sha256={sha256}"
        with (index_dir / "index.html").open("a") as f:
            f.write(f'<a href="{full_url}">{path.name}</a><br>\n')


def _update_non_cuda_wheel_index(
    backend,
    version=None,
    package_name="sglang_kernel",
    index_package_name="sglang-kernel",
    release_tag_suffix="",
):
    backend_dir = f"{backend}{version or ''}"
    index_dir = pathlib.Path(f"sgl-whl/{backend_dir}/{index_package_name}")
    index_dir.mkdir(exist_ok=True, parents=True)
    base_url = "https://github.com/sgl-project/whl/releases/download"

    for path in sorted(pathlib.Path("python/sglang/kernels/aot/dist").glob("*.whl")):
        # Skip the wheel if not for this backend
        if re.search(f"{backend}", path.name) is None:
            continue
        with open(path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        ver = re.findall(
            rf"{re.escape(package_name)}-([0-9.]+(?:\.post[0-9]+)?)(?:\+{backend}[0-9]+)?-",
            path.name,
        )[0]
        full_url = f"{base_url}/v{ver}{release_tag_suffix}/{path.name}#sha256={sha256}"
        with (index_dir / "index.html").open("a") as f:
            f.write(f'<a href="{full_url}">{path.name}</a><br>\n')


def update_wheel_index_xpu():
    _update_non_cuda_wheel_index(
        "xpu",
        package_name="sglang_kernel_xpu",
        index_package_name="sglang-kernel-xpu",
        release_tag_suffix="+xpu",
    )


def update_wheel_index_rocm(rocm_version):
    _update_non_cuda_wheel_index("rocm", rocm_version)


def update_wheel_index_musa(musa_version):
    _update_non_cuda_wheel_index("musa", musa_version)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default=DEFAULT_CUDA_VERSION)
    parser.add_argument("--rocm", type=str, default=None)
    parser.add_argument("--musa", type=str, default=None)
    parser.add_argument("--xpu", action="store_true")
    args = parser.parse_args()
    if args.xpu:
        update_wheel_index_xpu()
    elif args.musa is not None:
        update_wheel_index_musa(args.musa)
    elif args.rocm is not None:
        update_wheel_index_rocm(args.rocm)
    else:
        update_wheel_index(args.cuda)


if __name__ == "__main__":
    main()
