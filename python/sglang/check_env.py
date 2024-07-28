"""Check environment configurations and dependency versions."""

import importlib
import os
import resource
import subprocess
import sys
from collections import OrderedDict, defaultdict

import torch

# List of packages to check versions for
PACKAGE_LIST = [
    "sglang",
    "flashinfer",
    "requests",
    "tqdm",
    "numpy",
    "aiohttp",
    "fastapi",
    "hf_transfer",
    "huggingface_hub",
    "interegular",
    "packaging",
    "PIL",
    "psutil",
    "pydantic",
    "uvicorn",
    "uvloop",
    "zmq",
    "vllm",
    "outlines",
    "openai",
    "tiktoken",
    "anthropic",
    "litellm",
]


def get_package_versions(packages):
    """
    Get versions of specified packages.
    """
    versions = {}
    for package in packages:
        package_name = package.split("==")[0].split(">=")[0].split("<=")[0]
        try:
            module = importlib.import_module(package_name)
            if hasattr(module, "__version__"):
                versions[package_name] = module.__version__
        except ModuleNotFoundError:
            versions[package_name] = "Module Not Found"
    return versions


def get_cuda_info():
    """
    Get CUDA-related information if available.
    """
    cuda_info = {"CUDA available": torch.cuda.is_available()}

    if cuda_info["CUDA available"]:
        cuda_info.update(_get_gpu_info())
        cuda_info.update(_get_cuda_version_info())

    return cuda_info


def _get_gpu_info():
    """
    Get information about available GPUs.
    """
    devices = defaultdict(list)
    for k in range(torch.cuda.device_count()):
        devices[torch.cuda.get_device_name(k)].append(str(k))

    return {f"GPU {','.join(device_ids)}": name for name, device_ids in devices.items()}


def _get_cuda_version_info():
    """
    Get CUDA version information.
    """
    from torch.utils.cpp_extension import CUDA_HOME

    cuda_info = {"CUDA_HOME": CUDA_HOME}

    if CUDA_HOME and os.path.isdir(CUDA_HOME):
        cuda_info.update(_get_nvcc_info())
        cuda_info.update(_get_cuda_driver_version())

    return cuda_info


def _get_nvcc_info():
    """
    Get NVCC version information.
    """
    from torch.utils.cpp_extension import CUDA_HOME

    try:
        nvcc = os.path.join(CUDA_HOME, "bin/nvcc")
        nvcc_output = (
            subprocess.check_output(f'"{nvcc}" -V', shell=True).decode("utf-8").strip()
        )
        return {
            "NVCC": nvcc_output[
                nvcc_output.rfind("Cuda compilation tools") : nvcc_output.rfind("Build")
            ].strip()
        }
    except subprocess.SubprocessError:
        return {"NVCC": "Not Available"}


def _get_cuda_driver_version():
    """
    Get CUDA driver version.
    """
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        return {"CUDA Driver Version": output.decode().strip()}
    except subprocess.SubprocessError:
        return {"CUDA Driver Version": "Not Available"}


def get_gpu_topology():
    """
    Get GPU topology information.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return "\n" + result.stdout if result.returncode == 0 else None
    except subprocess.SubprocessError:
        return None


def check_env():
    """
    Check and print environment information.
    """
    env_info = OrderedDict()
    env_info["Python"] = sys.version.replace("\n", "")
    env_info.update(get_cuda_info())
    env_info["PyTorch"] = torch.__version__
    env_info.update(get_package_versions(PACKAGE_LIST))

    gpu_topo = get_gpu_topology()
    if gpu_topo:
        env_info["NVIDIA Topology"] = gpu_topo

    ulimit_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    env_info["ulimit soft"] = ulimit_soft

    for k, v in env_info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    check_env()
