"""Check environment configurations and dependency versions."""

import importlib.metadata
import os
import resource
import subprocess
import sys
from collections import OrderedDict, defaultdict

import torch

from sglang.srt.utils import is_hip, is_npu


def is_cuda_v2():
    return torch.version.cuda is not None


# List of packages to check versions
PACKAGE_LIST = [
    "sglang",
    "sgl_kernel",
    "flashinfer_python",
    "triton",
    "transformers",
    "torchao",
    "numpy",
    "aiohttp",
    "fastapi",
    "hf_transfer",
    "huggingface_hub",
    "interegular",
    "modelscope",
    "orjson",
    "outlines",
    "packaging",
    "psutil",
    "pydantic",
    "python-multipart",
    "pyzmq",
    "torchao",
    "uvicorn",
    "uvloop",
    "vllm",
    "xgrammar",
    "openai",
    "tiktoken",
    "anthropic",
    "litellm",
    "decord",
]


def get_package_versions(packages):
    """
    Get versions of specified packages.
    """
    versions = {}
    for package in packages:
        package_name = package.split("==")[0].split(">=")[0].split("<=")[0]
        try:
            version = importlib.metadata.version(package_name)
            versions[package_name] = version
        except ModuleNotFoundError:
            versions[package_name] = "Module Not Found"
    return versions


def get_cuda_info():
    """
    Get CUDA-related information if available.
    """
    if is_cuda_v2():
        cuda_info = {"CUDA available": torch.cuda.is_available()}

        if cuda_info["CUDA available"]:
            cuda_info.update(_get_gpu_info())
            cuda_info.update(_get_cuda_version_info())

        return cuda_info
    elif is_hip():
        cuda_info = {"ROCM available": torch.cuda.is_available()}

        if cuda_info["ROCM available"]:
            cuda_info.update(_get_gpu_info())
            cuda_info.update(_get_cuda_version_info())

        return cuda_info
    elif is_npu():
        import torch_npu

        cuda_info = {"NPU available": torch_npu.npu.is_available()}
        if cuda_info["NPU available"]:
            cuda_info.update(_get_npu_info())
            cuda_info.update(_get_cuda_version_info())
            PACKAGE_LIST.insert(0, "torch_npu")

        return cuda_info


def _get_gpu_info():
    """
    Get information about available GPUs.
    """
    devices = defaultdict(list)
    capabilities = defaultdict(list)
    for k in range(torch.cuda.device_count()):
        devices[torch.cuda.get_device_name(k)].append(str(k))
        capability = torch.cuda.get_device_capability(k)
        capabilities[f"{capability[0]}.{capability[1]}"].append(str(k))

    gpu_info = {}
    for name, device_ids in devices.items():
        gpu_info[f"GPU {','.join(device_ids)}"] = name

    if len(capabilities) == 1:
        # All GPUs have the same compute capability
        cap, gpu_ids = list(capabilities.items())[0]
        gpu_info[f"GPU {','.join(gpu_ids)} Compute Capability"] = cap
    else:
        # GPUs have different compute capabilities
        for cap, gpu_ids in capabilities.items():
            gpu_info[f"GPU {','.join(gpu_ids)} Compute Capability"] = cap

    return gpu_info


def _get_npu_info():
    """
    Get information about available NPUs.
    Cannot merged into `_get_gpu_info` due to torch_npu interface differences and NPUs do not have compute capabilities.
    """
    devices = defaultdict(list)
    for k in range(torch.npu.device_count()):
        devices[torch.npu.get_device_name(k)].append(str(k))

    npu_info = {}
    for name, device_ids in devices.items():
        npu_info[f"NPU {','.join(device_ids)}"] = name

    return npu_info


def _get_npu_cann_home():
    cann_envs = ["ASCEND_TOOLKIT_HOME", "ASCEND_INSTALL_PATH"]
    for var in cann_envs:
        path = os.environ.get(var)
        if path and os.path.exists(path):
            return path

    default_path = "/usr/local/Ascend/ascend-toolkit/latest"
    return default_path if os.path.exists(default_path) else None


def _get_cuda_version_info():
    """
    Get CUDA version information.
    """
    if is_cuda_v2():
        from torch.utils.cpp_extension import CUDA_HOME

        cuda_info = {"CUDA_HOME": CUDA_HOME}

        if CUDA_HOME and os.path.isdir(CUDA_HOME):
            cuda_info.update(_get_nvcc_info())
            cuda_info.update(_get_cuda_driver_version())

        return cuda_info
    elif is_hip():
        from torch.utils.cpp_extension import ROCM_HOME as ROCM_HOME

        cuda_info = {"ROCM_HOME": ROCM_HOME}

        if ROCM_HOME and os.path.isdir(ROCM_HOME):
            cuda_info.update(_get_nvcc_info())
            cuda_info.update(_get_cuda_driver_version())

        return cuda_info
    elif is_npu():
        CANN_HOME = _get_npu_cann_home()

        npu_info = {"CANN_HOME": CANN_HOME}

        if CANN_HOME:
            npu_info.update(_get_nvcc_info())
            npu_info.update(_get_cuda_driver_version())

        return npu_info
    else:
        cuda_info = {"CUDA_HOME": ""}
        return cuda_info


def _get_nvcc_info():
    """
    Get NVCC version information.
    """
    if is_cuda_v2():
        from torch.utils.cpp_extension import CUDA_HOME

        try:
            nvcc = os.path.join(CUDA_HOME, "bin/nvcc")
            nvcc_output = (
                subprocess.check_output(f'"{nvcc}" -V', shell=True)
                .decode("utf-8")
                .strip()
            )
            return {
                "NVCC": nvcc_output[
                    nvcc_output.rfind("Cuda compilation tools") : nvcc_output.rfind(
                        "Build"
                    )
                ].strip()
            }
        except subprocess.SubprocessError:
            return {"NVCC": "Not Available"}
    elif is_hip():
        from torch.utils.cpp_extension import ROCM_HOME

        try:
            hipcc = os.path.join(ROCM_HOME, "bin/hipcc")
            hipcc_output = (
                subprocess.check_output(f'"{hipcc}" --version', shell=True)
                .decode("utf-8")
                .strip()
            )
            return {
                "HIPCC": hipcc_output[
                    hipcc_output.rfind("HIP version") : hipcc_output.rfind("AMD clang")
                ].strip()
            }
        except subprocess.SubprocessError:
            return {"HIPCC": "Not Available"}
    elif is_npu():
        CANN_HOME = _get_npu_cann_home()

        cann_info = {}
        cann_version_file = os.path.join(CANN_HOME, "version.cfg")
        if os.path.exists(cann_version_file):
            with open(cann_version_file, "r", encoding="utf-8") as f:
                f.readline()  # discard first line comment in version.cfg
                cann_info["CANN"] = f.readline().split("[")[1].split("]")[0]
        else:
            cann_info["CANN"] = "Not Available"
        try:
            bisheng = os.path.join(CANN_HOME, "compiler/ccec_compiler/bin/bisheng")
            bisheng_output = (
                subprocess.check_output([bisheng, "--version"]).decode("utf-8").strip()
            )
            cann_info["BiSheng"] = bisheng_output.split("\n")[0].strip()
        except subprocess.SubprocessError:
            cann_info["BiSheng"] = "Not Available"
        return cann_info
    else:
        return {"NVCC": "Not Available"}


def _get_cuda_driver_version():
    """
    Get CUDA driver version.
    """
    versions = set()
    if is_cuda_v2():
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ]
            )
            versions = set(output.decode().strip().split("\n"))
            if len(versions) == 1:
                return {"CUDA Driver Version": versions.pop()}
            else:
                return {"CUDA Driver Versions": ", ".join(sorted(versions))}
        except subprocess.SubprocessError:
            return {"CUDA Driver Version": "Not Available"}
    elif is_hip():
        try:
            output = subprocess.check_output(
                [
                    "rocm-smi",
                    "--showdriverversion",
                    "--csv",
                ]
            )
            versions = set(output.decode().strip().split("\n"))
            versions.discard("name, value")
            ver = versions.pop()
            ver = ver.replace('"Driver version", ', "").replace('"', "")

            return {"ROCM Driver Version": ver}
        except subprocess.SubprocessError:
            return {"ROCM Driver Version": "Not Available"}
    elif is_npu():
        try:
            output = subprocess.check_output(
                [
                    "npu-smi",
                    "info",
                    "-t",
                    "board",
                    "-i",
                    "0",
                ]
            )
            for line in output.decode().strip().split("\n"):
                if "Software Version" in line:
                    version = line.split(":")[-1].strip()
                    break
            else:
                version = "Not Available"

            return {"Ascend Driver Version": version}
        except subprocess.SubprocessError:
            return {"Ascend Driver Version": "Not Available"}
    else:
        return {"CUDA Driver Version": "Not Available"}


def get_gpu_topology():
    """
    Get GPU topology information.
    """
    if is_cuda_v2():
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
    elif is_hip():
        try:
            result = subprocess.run(
                ["rocm-smi", "--showtopotype"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return "\n" + result.stdout if result.returncode == 0 else None
        except subprocess.SubprocessError:
            return None
    elif is_npu():
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "topo"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return "\n" + result.stdout if result.returncode == 0 else None
        except subprocess.SubprocessError:
            return None
    else:
        return None


def get_hypervisor_vendor():
    try:
        output = subprocess.check_output(["lscpu"], text=True)
        for line in output.split("\n"):
            if "Hypervisor vendor:" in line:
                return line.split(":")[1].strip()
        return None
    except:
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
        if is_cuda_v2():
            env_info["NVIDIA Topology"] = gpu_topo
        elif is_hip():
            env_info["AMD Topology"] = gpu_topo
        elif is_npu():
            env_info["Ascend NPU"] = gpu_topo

    hypervisor_vendor = get_hypervisor_vendor()
    if hypervisor_vendor:
        env_info["Hypervisor vendor"] = hypervisor_vendor

    ulimit_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    env_info["ulimit soft"] = ulimit_soft

    for k, v in env_info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    check_env()
