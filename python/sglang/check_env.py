"""Check environment configurations and dependency versions."""

import importlib.metadata
import os
import resource
import subprocess
import sys
from abc import abstractmethod
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
    "flashinfer_cubin",
    "flashinfer_jit_cache",
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
    "decord2",
]


class BaseEnv:
    """Base class for environment check"""

    def __init__(self):
        self.package_list = PACKAGE_LIST

    @abstractmethod
    def get_info(self) -> dict:
        """
        Get CUDA-related information if available.
        """
        raise NotImplementedError

    @abstractmethod
    def get_topology(self) -> dict:
        raise NotImplementedError

    def get_package_versions(self) -> dict:
        """
        Get versions of specified packages.
        """
        versions = {}
        for package in self.package_list:
            package_name = package.split("==")[0].split(">=")[0].split("<=")[0]
            try:
                version = importlib.metadata.version(package_name)
                versions[package_name] = version
            except ModuleNotFoundError:
                versions[package_name] = "Module Not Found"
        return versions

    def get_device_info(self):
        """
        Get information about available GPU devices.
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

    def get_hypervisor_vendor(self) -> dict:
        try:
            output = subprocess.check_output(["lscpu"], text=True)
            for line in output.split("\n"):
                if "Hypervisor vendor:" in line:
                    return {"Hypervisor vendor:": line.split(":")[1].strip()}
            return {}
        except:
            return {}

    def get_ulimit_soft(self) -> dict:
        ulimit_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        return {"ulimit soft": ulimit_soft}

    def check_env(self):
        """
        Check and print environment information.
        """
        env_info = OrderedDict()
        env_info["Python"] = sys.version.replace("\n", "")
        env_info.update(self.get_info())
        env_info["PyTorch"] = torch.__version__
        env_info.update(self.get_package_versions())
        env_info.update(self.get_topology())
        env_info.update(self.get_hypervisor_vendor())
        env_info.update(self.get_ulimit_soft())

        for k, v in env_info.items():
            print(f"{k}: {v}")


class GPUEnv(BaseEnv):
    """Environment checker for Nvidia GPU"""

    def get_info(self):
        cuda_info = {"CUDA available": torch.cuda.is_available()}

        if cuda_info["CUDA available"]:
            cuda_info.update(self.get_device_info())
            cuda_info.update(self._get_cuda_version_info())

        return cuda_info

    def _get_cuda_version_info(self):
        """
        Get CUDA version information.
        """
        from torch.utils.cpp_extension import CUDA_HOME

        cuda_info = {"CUDA_HOME": CUDA_HOME}

        if CUDA_HOME and os.path.isdir(CUDA_HOME):
            cuda_info.update(self._get_nvcc_info())
            cuda_info.update(self._get_cuda_driver_version())

        return cuda_info

    def _get_nvcc_info(self):
        """
        Get NVCC version information.
        """
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

    def _get_cuda_driver_version(self):
        """
        Get CUDA driver version.
        """
        versions = set()
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

    def get_topology(self):
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
            return {
                "NVIDIA Topology": (
                    "\n" + result.stdout if result.returncode == 0 else None
                )
            }
        except subprocess.SubprocessError:
            return {}


class HIPEnv(BaseEnv):
    """Environment checker for ROCm/HIP"""

    def get_info(self):
        cuda_info = {"ROCM available": torch.cuda.is_available()}

        if cuda_info["ROCM available"]:
            cuda_info.update(self.get_device_info())
            cuda_info.update(self._get_cuda_version_info())

        return cuda_info

    def _get_cuda_version_info(self):
        from torch.utils.cpp_extension import ROCM_HOME as ROCM_HOME

        cuda_info = {"ROCM_HOME": ROCM_HOME}

        if ROCM_HOME and os.path.isdir(ROCM_HOME):
            cuda_info.update(self._get_hipcc_info())
            cuda_info.update(self._get_rocm_driver_version())

        return cuda_info

    def _get_hipcc_info(self):
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

    def _get_rocm_driver_version(self):
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

    def get_topology(self):
        try:
            result = subprocess.run(
                ["rocm-smi", "--showtopotype"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return {
                "AMD Topology": "\n" + result.stdout if result.returncode == 0 else None
            }
        except subprocess.SubprocessError:
            return {}


class NPUEnv(BaseEnv):
    """Environment checker for Ascend NPU"""

    EXTRA_PACKAGE_LIST = [
        "torch_npu",
        "sgl-kernel-npu",
        "deep_ep",
    ]

    def __init__(self):
        super().__init__()
        self.package_list.extend(NPUEnv.EXTRA_PACKAGE_LIST)

    def get_info(self):
        cuda_info = {"NPU available": torch.npu.is_available()}
        if cuda_info["NPU available"]:
            cuda_info.update(self.get_device_info())
            cuda_info.update(self._get_cann_version_info())

        return cuda_info

    def get_device_info(self):
        """
        Get information about available NPUs.
        Need to override due to torch_npu interface differences.
        """
        devices = defaultdict(list)
        for k in range(torch.npu.device_count()):
            devices[torch.npu.get_device_name(k)].append(str(k))

        npu_info = {}
        for name, device_ids in devices.items():
            npu_info[f"NPU {','.join(device_ids)}"] = name

        return npu_info

    def _get_cann_version_info(self):
        cann_envs = ["ASCEND_TOOLKIT_HOME", "ASCEND_INSTALL_PATH"]
        for var in cann_envs:
            path = os.environ.get(var)
            if path and os.path.exists(path):
                CANN_HOME = path
                break
        else:
            default_path = "/usr/local/Ascend/ascend-toolkit/latest"
            CANN_HOME = default_path if os.path.exists(default_path) else None

        if CANN_HOME:
            npu_info = {"CANN_HOME": CANN_HOME}
            npu_info.update(self._get_cann_info(CANN_HOME))
            npu_info.update(self._get_ascend_driver_version())
            return npu_info
        else:
            return {"CANN_HOME": "Not found"}

    def _get_cann_info(self, CANN_HOME: str):
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

    def _get_ascend_driver_version(self):
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

    def get_topology(self):
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "topo"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return {
                "Ascend Topology": (
                    "\n" + result.stdout if result.returncode == 0 else None
                )
            }
        except subprocess.SubprocessError:
            return {}


if __name__ == "__main__":
    if is_cuda_v2():
        env = GPUEnv()
    elif is_hip():
        env = HIPEnv()
    elif is_npu():
        env = NPUEnv()
    env.check_env()
