import sys
import types

from sglang_simulator.hook import BaseHook


def install_load_utils_stub() -> None:
    """Install the kernel loader stub before importing the sgl_kernel package."""
    module_name = "sgl_kernel.load_utils"
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        module.__package__ = "sgl_kernel"
        sys.modules[module_name] = module

    module._load_architecture_specific_ops = lambda *args, **kwargs: None
    module._preload_cuda_library = lambda *args, **kwargs: None


class M_SGLangKernelLoadUtilHook(BaseHook):
    HOOK_CLASS_NAME = ""
    HOOK_MODULE_NAME = "sgl_kernel.load_utils"

    @classmethod
    def hook(cls, target):
        def override_load_architecture_specific_ops(*args, **kwargs):
            """
            ImportError:
            [sgl_kernel] CRITICAL: Could not load any common_ops library!
            """
            pass

        target._load_architecture_specific_ops = override_load_architecture_specific_ops
