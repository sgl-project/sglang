import sys
import types


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
