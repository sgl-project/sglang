import importlib.util
import os
import sys
import types


def _get_unavailable_kernel(name):
    if name.startswith("__"):
        raise AttributeError(name)

    def unavailable(*args, **kwargs):
        raise RuntimeError(
            f"SGLang Simulator attempted to execute native kernel sgl_kernel.{name}. "
            "The macOS simulator supports scheduling simulation only."
        )

    return unavailable


def install_load_utils_stub() -> None:
    """Install the kernel loader stub before importing the sgl_kernel package."""
    if (
        sys.platform == "darwin"
        and os.environ.get("SGLANG_USE_CPU_ENGINE") == "1"
        and "sgl_kernel" not in sys.modules
    ):
        # macOS only ships Metal kernels, not the CPU exports imported by
        # SGLang's scheduler modules. Simulation never executes model kernels.
        # Allow imports, but fail loudly if execution reaches a native op.
        package = types.ModuleType("sgl_kernel")
        package.__path__ = []
        package.__spec__ = importlib.util.spec_from_loader(
            "sgl_kernel", loader=None, is_package=True
        )
        package.__getattr__ = _get_unavailable_kernel
        sys.modules["sgl_kernel"] = package

    module_name = "sgl_kernel.load_utils"
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        module.__package__ = "sgl_kernel"
        sys.modules[module_name] = module

    module._load_architecture_specific_ops = lambda *args, **kwargs: None
    module._preload_cuda_library = lambda *args, **kwargs: None
