import sys
import types
from pathlib import Path


class _SimulatorQuantizationConfig:
    @staticmethod
    def override_quantization_method(*_args, **_kwargs):
        return None


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


def install_quantization_stub() -> None:
    """Avoid importing accelerator quantization kernels for dummy model loads."""
    module_name = "sglang.srt.layers.quantization"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)
    module.__package__ = "sglang.srt.layers"
    module.__path__ = [
        str(path)
        for entry in sys.path
        if (path := Path(entry) / "sglang/srt/layers/quantization").is_dir()
    ]
    module.QUANTIZATION_METHODS = {"fp8": _SimulatorQuantizationConfig}
    module.QuantizationConfig = _SimulatorQuantizationConfig
    module.get_quantization_config = lambda *_args, **_kwargs: None
    sys.modules[module_name] = module
