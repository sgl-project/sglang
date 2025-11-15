# TODO: also move pad_input_ids into this module
import importlib
import inspect
import logging
import pkgutil
from importlib import import_module

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

PROCESSOR_MAPPING = {}


def import_processors(package_name: str):
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                logger.warning(f"Ignore import error when loading {name}: {e}")
                continue
            all_members = inspect.getmembers(module, inspect.isclass)
            classes = [
                member
                for name, member in all_members
                if member.__module__ == module.__name__
            ]
            for cls in (
                cls for cls in classes if issubclass(cls, BaseMultimodalProcessor)
            ):
                assert hasattr(cls, "models")
                for arch in getattr(cls, "models"):
                    PROCESSOR_MAPPING[arch] = cls


def get_mm_processor(
    hf_config, server_args: ServerArgs, processor, transport_mode
) -> BaseMultimodalProcessor:

    if not PROCESSOR_MAPPING:
        # 尝试自动导入 processors 包（会触发各子模块的注册逻辑/装饰器）
        try:
            import_processors("sglang.srt.multimodal.processors")
        except Exception as e:
            logger.warning(
                "Failed to import multimodal processors from %s: %s",
                "sglang.srt.multimodal.processors",
                e,
            )

    # 1) 优先走注册表（常规路径）
    for model_cls, processor_cls in PROCESSOR_MAPPING.items():
        if model_cls.__name__ in getattr(hf_config, "architectures", []):
            return processor_cls(hf_config, server_args, processor, transport_mode)

    # 2) 兜底：按已知架构名做字符串匹配 + 延迟导入
    arch_list = getattr(hf_config, "architectures", []) or []
    arch_name = arch_list[0] if arch_list else ""

    # Qwen2.5-VL (如：Qwen2_5_VLForConditionalGeneration)
    if arch_name.startswith("Qwen2_5_VL"):
        mod = import_module("sglang.srt.multimodal.processors.qwen_vl")
        proc_cls = getattr(mod, "QwenVLImageProcessor")
        return proc_cls(hf_config, server_args, processor, transport_mode)

    # Gemma-3 (如：Gemma3ForConditionalGeneration)
    if arch_name.startswith("Gemma3"):
        mod = import_module("sglang.srt.multimodal.processors.gemma3")
        proc_cls = getattr(mod, "Gemma3ImageProcessor")
        return proc_cls(hf_config, server_args, processor, transport_mode)

    # 3) 仍未匹配到，给出清晰报错
    raise ValueError(
        f"No processor registered for architecture: {hf_config.architectures}.\n"
        f"Registered architectures: {[model_cls.__name__ for model_cls in PROCESSOR_MAPPING.keys()]}"
    )
