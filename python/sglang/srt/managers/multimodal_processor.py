# TODO: also move pad_input_ids into this module
import importlib
import inspect
import logging
import pkgutil

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
)
from sglang.srt.runtime_context import get_model
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

PROCESSOR_MAPPING = {}
# Keep import failures around so that a later lookup can explain why a
# processor was not registered. A failed import is intentionally non-fatal in
# ``import_processors`` because not every optional processor dependency is
# available in every installation.
PROCESSOR_IMPORT_ERRORS = {}


def import_processors(package_name: str, overwrite: bool = False):
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                PROCESSOR_IMPORT_ERRORS[name] = e
                logger.warning(f"Ignore import error when loading {name}: {e}")
                continue

            PROCESSOR_IMPORT_ERRORS.pop(name, None)

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
                    if overwrite:
                        for model_cls, processor_cls in PROCESSOR_MAPPING.items():
                            if model_cls.__name__ == arch.__name__:
                                del PROCESSOR_MAPPING[model_cls]
                                break

                    PROCESSOR_MAPPING[arch] = cls


def get_mm_processor_cls(hf_config, model_config=None):
    """The class :func:`get_mm_processor` would instantiate, or ``None`` when the
    architecture has no registered processor."""
    model_impl = str(get_model().model_impl).lower()
    uses_transformers_backend = model_impl == "transformers"

    if model_impl == "auto" and model_config is not None:
        from sglang.srt.model_loader.utils import get_resolved_model_impl

        uses_transformers_backend = (
            get_resolved_model_impl(model_config) == ModelImpl.TRANSFORMERS
        )

    for model_cls, processor_cls in PROCESSOR_MAPPING.items():
        if model_cls.__name__ not in hf_config.architectures:
            continue
        if not uses_transformers_backend or getattr(
            processor_cls, "supports_transformers_backend", False
        ):
            return processor_cls

    if uses_transformers_backend:
        from sglang.srt.multimodal.processors.transformers_auto import (
            TransformersAutoMultimodalProcessor,
        )

        return TransformersAutoMultimodalProcessor

    return None


def get_mm_processor(
    hf_config,
    server_args: ServerArgs,
    processor,
    transport_mode,
    model_config=None,
    **kwargs,
) -> BaseMultimodalProcessor:
    processor_cls = get_mm_processor_cls(hf_config, model_config)
    if processor_cls is None:
        failed_imports = "\n".join(
            f"  {module_name}: {type(error).__name__}: {error}"
            for module_name, error in PROCESSOR_IMPORT_ERRORS.items()
        )
        failed_imports_message = (
            f"\nFailed processor imports:\n{failed_imports}" if failed_imports else ""
        )
        raise ValueError(
            f"No processor registered for architecture: {hf_config.architectures}.\n"
            f"Registered architectures: {[model_cls.__name__ for model_cls in PROCESSOR_MAPPING.keys()]}"
            f"{failed_imports_message}"
        )
    return processor_cls(hf_config, server_args, processor, transport_mode, **kwargs)
