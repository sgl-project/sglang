"""Per-model admission checks for multimodal inputs on scheduler ranks."""

from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

MultimodalInputValidator = Callable[[Any], None]


@runtime_checkable
class ValidatesMultimodalInputs(Protocol):
    """A model class that rejects malformed inputs before they reach its encoder.

    The returned validator takes `MultimodalInputs`/`MultimodalProcessorOutput`
    and raises on malformed metadata. It must read only broadcast request
    metadata, so every scheduler rank reaches the same verdict without a
    collective.
    """

    @classmethod
    def build_mm_input_validator(
        cls, hf_config: Any
    ) -> Optional[MultimodalInputValidator]: ...


def resolve_mm_input_validator(
    model_config: "ModelConfig",
) -> Optional[MultimodalInputValidator]:
    if not model_config.is_multimodal:
        return None
    from sglang.srt.model_loader.utils import get_model_architecture

    model_cls, _ = get_model_architecture(model_config)
    if not issubclass(model_cls, ValidatesMultimodalInputs):
        return None
    return model_cls.build_mm_input_validator(model_config.hf_config)
