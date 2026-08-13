import hashlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import MultimodalDataItem

LOCAL_PREPROCESSED_KEY = "encoder_local_preprocessed"


@dataclass(frozen=True)
class EncoderMediaProcessorConfig:
    """Optional model-declared media loading behavior for encoder mode."""

    image_decode_mode: bool | str = False
    preserve_media_metadata: bool = False


@runtime_checkable
class EncoderMediaProcessorConfigProvider(Protocol):
    """Model contract for optional encoder-side media preprocessing."""

    encoder_media_processor_config: EncoderMediaProcessorConfig


def resolve_encoder_media_processor_config(
    model: object,
) -> EncoderMediaProcessorConfig:
    """Resolve a model-declared capability without model-name dispatch."""
    if isinstance(model, EncoderMediaProcessorConfigProvider):
        return model.encoder_media_processor_config
    return EncoderMediaProcessorConfig()


def hash_raw_encoder_item(value: Any) -> int:
    """Hash raw CPU media including layout metadata, before owner materialization."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    elif not isinstance(value, np.ndarray):
        from PIL import Image

        if not isinstance(value, Image.Image):
            raise TypeError(f"Unsupported raw encoder item: {type(value)}")
        value = np.asarray(value)

    value = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(value.dtype.str.encode())
    hasher.update(repr(value.shape).encode())
    hasher.update(memoryview(value))
    return int.from_bytes(hasher.digest()[:8], byteorder="big", signed=False)


class EncoderPreprocessOutput(dict):
    """Processor output that preserves one feature object per multimodal item.

    Most Hugging Face processors concatenate every image into one tensor before
    the vision model decides its encoder-DP assignment.  A model-specific
    ``preprocess_mm_for_encoder`` hook can return this mapping instead, allowing
    the encoder to carry raw or partially processed items to the model.  The
    model can then materialize only the items owned by its local vision rank.

    The mapping remains compatible with existing encoder metadata helpers;
    ``mm_items`` is an out-of-band, per-item representation used only by the
    encoder forward/cache paths.
    """

    def __init__(
        self,
        values: Mapping[str, Any] | None = None,
        *,
        mm_items: Iterable[MultimodalDataItem],
        item_sizes: Sequence[int] | None = None,
        materialize_local_items: (
            Callable[[list[MultimodalDataItem]], Sequence[torch.Tensor]] | None
        ) = None,
    ) -> None:
        super().__init__(values or {})
        self.mm_items = list(mm_items)
        if not self.mm_items:
            raise ValueError("EncoderPreprocessOutput requires at least one item")
        self.item_sizes = list(item_sizes) if item_sizes is not None else None
        if self.item_sizes is not None and len(self.item_sizes) != len(self.mm_items):
            raise ValueError("Encoder preprocess item_sizes must match mm_items")
        self.materialize_local_items = materialize_local_items

    def local_item_indices(self, rank: int, world_size: int) -> list[int]:
        """Return the same size-balanced owner assignment used by vision DP."""
        if self.materialize_local_items is None:
            return []
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(
                f"Invalid encoder preprocess rank {rank} for world size {world_size}"
            )
        if world_size == 1:
            return list(range(len(self.mm_items)))
        if self.item_sizes is None:
            raise ValueError(
                "Owner-side encoder preprocessing requires per-item load sizes"
            )

        from sglang.srt.multimodal.mm_utils import get_dp_encoder_lb_assignment

        shuffled, counts, _ = get_dp_encoder_lb_assignment(self.item_sizes, world_size)
        start = sum(counts[:rank])
        return shuffled[start : start + counts[rank]]

    def materialize_for_rank(self, rank: int, world_size: int) -> None:
        """Materialize only this vision-DP rank's items in-place."""
        indices = self.local_item_indices(rank, world_size)
        if not indices:
            return
        items = [self.mm_items[index] for index in indices]
        materialize = self.materialize_local_items
        assert materialize is not None
        features = list(materialize(items))
        if len(features) != len(items):
            raise ValueError(
                "Encoder local materializer must return one feature per item"
            )
        for item, feature in zip(items, features):
            item.feature = feature
            item.model_specific_data[LOCAL_PREPROCESSED_KEY] = True


def get_encoder_preprocessed_items(
    processor_output: Mapping[str, Any],
) -> list[MultimodalDataItem] | None:
    if isinstance(processor_output, EncoderPreprocessOutput):
        return processor_output.mm_items
    return None


def invoke_encoder_preprocessor(
    preprocessor,
    mm_data,
    modality,
    config,
    **available_context,
):
    """Call a model hook with only the optional context it declares.

    Existing hooks keep their three-argument contract. New model integrations
    can request shared processor state or backend policy by adding named
    keyword-only parameters, without requiring encode-server model branches.
    """
    parameters = inspect.signature(preprocessor).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    context = (
        available_context
        if accepts_kwargs
        else {
            name: value
            for name, value in available_context.items()
            if name in parameters
        }
    )
    return preprocessor(mm_data, modality, config, **context)
