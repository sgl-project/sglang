from collections.abc import Mapping
from typing import Any, Iterable

from sglang.srt.managers.schedule_batch import MultimodalDataItem


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
    ) -> None:
        super().__init__(values or {})
        self.mm_items = list(mm_items)
        if not self.mm_items:
            raise ValueError("EncoderPreprocessOutput requires at least one item")


def get_encoder_preprocessed_items(
    processor_output: Mapping[str, Any],
) -> list[MultimodalDataItem] | None:
    if isinstance(processor_output, EncoderPreprocessOutput):
        return processor_output.mm_items
    return None
