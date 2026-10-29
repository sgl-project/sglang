from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import MultimodalDataItem


class TokenSpaceServingSpec:
    """Serving declarations for one model family, read only by the sglang wrapper.

    Parallel to `TokenSpaceProcessStrategy`: each has its own registry by
    architecture and neither references the other. A model without a registered
    spec is served with these defaults.
    """

    # hf_config.architectures names this spec serves.
    models: tuple[str, ...] = ()
    # Media IO threads when neither --mm-io-worker-num nor SGLANG_IO_WORKERS is set.
    auto_mm_io_worker_num: int = 4

    def __init__(self, hf_config):
        self.hf_config = hf_config

    def build_mm_tokens(self, processor):
        """The prompt's media markers that request validation reads, or None.

        `/v1/decisions` rejects image requests when the processor exposes none.
        """
        return None

    def split_mm_items(
        self, mm_items: list["MultimodalDataItem"]
    ) -> list["MultimodalDataItem"]:
        """Split each modality's item into one item per media, for per-media cache keys.

        Runs on processor worker threads, so it must not mutate the spec. The default
        is the legacy processor's split, which finds media boundaries by field name.
        """
        from sglang.srt.managers.mm_utils import get_new_expanded_mm_items

        return get_new_expanded_mm_items(mm_items)
