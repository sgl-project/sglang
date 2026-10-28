from typing import Optional


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
    # Device every medium of a video request preprocesses and decodes on, unless
    # mm_process_config sets video_preprocessing_device; None keeps the defaults.
    video_preprocessing_device: Optional[str] = None

    def __init__(self, hf_config):
        self.hf_config = hf_config

    def build_mm_tokens(self, processor):
        """The prompt's media markers that request validation reads, or None.

        `/v1/decisions` rejects image requests when the processor exposes none.
        """
        return None
