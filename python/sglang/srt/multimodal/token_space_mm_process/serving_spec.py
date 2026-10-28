from typing import Optional


class TokenSpaceServingSpec:
    """Serving declarations for one model family, read only by the sglang wrapper.

    Parallel to `TokenSpaceProcessStrategy`: each has its own registry by
    architecture and neither references the other. A model without a registered
    spec is served with these defaults.
    """

    # hf_config.architectures names this spec serves.
    models: tuple[str, ...] = ()
    # Media IO threads when --mm-io-worker-num is unset; None keeps the processor default.
    auto_mm_io_worker_num: Optional[int] = None
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
