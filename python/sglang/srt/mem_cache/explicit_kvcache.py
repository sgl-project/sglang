from __future__ import annotations

import logging

from sglang.srt.managers.io_struct import ExKVCacheParams

logger = logging.getLogger(__name__)


class ExKVCache:
    def __init__(self, params: ExKVCacheParams):
        self.cached_token_count = params.cached_token_count

    def params(self) -> ExKVCacheParams:
        return ExKVCacheParams(cached_token_count=self.cached_token_count)
