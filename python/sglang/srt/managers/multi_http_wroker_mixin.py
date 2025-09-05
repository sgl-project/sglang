from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.detokenizer_manager import DetokenizerManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class MultiHttpWorkerTokenizerMixin:
    def maybe_init_multi_http_worker(self: TokenizerManager, is_sub_tokenizer: bool):
        """Init multi http worker related settings as a sub tokenizer in multi tokenizer manager mode"""

        self.is_sub_tokenizer = is_sub_tokenizer

        if not is_sub_tokenizer:
            return


class MultiHttpWorkerDetokenizerMixin:
    pass


class MultiHttpWorkerCollector:
    pass
