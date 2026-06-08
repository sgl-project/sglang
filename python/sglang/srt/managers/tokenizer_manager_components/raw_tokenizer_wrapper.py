from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sglang.srt.managers.async_dynamic_batch_tokenizer import AsyncDynamicbatchTokenizer


@dataclass(slots=True, kw_only=True)
class RawTokenizerWrapper:
    tokenizer: Optional[Any] = None
    processor: Optional[Any] = None
    mm_processor: Optional[Any] = None
    async_dynamic_batch_tokenizer: Optional[AsyncDynamicbatchTokenizer] = None
