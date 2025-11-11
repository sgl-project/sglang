"""Lightweight request types and enums from schedule_batch.py."""

from __future__ import annotations

import enum
from typing import List, Union


class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def to_json(self):
        raise NotImplementedError()


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISHED_MATCHED_REGEX(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__(is_error=True)
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


class RequestStage(str, enum.Enum):
    # Tokenizer
    TOKENIZE = "tokenize"
    TOKENIZER_DISPATCH = "dispatch"

    # DP controller
    DC_DISPATCH = "dc_dispatch"

    # common/non-disaggregation
    PREFILL_WAITING = "prefill_waiting"
    REQUEST_PROCESS = "request_process"
    DECODE_LOOP = "decode_loop"
    PREFILL_FORWARD = "prefill_forward"
    PREFILL_CHUNKED_FORWARD = "chunked_prefill"

    # disaggregation prefill
    PREFILL_PREPARE = "prefill_prepare"
    PREFILL_BOOTSTRAP = "prefill_bootstrap"
    PREFILL_TRANSFER_KV_CACHE = "prefill_transfer_kv_cache"

    # disaggregation decode
    DECODE_PREPARE = "decode_prepare"
    DECODE_BOOTSTRAP = "decode_bootstrap"
    DECODE_WAITING = "decode_waiting"
    DECODE_TRANSFERRED = "decode_transferred"
    DECODE_FAKE_OUTPUT = "fake_output"
    DECODE_QUICK_FINISH = "quick_finish"
