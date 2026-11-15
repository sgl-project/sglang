"""Disaggregation mode, kept free of heavy imports for request and IPC code."""

from enum import Enum

from sglang.srt.runtime_context import get_disagg

FAKE_BOOTSTRAP_HOST = "2.2.2.2"


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

    @staticmethod
    def to_engine_type(mode: str) -> str:
        if mode == DisaggregationMode.PREFILL.value:
            return "prefill"
        elif mode == DisaggregationMode.DECODE.value:
            return "decode"
        return "unified"


def current_disaggregation_mode() -> DisaggregationMode:
    """The PD role now; a runtime role switch overrides it in the disagg config."""
    return DisaggregationMode(get_disagg().disaggregation_mode)
