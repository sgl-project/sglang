from typing import NamedTuple, Optional

from sglang.srt.model_executor.runner_utils.capture_mode import get_capture_dsa_variant


class IndexerCaptureOptions(NamedTuple):
    select_all: bool
    bypass_candidates: bool


def resolve_indexer_capture_options(
    compress_ratio: int, capture_variant: Optional[str]
) -> IndexerCaptureOptions:
    select_all = capture_variant == "candidate_all" or (
        capture_variant == "candidate_c2_all" and compress_ratio == 2
    )
    bypass_candidates = capture_variant in (
        "candidate_all",
        "candidate_c2_all",
        "candidate_unfiltered",
    )
    return IndexerCaptureOptions(select_all, bypass_candidates)


def get_indexer_capture_options(compress_ratio: int) -> IndexerCaptureOptions:
    return resolve_indexer_capture_options(compress_ratio, get_capture_dsa_variant())
