"""Document serving policy and CLI declarations for Unlimited-OCR v2."""

import msgspec

from sglang.srt.arg_groups.arg_utils import A

DEFAULT_LOOKAHEAD_PAGES = 1
MAX_PAGE_PROMPT_TOKENS = 1 + 273 + 1760
WINDOW_SLACK_TOKENS = 1024
FALLBACK_CONTEXT_LENGTH = 131072


class UocrDocumentConfig(msgspec.Struct, frozen=True):
    _NS_PATH = "mm"

    uocr_doc_split: A[
        bool,
        "Unlimited-OCR v2: opt into approximate page-window document processing "
        "with KV reuse. Requires a single data-parallel rank.",
    ] = False
    uocr_doc_page_window: A[
        int,
        "Unlimited-OCR v2: maximum pages a split window transcribes, excluding "
        "lookahead pages. Applies to every image mode. Must be at least 1.",
    ] = 1
    uocr_max_page_tokens: A[
        int,
        "Unlimited-OCR v2: output-token limit per page. The built-in page guard "
        "enforces it without --enable-custom-logit-processor. Also sizes the "
        "default context of a split server. 0 disables the length limit.",
    ] = 0

    def __post_init__(self):
        if self.uocr_doc_page_window < 1:
            raise ValueError("--uocr-doc-page-window must be at least 1")
        if self.uocr_max_page_tokens < 0:
            raise ValueError("--uocr-max-page-tokens must be non-negative")

    @classmethod
    def from_args(cls, args):
        return cls(
            uocr_doc_split=args.uocr_doc_split,
            uocr_doc_page_window=args.uocr_doc_page_window,
            uocr_max_page_tokens=args.uocr_max_page_tokens,
        )

    def default_context_length(self) -> int:
        if not self.uocr_doc_split or self.uocr_max_page_tokens == 0:
            return FALLBACK_CONTEXT_LENGTH
        needed = (
            (self.uocr_doc_page_window + DEFAULT_LOOKAHEAD_PAGES)
            * MAX_PAGE_PROMPT_TOKENS
            + self.uocr_doc_page_window * self.uocr_max_page_tokens
            + WINDOW_SLACK_TOKENS
        )
        return min(needed, FALLBACK_CONTEXT_LENGTH)
