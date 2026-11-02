"""``custom_params`` keys of a split-document window (Unlimited-OCR v2).

The serving layer writes them; the scheduler's document reuse and the page
guard in ``custom_logit_processor`` read them. A leaf module so neither reader
imports the other.
"""

KEY_DOC_ID = "uocr_doc_id"
KEY_KEEP = "uocr_doc_keep"
KEY_REUSE = "uocr_doc_reuse"
# Leading pages of this window's prompt the previous window already
# transcribed; the rewrite drops them. Zero once the plan is contiguous.
KEY_OVERLAP = "uocr_doc_overlap"
# Global document index of this window's first page. Added to the RoPE page
# axis so a page has the same parity in every window that contains it.
KEY_PAGE_BASE = "uocr_doc_page_base"
# Trailing prompt pages the window reads but does not transcribe: the next
# window's first page, so the last own page's final image block holds the
# next page's opening as in a single-pass run. The window stops on its marker.
KEY_LOOKAHEAD = "uocr_doc_lookahead"
