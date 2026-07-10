"""
Evaluation utilities for the OCR benchmark (olmOCR-bench test classes).

Implements:
  - text_presence      : short text segment must be present in OCR output
  - text_absence       : text (headers/footers/page numbers) must NOT appear
  - natural_reading_order : two text spans must appear in correct relative order
  - table_accuracy     : cell values with correct neighbor relationships (Markdown + HTML)
  - math_formula_accuracy : LaTeX key-symbol token matching (simplified; no KaTeX/playwright)

Also provides:
  - normalized_edit_distance() for OmniDocBench-style text quality measurement
  - aggregate_results() / print_results_table() for summary reporting
"""

import re
import unicodedata
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Dict, List, Optional

# ── Unicode normalization ─────────────────────────────────────────────────────

_HYPHEN_RE = re.compile(
    r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D]"
)
_DQUOTE_RE = re.compile(
    r"[\u00AB\u00BB\u201C\u201D\u201E\u201F\u2033\u2036\u276E\u276F\u3003\uFF02]"
)
_SQUOTE_RE = re.compile(
    r"[\u2018\u2019\u201A\u201B\u2032\u2035\u2039\u203A\u2C8D\uFF07]"
)
_MARKDOWN_RE = re.compile(r"(\*{1,3}|_{1,3}|`{1,3}|~~|#{1,6}\s?)")


def normalize_text(text: str) -> str:
    """Apply olmOCR-bench standard Unicode normalization."""
    text = unicodedata.normalize("NFC", text)
    text = _HYPHEN_RE.sub("-", text)
    text = _DQUOTE_RE.sub('"', text)
    text = _SQUOTE_RE.sub("'", text)
    return text


def strip_markdown(text: str) -> str:
    """Remove Markdown syntax markers for soft matching."""
    return _MARKDOWN_RE.sub("", text)


# ── Matching helpers ──────────────────────────────────────────────────────────


def fuzzy_contains(needle: str, haystack: str, threshold: float = 0.85) -> bool:
    """Check if needle appears in haystack using fuzzy sliding-window matching."""
    needle = normalize_text(strip_markdown(needle).strip())
    haystack = normalize_text(strip_markdown(haystack))

    # Fast exact check first
    if needle.lower() in haystack.lower():
        return True

    n = len(needle)
    if n == 0:
        return True

    step = max(1, n // 4)
    for i in range(0, max(1, len(haystack) - n + 1), step):
        window = haystack[i : i + n]
        ratio = SequenceMatcher(None, needle.lower(), window.lower()).ratio()
        if ratio >= threshold:
            return True
    return False


def exact_contains(needle: str, haystack: str, case_sensitive: bool = True) -> bool:
    """Check if needle appears exactly in haystack (after normalization)."""
    needle = normalize_text(strip_markdown(needle).strip())
    haystack = normalize_text(strip_markdown(haystack))
    if not case_sensitive:
        return needle.lower() in haystack.lower()
    return needle in haystack


def _get_words_slice(text: str, first_n: Optional[int], last_n: Optional[int]) -> str:
    """Return the first or last N whitespace-separated words of text."""
    if first_n is None and last_n is None:
        return text
    words = text.split()
    if first_n is not None:
        return " ".join(words[:first_n])
    if last_n is not None:
        return " ".join(words[-last_n:])
    return text


# ── olmOCR-bench test evaluators ─────────────────────────────────────────────


def eval_text_presence(test: dict, ocr_output: str) -> bool:
    """Evaluate a present/text_presence test: target text must appear in OCR output.

    Supports olmOCR-bench flat schema (max_diffs, first_n, last_n) and the
    legacy nested-position schema.
    """
    needle = test.get("text", "")
    max_diffs = test.get("max_diffs", 0)
    case_sensitive = test.get("case_sensitive", True)
    first_n = test.get("first_n", None)
    last_n = test.get("last_n", None)

    haystack = _get_words_slice(ocr_output, first_n, last_n)

    if max_diffs == 0:
        return exact_contains(needle, haystack, case_sensitive=case_sensitive)
    # Fuzzy: compute similarity threshold from allowed diffs
    n = max(1, len(needle))
    threshold = max(0.6, (n - max_diffs) / n)
    return fuzzy_contains(needle, haystack, threshold=threshold)


def eval_text_absence(test: dict, ocr_output: str) -> bool:
    """Evaluate an absent/text_absence test: target text must NOT appear in OCR output.

    Supports olmOCR-bench flat schema (max_diffs, first_n, last_n) and the
    legacy nested-position schema.
    """
    needle = test.get("text", "")
    max_diffs = test.get("max_diffs", 0)
    case_sensitive = test.get("case_sensitive", False)
    first_n = test.get("first_n", None)
    last_n = test.get("last_n", None)

    haystack = _get_words_slice(ocr_output, first_n, last_n)

    if max_diffs == 0:
        present = exact_contains(needle, haystack, case_sensitive=case_sensitive)
    else:
        n = max(1, len(needle))
        threshold = max(0.6, (n - max_diffs) / n)
        present = fuzzy_contains(needle, haystack, threshold=threshold)
    return not present


def eval_reading_order(test: dict, ocr_output: str) -> bool:
    """Evaluate an order/natural_reading_order test: 'before' text must precede 'after'.

    Uses max_diffs to choose exact vs fuzzy matching.
    """
    before_text = test.get("before", "")
    after_text = test.get("after", "")
    max_diffs = test.get("max_diffs", 0)
    fuzzy = max_diffs > 0

    output_norm = normalize_text(strip_markdown(ocr_output))

    def find_approx_pos(needle: str, text: str) -> int:
        needle = normalize_text(strip_markdown(needle).strip())
        n = len(needle)
        if n == 0:
            return 0
        # Exact first
        idx = text.lower().find(needle.lower())
        if idx != -1:
            return idx
        # Fuzzy fallback
        step = max(1, n // 4)
        best_pos, best_ratio = -1, 0.0
        for i in range(0, max(1, len(text) - n + 1), step):
            window = text[i : i + n]
            ratio = SequenceMatcher(None, needle.lower(), window.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i
        return best_pos if best_ratio >= 0.80 else -1

    if fuzzy:
        pos_before = find_approx_pos(before_text, output_norm)
        pos_after = find_approx_pos(after_text, output_norm)
    else:
        b = normalize_text(strip_markdown(before_text).strip())
        a = normalize_text(strip_markdown(after_text).strip())
        pos_before = output_norm.find(b)
        pos_after = output_norm.find(a)

    if pos_before == -1 or pos_after == -1:
        return False
    return pos_before < pos_after


# ── Table parsing helpers ─────────────────────────────────────────────────────


def _parse_markdown_table(text: str) -> List[List[str]]:
    """Parse a Markdown table into a list-of-rows, each row a list of cells."""
    rows: List[List[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "|" not in stripped:
            continue
        # Skip separator rows like |---|---|
        if re.match(r"^\|?[-:| ]+\|?$", stripped):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells:
            rows.append(cells)
    return rows


class _HTMLTableParser(HTMLParser):
    """Minimal HTML table parser (does not handle colspan/rowspan)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._current_row: List[str] = []
        self._current_cell: str = ""
        self._in_cell: bool = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = ""

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._current_row.append(self._current_cell.strip())
            self._in_cell = False
        elif tag == "tr" and self._current_row:
            self.rows.append(self._current_row)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell += data


def _extract_tables(ocr_output: str) -> List[List[List[str]]]:
    """Extract all tables (HTML + Markdown) from OCR output."""
    tables: List[List[List[str]]] = []

    # HTML tables
    for match in re.finditer(
        r"<table[^>]*>.*?</table>", ocr_output, re.DOTALL | re.IGNORECASE
    ):
        parser = _HTMLTableParser()
        parser.feed(match.group(0))
        if parser.rows:
            tables.append(parser.rows)

    # Markdown tables
    md_pattern = re.compile(
        r"(\|[^\n]+\|\n(?:\|[-:| ]+\|\n)?(?:\|[^\n]+\|?\n?)+)", re.MULTILINE
    )
    for match in md_pattern.finditer(ocr_output):
        rows = _parse_markdown_table(match.group(0))
        if len(rows) >= 2:  # At least header + one data row
            tables.append(rows)

    return tables


def eval_table_flat(test: dict, ocr_output: str) -> bool:
    """
    Evaluate a flat-schema olmOCR-bench 'table' test.

    Schema fields:
      cell          – text of the target cell to locate
      up/down/left/right – expected text of the neighbor in that direction (null = skip)
      top_heading   – expected column heading (row 0, same column)
      left_heading  – expected row heading (column 0, same row)

    All non-null fields must fuzzy-match for the test to pass.

    Supports:
      1. Structured tables (HTML with <tr>/<td>, Markdown)  – full positional check
      2. Flat <table>content</table> blocks (DeepSeek-OCR-2 format) – text presence
         fallback when no rows are parseable from HTML
    """
    cell_text = test.get("cell", "")
    directional = {
        "up": test.get("up"),
        "down": test.get("down"),
        "left": test.get("left"),
        "right": test.get("right"),
    }
    top_heading = test.get("top_heading")
    left_heading = test.get("left_heading")
    checks = [(d, v) for d, v in directional.items() if v is not None]

    # ── 1. Structured tables (HTML <tr>/<td> or Markdown) ────────────────────
    for rows in _extract_tables(ocr_output):
        if not rows:
            continue
        header_row = rows[0]
        for r_idx, row in enumerate(rows):
            for c_idx, cell in enumerate(row):
                if not fuzzy_contains(cell_text, cell, threshold=0.85):
                    continue
                # Verify all directional neighbors
                all_ok = True
                for direction, expected in checks:
                    if direction == "up" and r_idx > 0:
                        prev_row = rows[r_idx - 1]
                        nb = prev_row[c_idx] if c_idx < len(prev_row) else ""
                        if not fuzzy_contains(expected, nb, threshold=0.85):
                            all_ok = False
                            break
                    elif direction == "down" and r_idx < len(rows) - 1:
                        next_row = rows[r_idx + 1]
                        nb = next_row[c_idx] if c_idx < len(next_row) else ""
                        if not fuzzy_contains(expected, nb, threshold=0.85):
                            all_ok = False
                            break
                    elif direction == "left" and c_idx > 0:
                        nb = row[c_idx - 1]
                        if not fuzzy_contains(expected, nb, threshold=0.85):
                            all_ok = False
                            break
                    elif direction == "right" and c_idx < len(row) - 1:
                        nb = row[c_idx + 1]
                        if not fuzzy_contains(expected, nb, threshold=0.85):
                            all_ok = False
                            break
                    else:
                        all_ok = False
                        break  # expected neighbor out of bounds
                if not all_ok:
                    continue
                # Verify top_heading (column header, row 0)
                if top_heading is not None:
                    th = header_row[c_idx] if c_idx < len(header_row) else ""
                    if not fuzzy_contains(top_heading, th, threshold=0.85):
                        continue
                # Verify left_heading (first cell of same row)
                if left_heading is not None:
                    lh = row[0] if row else ""
                    if not fuzzy_contains(left_heading, lh, threshold=0.85):
                        continue
                return True

    # ── 2. Flat <table>…</table> fallback (DeepSeek-OCR-2 format) ────────────
    # The model emits <table>AllCellsConcatenated</table> without <tr>/<td>.
    # Fall back to checking that cell + all non-null headings/neighbors appear
    # somewhere within the same table block.
    flat_blocks = re.findall(
        r"<table[^>]*>(.*?)</table>", ocr_output, re.DOTALL | re.IGNORECASE
    )
    for flat_text in flat_blocks:
        # Strip any residual HTML tags (e.g. inline <br>) and bounding-box annotations
        flat_clean = re.sub(r"<[^>]+>", " ", flat_text)
        flat_clean = re.sub(r"\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]", " ", flat_clean)
        if not fuzzy_contains(cell_text, flat_clean, threshold=0.85):
            continue
        if top_heading is not None and not fuzzy_contains(
            top_heading, flat_clean, threshold=0.85
        ):
            continue
        if left_heading is not None and not fuzzy_contains(
            left_heading, flat_clean, threshold=0.85
        ):
            continue
        all_ok = all(
            fuzzy_contains(expected, flat_clean, threshold=0.85)
            for _, expected in checks
        )
        if all_ok:
            return True

    return False


def eval_baseline(test: dict, ocr_output: str) -> bool:
    """
    Evaluate a baseline/sanity test.

    When check_disallowed_characters is False (the common case), always passes.
    When True, checks that the OCR output contains no non-printable control
    characters (excluding normal whitespace).
    """
    if not test.get("check_disallowed_characters", False):
        return True
    for ch in ocr_output:
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in ("\n", "\t", "\r", " "):
            return False
    return bool(ocr_output.strip())  # also fail if completely empty


def eval_table_accuracy(test: dict, ocr_output: str) -> bool:
    """
    Evaluate a table_accuracy test (legacy nested schema).

    Checks that a cell with ``cell_text`` exists in a table and that its
    neighbor in the specified ``relationship`` (above/below/left/right)
    contains ``neighbor_text``.
    """
    cell_text = test.get("cell_text", "")
    neighbor_text = test.get("neighbor_text", "")
    relationship = test.get("relationship", "")

    for rows in _extract_tables(ocr_output):
        for r_idx, row in enumerate(rows):
            for c_idx, cell in enumerate(row):
                if not fuzzy_contains(cell_text, cell, threshold=0.88):
                    continue
                # Found the target cell — check its neighbor
                if relationship == "above" and r_idx > 0:
                    prev_row = rows[r_idx - 1]
                    nb = prev_row[c_idx] if c_idx < len(prev_row) else ""
                    if fuzzy_contains(neighbor_text, nb, threshold=0.88):
                        return True
                elif relationship == "below" and r_idx < len(rows) - 1:
                    next_row = rows[r_idx + 1]
                    nb = next_row[c_idx] if c_idx < len(next_row) else ""
                    if fuzzy_contains(neighbor_text, nb, threshold=0.88):
                        return True
                elif relationship == "left" and c_idx > 0:
                    nb = row[c_idx - 1]
                    if fuzzy_contains(neighbor_text, nb, threshold=0.88):
                        return True
                elif relationship == "right" and c_idx < len(row) - 1:
                    nb = row[c_idx + 1]
                    if fuzzy_contains(neighbor_text, nb, threshold=0.88):
                        return True
    return False


# Math formula evaluation ─────────────────────────────────────────────────────

_MATH_REGION_RE = re.compile(
    r"\$\$[\s\S]*?\$\$"  # $$ block $$
    r"|\$[^$\n]+?\$"  # inline $...$
    r"|\\?\\\[[\s\S]*?\\?\\\]"  # \[...\]
    r"|\\?\\\([\s\S]*?\\?\\\)",  # \(...\)
)
_LATEX_TOKEN_RE = re.compile(r"\\[a-zA-Z]+|[a-zA-Z0-9]|[+\-*/=<>^_{}()\[\]]")


def eval_math_formula_accuracy(test: dict, ocr_output: str) -> bool:
    """
    Simplified math formula accuracy check.

    Checks that the key symbol tokens from a LaTeX expression appear in
    math-delimited regions of the OCR output.

    Note: Full KaTeX bounding-box matching (as used by the official
    olmOCR-bench) requires playwright and is not performed here.
    """
    latex = (test.get("math") or test.get("latex") or "").strip()
    if not latex:
        return False

    math_text = " ".join(m.group(0) for m in _MATH_REGION_RE.finditer(ocr_output))
    if not math_text:
        # Fall back to full output if no delimited regions found
        math_text = ocr_output

    tokens = _LATEX_TOKEN_RE.findall(latex)
    if not tokens:
        return False

    present = sum(1 for t in tokens if t in math_text)
    return present / len(tokens) >= 0.70


# ── Main dispatcher ───────────────────────────────────────────────────────────

_TEST_EVALUATORS = {
    # olmOCR-bench flat-JSONL type names
    "present": eval_text_presence,
    "absent": eval_text_absence,
    "order": eval_reading_order,
    "math": eval_math_formula_accuracy,
    "table": eval_table_flat,
    "baseline": eval_baseline,
    # Legacy / aliased names
    "text_presence": eval_text_presence,
    "text_absence": eval_text_absence,
    "natural_reading_order": eval_reading_order,
    "table_accuracy": eval_table_accuracy,
    "math_formula_accuracy": eval_math_formula_accuracy,
}


def evaluate_olmocr_tests(tests: List[dict], ocr_output: str) -> List[dict]:
    """Run all olmOCR-bench unit tests against OCR output; return per-test results."""
    results: List[dict] = []
    for test in tests:
        test_type = test.get("type", "")
        evaluator = _TEST_EVALUATORS.get(test_type)
        if evaluator is None:
            results.append(
                {
                    "type": test_type,
                    "passed": False,
                    "error": f"Unknown type: {test_type}",
                }
            )
            continue
        try:
            passed = bool(evaluator(test, ocr_output))
        except Exception as exc:
            results.append({"type": test_type, "passed": False, "error": str(exc)})
            continue
        results.append({"type": test_type, "passed": passed})
    return results


# ── Aggregation & reporting ───────────────────────────────────────────────────


def aggregate_results(split_name: str, sample_results: List[dict]) -> dict:
    """Aggregate per-sample results into split-level statistics."""
    by_type: Dict[str, Dict[str, int]] = {}
    total_passed = 0
    total_tests = 0
    error_count = 0

    for sample in sample_results:
        if "error" in sample and not sample.get("test_results"):
            error_count += 1
            continue
        for tr in sample.get("test_results", []):
            t = tr.get("type", "unknown")
            by_type.setdefault(t, {"passed": 0, "total": 0})
            by_type[t]["total"] += 1
            total_tests += 1
            if tr.get("passed"):
                by_type[t]["passed"] += 1
                total_passed += 1

    type_scores = {
        t: round(100.0 * v["passed"] / v["total"], 1) if v["total"] > 0 else 0.0
        for t, v in by_type.items()
    }
    overall = round(100.0 * total_passed / total_tests, 1) if total_tests > 0 else 0.0

    return {
        "split": split_name,
        "total_samples": len(sample_results),
        "error_samples": error_count,
        "total_tests": total_tests,
        "total_passed": total_passed,
        "overall_score": overall,
        "by_type": type_scores,
        "by_type_counts": by_type,
    }


def print_results_table(all_results: Dict[str, dict]) -> None:
    """Print a formatted results summary table to stdout."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  olmOCR-bench Results Summary  (DeepSeek-OCR-2 via sglang)")
    print(sep)
    print(f"{'Split':<22} {'Tests':>8} {'Passed':>8} {'Score':>8}")
    print("-" * 50)
    splits = list(all_results.keys())
    scores = []
    for split in splits:
        r = all_results[split]
        score = r.get("overall_score", 0.0)
        scores.append(score)
        print(
            f"{split:<22} {r['total_tests']:>8} {r['total_passed']:>8} {score:>7.1f}%"
        )
    print("-" * 50)

    total_tests = sum(r["total_tests"] for r in all_results.values())
    total_passed = sum(r["total_passed"] for r in all_results.values())
    overall = round(100.0 * total_passed / total_tests, 1) if total_tests > 0 else 0.0
    mean_score = round(sum(scores) / len(scores), 1) if scores else 0.0
    print(f"{'TOTAL':<22} {total_tests:>8} {total_passed:>8} {overall:>7.1f}%")
    print(f"{'Mean across splits':<22} {'':>17} {mean_score:>7.1f}%")
    print(sep)

    # Per-type breakdown
    all_types: set = set()
    for r in all_results.values():
        all_types.update(r.get("by_type", {}).keys())

    if all_types:
        print("\nPer-test-type breakdown:")
        print(f"{'Test Type':<35} {'Tests':>8} {'Score':>8}")
        print("-" * 55)
        for t in sorted(all_types):
            totals = {"passed": 0, "total": 0}
            for r in all_results.values():
                counts = r.get("by_type_counts", {}).get(t, {"passed": 0, "total": 0})
                totals["passed"] += counts["passed"]
                totals["total"] += counts["total"]
            type_score = (
                round(100.0 * totals["passed"] / totals["total"], 1)
                if totals["total"] > 0
                else 0.0
            )
            print(f"{t:<35} {totals['total']:>8} {type_score:>7.1f}%")
        print(sep)


# ── Normalized Edit Distance (OmniDocBench-style text quality metric) ─────────


def normalized_edit_distance(pred: str, ref: str) -> float:
    """
    Character-level Normalized Edit Distance in [0, 1].
    0.0 = identical, 1.0 = completely different.
    """
    pred = normalize_text(pred.strip())
    ref = normalize_text(ref.strip())
    if not ref and not pred:
        return 0.0
    if not ref or not pred:
        return 1.0

    m, n = len(pred), len(ref)
    # Space-optimised single-row DP
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if pred[i - 1] == ref[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n] / max(m, n)
