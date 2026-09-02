"""
Generate a self-contained HTML verification report from olmOCR-bench results.

Requires results saved with --save-raw-outputs.

Usage:
    # 1. Run benchmark with raw outputs saved
    python benchmark/ocr/bench_sglang.py --port 30000 --split arxiv_math \\
        --max-samples 20 --save-raw-outputs

    # 2. Generate HTML report for a single split
    python benchmark/ocr/generate_report.py --split arxiv_math

    # 3. Generate HTML report for all splits in a results directory
    python benchmark/ocr/generate_report.py --results-dir ./ocr_bench_results

    # 4. Show only failing tests
    python benchmark/ocr/generate_report.py --split arxiv_math --failures-only

Open the generated .html file in any browser — formulas are rendered via MathJax.
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MATH_DELIM_RE = re.compile(
    r"(\\\[[\s\S]*?\\\]"  # \[...\]
    r"|\\\([\s\S]*?\\\)"  # \(...\)
    r"|\$\$[\s\S]*?\$\$"  # $$...$$
    r"|\$[^$\n]+?\$)",  # $...$
    re.DOTALL,
)


def _ocr_to_html(text: str) -> str:
    """
    Convert raw OCR output to readable HTML.
    Preserves LaTeX delimiters for MathJax, strips bounding-box annotations,
    and wraps block-level elements in <p> tags.
    """
    lines = text.splitlines()
    out_lines = []
    for line in lines:
        # Strip bounding box annotations like  text[[x1, y1, x2, y2]]
        line = re.sub(r"^\s*\w[\w_]*\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]\s*", "", line)
        # HTML-escape everything EXCEPT LaTeX delimiters
        parts = _MATH_DELIM_RE.split(line)
        escaped = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                escaped += html.escape(part)
            else:
                escaped += part  # LaTeX — pass through for MathJax
        out_lines.append(escaped)
    return "<br>".join(out_lines)


def _latex_to_display(latex: str) -> str:
    """Wrap a raw LaTeX string in display-math delimiters for MathJax."""
    stripped = latex.strip()
    # Already delimited → pass through
    if stripped.startswith(("\\[", "$$", "\\(")):
        return stripped
    return f"\\[ {stripped} \\]"


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script>
  window.MathJax = {{
    tex: {{ inlineMath: [['\\\\(','\\\\)'], ['$','$']], displayMath: [['\\\\[','\\\\]'],['$$','$$']] }},
    options: {{ skipHtmlTags: ['script','noscript','style','textarea'] }}
  }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background: #f5f5f5; color: #222; }}
  h1   {{ background: #1a73e8; color: #fff; margin: 0; padding: 16px 24px; font-size: 1.3em; }}
  .summary {{ background: #fff; border-bottom: 1px solid #ddd; padding: 12px 24px;
              display: flex; gap: 32px; flex-wrap: wrap; font-size: 0.9em; }}
  .stat {{ display: flex; flex-direction: column; }}
  .stat span:first-child {{ font-weight: 600; font-size: 1.1em; }}
  .sample {{ background: #fff; border: 1px solid #ddd; border-radius: 6px;
             margin: 16px 24px; overflow: hidden; }}
  .sample-header {{ padding: 10px 16px; font-weight: 600; font-size: 0.85em;
                    background: #f0f0f0; display: flex; justify-content: space-between;
                    align-items: center; cursor: pointer; user-select: none; }}
  .sample-header:hover {{ background: #e8e8e8; }}
  .sample-body {{ padding: 0 16px 16px; }}
  .tests {{ margin-top: 12px; display: flex; flex-direction: column; gap: 12px; }}
  .test  {{ border: 1px solid #ddd; border-radius: 4px; overflow: hidden; }}
  .test-header {{ padding: 6px 12px; font-size: 0.8em; font-weight: 600;
                  display: flex; gap: 12px; align-items: center; }}
  .pass {{ background: #e6f4ea; border-left: 4px solid #34a853; }}
  .fail {{ background: #fce8e6; border-left: 4px solid #ea4335; }}
  .test-body {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}
  .pane {{ padding: 10px 14px; font-size: 0.85em; }}
  .pane:first-child {{ border-right: 1px solid #ddd; }}
  .pane h4 {{ margin: 0 0 6px; font-size: 0.75em; text-transform: uppercase;
              letter-spacing: 0.05em; color: #666; }}
  .pane pre {{ margin: 0; white-space: pre-wrap; word-break: break-all;
               font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.9em;
               background: #f8f8f8; padding: 6px 8px; border-radius: 3px; }}
  .rendered {{ font-size: 1em; padding: 4px 0; min-height: 24px; }}
  .ocr-output {{ grid-column: 1 / -1; padding: 10px 14px; font-size: 0.82em; }}
  .ocr-output h4 {{ margin: 0 0 6px; font-size: 0.75em; text-transform: uppercase;
                    letter-spacing: 0.05em; color: #666; }}
  .ocr-raw  {{ white-space: pre-wrap; word-break: break-all; max-height: 300px;
               overflow-y: auto; background: #f8f8f8; padding: 8px 10px; border-radius: 3px;
               font-family: monospace; font-size: 0.85em; line-height: 1.5; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
            font-size: 0.78em; font-weight: 700; }}
  .badge-pass {{ background: #34a853; color: #fff; }}
  .badge-fail {{ background: #ea4335; color: #fff; }}
  .badge-score {{ background: #1a73e8; color: #fff; }}
  .toggle {{ font-size: 0.8em; color: #666; }}
  details summary {{ list-style: none; }}
  details summary::-webkit-details-marker {{ display: none; }}
</style>
</head>
<body>
<h1>{title}</h1>
"""

_HTML_TAIL = """\
</body>
</html>
"""


def _render_sample(sample: dict, failures_only: bool) -> str:
    """Render one sample (PDF page) as an HTML block."""
    pdf = sample.get("pdf", sample.get("pdf_rel", "?"))
    page = sample.get("page", "?")
    passed = sample.get("passed", 0)
    total = sample.get("total", 0)
    error = sample.get("error")
    ocr_output = sample.get("ocr_output", "")
    test_results = sample.get("test_results", [])
    test_inputs = sample.get("test_inputs", [])

    if failures_only and passed == total and not error:
        return ""

    pct = f"{100*passed//total}%" if total else "—"
    header_cls = "fail" if (error or passed < total) else "pass"

    parts = [f'<div class="sample">']
    parts.append(
        f'<details {"open" if (error or passed < total) else ""}>'
        f'<summary class="sample-header {header_cls}">'
        f"<span>📄 {html.escape(pdf)}  &nbsp;·&nbsp; page {page}</span>"
        f"<span>"
        f'<span class="badge badge-score">{passed}/{total} &nbsp;{pct}</span>'
        f"</span>"
        f"</summary>"
    )
    parts.append('<div class="sample-body">')

    if error:
        parts.append(
            f'<p style="color:#ea4335;font-weight:600">ERROR: {html.escape(str(error))}</p>'
        )
    else:
        parts.append('<div class="tests">')
        for i, tr in enumerate(test_results):
            if failures_only and tr.get("passed"):
                continue
            ttype = tr.get("type", "?")
            ok = tr.get("passed", False)
            badge = (
                '<span class="badge badge-pass">PASS</span>'
                if ok
                else '<span class="badge badge-fail">FAIL</span>'
            )
            test_err = tr.get("error", "")
            # Get expected values from test_inputs (parallel list)
            ti = test_inputs[i] if i < len(test_inputs) else {}

            test_cls = "pass" if ok else "fail"
            parts.append(f'<div class="test">')
            parts.append(
                f'<div class="test-header {test_cls}">'
                f"{badge} &nbsp; <code>type={ttype!r}</code>"
                + (
                    f' &nbsp; <span style="color:#c00">{html.escape(test_err)}</span>'
                    if test_err
                    else ""
                )
                + f"</div>"
            )
            # Expected pane
            parts.append('<div class="test-body">')
            parts.append('<div class="pane">')
            parts.append("<h4>Expected</h4>")
            if ttype in ("math", "math_formula_accuracy"):
                latex = ti.get("math", "")
                parts.append(f"<pre>{html.escape(latex)}</pre>")
                if latex:
                    parts.append(
                        f'<div class="rendered">{_latex_to_display(latex)}</div>'
                    )
            elif ttype in ("present", "absent", "text_presence", "text_absence"):
                parts.append(f'<pre>{html.escape(ti.get("text", ""))}</pre>')
            elif ttype in ("order", "natural_reading_order"):
                before = ti.get("before", "")
                after = ti.get("after", "")
                parts.append(
                    f"<pre>before: {html.escape(before)}\nafter:  {html.escape(after)}</pre>"
                )
            parts.append("</div>")
            # Right pane — OCR formulas extracted (if math type)
            parts.append('<div class="pane">')
            parts.append("<h4>OCR extracted formulas</h4>")
            if ttype in ("math", "math_formula_accuracy") and ocr_output:
                matches = _MATH_DELIM_RE.findall(ocr_output)
                if matches:
                    for m in matches[:6]:  # show at most 6 matches
                        parts.append(f'<div class="rendered">{m}</div>')
                    if len(matches) > 6:
                        parts.append(
                            f'<p style="color:#888;font-size:0.8em">… and {len(matches)-6} more</p>'
                        )
                else:
                    parts.append(
                        '<p style="color:#888;font-size:0.8em">No LaTeX delimiters found in OCR output.</p>'
                    )
            else:
                parts.append(
                    '<p style="color:#888;font-size:0.8em">(see full OCR output below)</p>'
                )
            parts.append("</div>")
            parts.append("</div>")  # test-body
            parts.append("</div>")  # test

        parts.append("</div>")  # tests

    # Full OCR output
    if ocr_output:
        parts.append('<div class="ocr-output">')
        parts.append(f"<h4>Full OCR output ({len(ocr_output)} chars)</h4>")
        parts.append(f'<div class="ocr-raw">{_ocr_to_html(ocr_output)}</div>')
        parts.append("</div>")
    elif not error:
        parts.append(
            '<p style="color:#888;font-size:0.85em">No raw OCR output stored. '
            "Re-run with <code>--save-raw-outputs</code> to include it here.</p>"
        )

    parts.append("</div>")  # sample-body
    parts.append("</details>")
    parts.append("</div>")  # sample
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    split_name: str,
    split_result: dict,
    out_path: Path,
    failures_only: bool = False,
) -> None:
    total_passed = split_result.get("total_passed", 0)
    total_tests = split_result.get("total_tests", 0)
    overall = split_result.get("overall_score", 0.0)
    error_samples = split_result.get("error_samples", 0)
    total_samples = split_result.get("total_samples", 0)

    title = f"OCR Bench — {split_name}"
    body = _HTML_HEAD.format(title=title)

    # Summary bar
    by_type = split_result.get("by_type", {})
    type_badges = " &nbsp; ".join(
        f'<span class="badge badge-score">{t}: {v:.1f}%</span>'
        for t, v in by_type.items()
    )
    body += f"""
<div class="summary">
  <div class="stat"><span>{total_passed}/{total_tests}</span><span>tests passed</span></div>
  <div class="stat"><span>{overall:.1f}%</span><span>overall score</span></div>
  <div class="stat"><span>{error_samples}/{total_samples}</span><span>error samples</span></div>
  <div class="stat">{type_badges}</div>
</div>
"""
    if failures_only:
        body += '<p style="margin:12px 24px;color:#888;font-size:0.9em">Showing failures only.</p>'

    samples = split_result.get("samples", [])
    if not samples:
        body += '<p style="margin:24px;color:#888">No per-sample data found. The result JSON may not include sample-level details.</p>'
    else:
        for sample in samples:
            body += _render_sample(sample, failures_only)

    body += _HTML_TAIL
    out_path.write_text(body, encoding="utf-8")
    print(f"Report → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HTML verification report from olmOCR-bench results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default="./ocr_bench_results",
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Single split to report (default: all JSONs in results-dir)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write HTML files (default: same as results-dir)",
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Include only samples/tests that failed",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split:
        json_files = [results_dir / f"{args.split}.json"]
    else:
        json_files = sorted(results_dir.glob("*.json"))
        json_files = [f for f in json_files if f.stem != "summary"]

    if not json_files:
        sys.exit(f"No result JSON files found in {results_dir}")

    for jf in json_files:
        if not jf.exists():
            print(f"  SKIP (not found): {jf}")
            continue
        split_name = jf.stem
        with open(jf, encoding="utf-8") as fh:
            data = json.load(fh)

        suffix = "_failures" if args.failures_only else ""
        out_path = output_dir / f"{split_name}{suffix}_report.html"
        generate_report(split_name, data, out_path, failures_only=args.failures_only)


if __name__ == "__main__":
    main()
