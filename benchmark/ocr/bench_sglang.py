"""
Benchmark DeepSeek-OCR-2 (and similar OCR VLMs) on olmOCR-bench via a running sglang server.

Usage:
    # 0. Download the dataset (one-time, ~2 GB with PDFs via Git LFS)
    hf download --repo-type dataset \\
        allenai/olmOCR-bench --local-dir ./olmOCR-bench

    # 1. Start the sglang server (matches run.sh)
    python -m sglang.launch_server \\
        --model-path deepseek-ai/DeepSeek-OCR-2 --host 127.0.0.1 --port 30000

    # 2. Run the full benchmark (all 7 splits, ~7,010 tests)
    python -m benchmark.ocr.bench_sglang --port 30000 --split all --concurrency 8

    # 3. Quick run on a single split
    python -m benchmark.ocr.bench_sglang --port 30000 --split arxiv_math --concurrency 16

    # 4. Limit pages for a fast smoke-test
    python -m benchmark.ocr.bench_sglang --port 30000 --split old_scans --max-samples 10

    # 5. Custom dataset location
    python -m benchmark.ocr.bench_sglang --bench-dir /data/olmOCR-bench/bench_data

Dataset:
    allenai/olmOCR-bench  –  7 splits, 1,403 PDFs, 7,010 unit tests
    Splits: arxiv_math | old_scans_math | table_tests | old_scans |
            headers_footers | multi_column | long_tiny_text
    PDFs are stored via Git LFS; hf download (step 0) is required.

Reference scores (olmOCR-bench):
    DeepSeek-OCR v1  : 75.7 ± 1.0
    DeepSeek-OCR-2   : 76.3  (reported on HF model card)
    olmOCR v0.4.0    : 82.4 ± 1.1
    PaddleOCR-VL     : 80.0 ± 1.0
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import aiohttp
from tqdm.asyncio import tqdm as atqdm

# ---------------------------------------------------------------------------
# Paths: allow running from the repo root or from benchmark/ocr/
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_utils import (
    aggregate_results,
    evaluate_olmocr_tests,
    print_results_table,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLMOCR_BENCH_SPLITS = [
    "arxiv_math",
    "old_scans_math",
    "table_tests",
    "old_scans",
    "headers_footers",
    "multi_column",
    "long_tiny_text",
]

# DeepSeek-OCR-2 prompt formats (https://github.com/deepseek-ai/DeepSeek-OCR-2)
_PROMPT_MARKDOWN = "<|grounding|>Convert the document to markdown."
_PROMPT_FREE_OCR = "Free OCR."


# ---------------------------------------------------------------------------
# Argument dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchArgs:
    port: int = 30000
    host: str = "127.0.0.1"
    model: str = "deepseek-ai/DeepSeek-OCR-2"
    split: str = "all"
    concurrency: int = 8
    output_dir: str = "./ocr_bench_results"
    max_samples: int = -1
    prompt_mode: str = "markdown"
    bench_dir: str = "./olmOCR-bench/bench_data"
    request_timeout: int = 300
    save_raw_outputs: bool = False
    render_dpi: int = 150
    debug: bool = False
    debug_accuracy: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--port", type=int, default=BenchArgs.port, help="sglang server port"
        )
        parser.add_argument(
            "--host", type=str, default=BenchArgs.host, help="sglang server host"
        )
        parser.add_argument(
            "--model",
            type=str,
            default=BenchArgs.model,
            help="Model identifier (must match the running server)",
        )
        parser.add_argument(
            "--split",
            type=str,
            default=BenchArgs.split,
            choices=OLMOCR_BENCH_SPLITS + ["all"],
            help="Dataset split to evaluate. Use 'all' for all splits.",
        )
        parser.add_argument(
            "--concurrency",
            type=int,
            default=BenchArgs.concurrency,
            help="Max concurrent requests to the sglang server",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=BenchArgs.output_dir,
            help="Directory for result JSON files",
        )
        parser.add_argument(
            "--max-samples",
            type=int,
            default=BenchArgs.max_samples,
            help="Max samples per split (-1 = all)",
        )
        parser.add_argument(
            "--prompt-mode",
            type=str,
            default=BenchArgs.prompt_mode,
            choices=["markdown", "free_ocr"],
            help=(
                "Prompt mode for the OCR model: "
                "'markdown' → '<|grounding|>Convert the document to markdown.'; "
                "'free_ocr' → 'Free OCR.'"
            ),
        )
        parser.add_argument(
            "--bench-dir",
            type=str,
            default=BenchArgs.bench_dir,
            help=(
                "Local directory containing the olmOCR-bench bench_data/ folder "
                "(JSONL files + pdfs/ sub-directory). Download first with: "
                "hf download --repo-type dataset allenai/olmOCR-bench "
                "--local-dir ./olmOCR-bench"
            ),
        )
        parser.add_argument(
            "--request-timeout",
            type=int,
            default=BenchArgs.request_timeout,
            help="Per-request timeout in seconds",
        )
        parser.add_argument(
            "--save-raw-outputs",
            action="store_true",
            default=False,
            help="Include raw OCR text in result JSON (useful for debugging)",
        )
        parser.add_argument(
            "--render-dpi",
            type=int,
            default=BenchArgs.render_dpi,
            help="DPI for rendering PDF pages to images",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            default=False,
            help=(
                "Enable debug logging: print per-sample errors immediately, "
                "show full tracebacks, and abort on the first server connection failure."
            ),
        )
        parser.add_argument(
            "--debug-accuracy",
            action="store_true",
            default=False,
            help=(
                "Print per-sample accuracy details: input PDF path, expected "
                "expressions/text, OCR output, and pass/fail per test."
            ),
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "BenchArgs":
        import dataclasses

        attrs = [f.name for f in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


# ---------------------------------------------------------------------------
# Server preflight check
# ---------------------------------------------------------------------------


async def preflight_check(api_url: str, model: str, debug: bool) -> None:
    """
    Send a minimal request to the server before the benchmark starts.
    Raises SystemExit with a clear message if the server is unreachable or
    returns an unexpected error.
    """
    print(f"  Preflight check → {api_url} … ", end="", flush=True)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status in (200, 400):  # 400 = bad request but server is alive
                    print("OK")
                    return
                body = await resp.text()
                print(f"FAILED (HTTP {resp.status})")
                raise SystemExit(
                    f"Server returned HTTP {resp.status}.\nResponse: {body[:400]}\n"
                    f"Ensure the server is running: python -m sglang.launch_server "
                    f"--model-path {model} --host 127.0.0.1 --port <PORT>"
                )
    except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as exc:
        print("FAILED")
        msg = (
            f"Cannot reach sglang server at {api_url}\n"
            f"Error: {exc}\n"
            "Check:\n"
            "  1. Is the server running?  (python -m sglang.launch_server ...)\n"
            "  2. Is --port correct?\n"
            "  3. Are you running inside the same docker container as the server?"
        )
        if debug:
            msg += f"\n\nFull traceback:\n{traceback.format_exc()}"
        raise SystemExit(msg)


# ---------------------------------------------------------------------------
# PDF → base64 PNG
# ---------------------------------------------------------------------------


def pdf_page_to_base64_png(pdf_bytes: bytes, page_num: int = 0, dpi: int = 150) -> str:
    """
    Render a single PDF page to a base64-encoded PNG string.

    Tries PyMuPDF (fitz) first; falls back to pdf2image / poppler.
    page_num is 0-indexed.
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page_num >= len(doc):
            page_num = len(doc) - 1
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()
        return base64.b64encode(img_bytes).decode("utf-8")

    except ImportError:
        pass  # Try pdf2image below

    try:
        from pdf2image import convert_from_bytes

        images = convert_from_bytes(
            pdf_bytes, dpi=dpi, first_page=page_num + 1, last_page=page_num + 1
        )
        if not images:
            raise ValueError(f"pdf2image returned no images for page {page_num}")
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except ImportError as exc:
        raise ImportError(
            "No PDF rendering library found. "
            "Install PyMuPDF (pip install pymupdf) or pdf2image (pip install pdf2image)."
        ) from exc


# ---------------------------------------------------------------------------
# OCR request via sglang OpenAI-compatible API
# ---------------------------------------------------------------------------


async def run_ocr_request(
    session: aiohttp.ClientSession,
    api_url: str,
    model: str,
    image_b64: str,
    text_prompt: str,
    timeout: int,
) -> Tuple[str, float]:
    """
    Send an image + text prompt to the sglang /v1/chat/completions endpoint.

    Returns (ocr_text, latency_seconds).
    On error returns ("ERROR: ...", -1.0).
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    t0 = time.perf_counter()
    try:
        async with session.post(
            api_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            latency = time.perf_counter() - t0
            if resp.status != 200:
                body = await resp.text()
                return f"ERROR: HTTP {resp.status} – {body[:200]}", -1.0
            data = await resp.json()
            text = data["choices"][0]["message"].get("content") or ""
            return text, latency
    except asyncio.TimeoutError:
        return "ERROR: request timed out", -1.0
    except Exception:
        return f"ERROR: {traceback.format_exc(limit=3)}", -1.0


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Local dataset loading (olmOCR-bench flat JSONL)
# ---------------------------------------------------------------------------


def load_jsonl_split(bench_dir: Path, split_name: str) -> List[dict]:
    """
    Load test cases from a local olmOCR-bench JSONL file and group them by
    (pdf, page) so each unique PDF page becomes one benchmark "sample".

    Requires the dataset to have been downloaded first::

        hf download --repo-type dataset \\
            allenai/olmOCR-bench --local-dir ./olmOCR-bench
    """
    jsonl_path = bench_dir / f"{split_name}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"JSONL not found: {jsonl_path}\n"
            "Download the dataset first:\n"
            "  hf download --repo-type dataset "
            "--resume-download allenai/olmOCR-bench --local-dir ./olmOCR-bench"
        )

    test_cases: List[dict] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                test_cases.append(json.loads(line))

    # Group test cases by (pdf_relative_path, page)
    pdf_dir = bench_dir / "pdfs"
    groups: Dict[Tuple[str, int], dict] = {}
    for tc in test_cases:
        pdf_rel: str = tc["pdf"]
        page: int = tc.get("page", 1)
        key = (pdf_rel, page)
        if key not in groups:
            groups[key] = {
                "pdf_path": str(pdf_dir / pdf_rel),
                "pdf_rel": pdf_rel,
                "page": page,
                "tests": [],
            }
        groups[key]["tests"].append(tc)

    return list(groups.values())


def _log_sample_error(label: str, error: str, debug: bool) -> None:
    """Print a sample error. Always shows a one-liner; full detail only in debug mode."""
    short = error.splitlines()[0] if error else "unknown error"
    print(f"  [ERROR] {label}: {short}", file=sys.stderr)
    if debug and len(error.splitlines()) > 1:
        print(error, file=sys.stderr)


async def process_sample(
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    api_url: str,
    args: BenchArgs,
    text_prompt: str,
    sample: dict,
) -> dict:
    """Process one benchmark sample (a single PDF page) and return its result dict."""
    label = f"{sample.get('pdf_rel', '?')} page {sample.get('page', '?')}"
    async with semaphore:
        pdf_path = Path(sample["pdf_path"])
        if not pdf_path.exists():
            err = f"PDF not found: {pdf_path}"
            _log_sample_error(label, err, args.debug)
            return {"error": err, "test_results": [], "passed": 0, "total": 0}

        try:
            pdf_bytes = pdf_path.read_bytes()
        except Exception as exc:
            err = f"PDF read error: {exc}"
            _log_sample_error(label, err, args.debug)
            return {"error": err, "test_results": [], "passed": 0, "total": 0}

        page_num = sample["page"] - 1  # convert 1-indexed → 0-indexed

        try:
            image_b64 = pdf_page_to_base64_png(
                pdf_bytes, page_num=page_num, dpi=args.render_dpi
            )
        except Exception as exc:
            err = f"PDF render error: {exc}"
            if args.debug:
                err += "\n" + traceback.format_exc()
            _log_sample_error(label, err, args.debug)
            return {"error": err, "test_results": [], "passed": 0, "total": 0}

        ocr_text, latency = await run_ocr_request(
            session, api_url, args.model, image_b64, text_prompt, args.request_timeout
        )

        if ocr_text.startswith("ERROR:"):
            _log_sample_error(label, ocr_text, args.debug)
            return {"error": ocr_text, "test_results": [], "passed": 0, "total": 0}

        tests = sample["tests"]
        test_results = evaluate_olmocr_tests(tests, ocr_text)

        # Accuracy debug: show input expectations and full OCR output
        if args.debug_accuracy:
            sep = "-" * 72
            print(f"\n{sep}", flush=True)
            print(f"[INPUT ] PDF  : {pdf_path}", flush=True)
            print(
                f"[INPUT ] Page : {sample['page']}  |  {len(tests)} test(s)", flush=True
            )
            for i, t in enumerate(tests):
                ttype = t.get("type", "?")
                if ttype in ("present", "absent", "text_presence", "text_absence"):
                    expected = t.get("text", "")
                elif ttype in ("math", "math_formula_accuracy"):
                    expected = t.get("math") or t.get("latex", "")
                elif ttype in ("order", "natural_reading_order"):
                    expected = (
                        f"before={t.get('before', '')!r}  after={t.get('after', '')!r}"
                    )
                else:
                    expected = str(
                        {
                            k: v
                            for k, v in t.items()
                            if k not in ("pdf", "page", "id", "type", "url", "checked")
                        }
                    )
                print(
                    f"[INPUT ]   [{i+1}] type={ttype!r:12s}  expected: {expected[:120]}",
                    flush=True,
                )
            # Print OCR output (truncate long outputs)
            ocr_preview = (
                ocr_text
                if len(ocr_text) <= 800
                else ocr_text[:800] + f"\n... [{len(ocr_text)} chars total, truncated]"
            )
            print(
                f"[OUTPUT] OCR text ({len(ocr_text)} chars, latency={latency:.2f}s):",
                flush=True,
            )
            print(ocr_preview, flush=True)

        # Log individual test failures in debug mode
        if args.debug or args.debug_accuracy:
            for tr in test_results:
                status = "PASS" if tr.get("passed") else "FAIL"
                detail = tr.get("error", "") if not tr.get("passed") else ""
                suffix = f" | {detail}" if detail else ""
                print(
                    f"[RESULT]   [{status}] type={tr.get('type')!r}{suffix}",
                    flush=True,
                )
            if args.debug_accuracy:
                print(sep, flush=True)

        result: dict = {
            "pdf": sample["pdf_rel"],
            "page": sample["page"],
            "latency": round(latency, 3),
            "test_results": test_results,
            "passed": sum(1 for r in test_results if r.get("passed")),
            "total": len(test_results),
            # Store expected values so the HTML report can render them
            "test_inputs": [
                {
                    "type": t.get("type"),
                    "math": t.get("math") or t.get("latex", ""),
                    "text": t.get("text", ""),
                    "before": t.get("before", ""),
                    "after": t.get("after", ""),
                    # table-specific fields
                    "cell": t.get("cell", ""),
                    "up": t.get("up"),
                    "down": t.get("down"),
                    "left": t.get("left"),
                    "right": t.get("right"),
                    "top_heading": t.get("top_heading"),
                    "left_heading": t.get("left_heading"),
                }
                for t in tests
            ],
        }
        if args.save_raw_outputs:
            result["ocr_output"] = ocr_text
        return result


# ---------------------------------------------------------------------------
# Split-level runner
# ---------------------------------------------------------------------------


async def run_split(
    split_name: str,
    dataset,
    args: BenchArgs,
    api_url: str,
    text_prompt: str,
) -> dict:
    """Evaluate one olmOCR-bench split; return aggregated results dict."""
    samples = list(dataset)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    semaphore = asyncio.Semaphore(args.concurrency)
    sample_results: List[dict] = []
    _first_error: List[str] = []  # capture first error for summary

    connector = aiohttp.TCPConnector(limit=args.concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_sample(semaphore, session, api_url, args, text_prompt, sample)
            for sample in samples
        ]
        for future in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"  [{split_name}]",
            leave=True,
        ):
            result = await future
            if "error" in result and not _first_error:
                _first_error.append(result["error"])
            sample_results.append(result)

    agg = aggregate_results(split_name, sample_results)
    agg["samples"] = sample_results  # include per-sample data for report generation

    # Always surface error summary so silent 0/0 can't happen
    error_count = agg.get("error_samples", 0)
    if error_count > 0:
        first = _first_error[0] if _first_error else "(unknown)"
        short = first.splitlines()[0]
        print(
            f"  WARNING: {error_count}/{len(samples)} samples errored and were skipped.",
            file=sys.stderr,
        )
        print(f"  First error: {short}", file=sys.stderr)
        if not args.debug:
            print(
                "  Re-run with --debug for full per-sample error details.",
                file=sys.stderr,
            )

    return agg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> BenchArgs:
    parser = argparse.ArgumentParser(
        description="Benchmark OCR VLMs on olmOCR-bench via sglang",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    BenchArgs.add_cli_args(parser)
    ns = parser.parse_args()
    return BenchArgs.from_cli_args(ns)


async def main() -> None:
    args = parse_args()

    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    text_prompt = (
        _PROMPT_MARKDOWN if args.prompt_mode == "markdown" else _PROMPT_FREE_OCR
    )
    splits_to_run = OLMOCR_BENCH_SPLITS if args.split == "all" else [args.split]

    print("=" * 60)
    print(f"  OCR Accuracy Benchmark  –  olmOCR-bench")
    print("=" * 60)
    print(f"  Model       : {args.model}")
    print(f"  Server      : {api_url}")
    print(f"  Prompt mode : {args.prompt_mode}")
    print(f"  Splits      : {', '.join(splits_to_run)}")
    print(f"  Concurrency : {args.concurrency}")
    print(f"  Bench dir   : {args.bench_dir}")
    print(f"  Output dir  : {args.output_dir}")
    if args.debug:
        print(f"  Debug mode  : ON (errors)")
    if args.debug_accuracy:
        print(f"  Debug mode  : ON (accuracy — input/output per sample)")
    print("=" * 60)

    await preflight_check(api_url, args.model, args.debug)

    bench_dir = Path(args.bench_dir)
    if not bench_dir.exists():
        raise SystemExit(
            f"Benchmark directory not found: {bench_dir}\n"
            "Download the dataset first:\n"
            "  hf download --repo-type dataset "
            "allenai/olmOCR-bench --local-dir ./olmOCR-bench"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    all_results: Dict[str, dict] = {}

    for split in splits_to_run:
        print(f"\nLoading split '{split}' from {bench_dir} …")
        try:
            samples = load_jsonl_split(bench_dir, split)
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc}")
            continue
        except Exception as exc:
            print(f"  WARNING: could not load split '{split}': {exc}")
            continue

        n = (
            len(samples)
            if args.max_samples <= 0
            else min(len(samples), args.max_samples)
        )
        print(
            f"  {n} PDF pages to evaluate ({sum(len(s['tests']) for s in samples[:n])} tests) …"
        )

        split_result = await run_split(split, samples, args, api_url, text_prompt)
        all_results[split] = split_result

        # Save per-split JSON
        out_path = os.path.join(args.output_dir, f"{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_result, f, indent=2, ensure_ascii=False)
        print(
            f"  Score: {split_result['overall_score']:.1f}%  "
            f"({split_result['total_passed']}/{split_result['total_tests']} tests passed)"
        )
        print(f"  Saved → {out_path}")

    if not all_results:
        print("No results collected – exiting.")
        return

    # Print final table
    print_results_table(all_results)

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull summary saved → {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
