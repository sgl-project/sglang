# OCR Accuracy Benchmark

Evaluates `deepseek-ai/DeepSeek-OCR-2` (and any compatible OCR VLM) on
**olmOCR-bench** (AllenAI), the benchmark explicitly used in DeepSeek-OCR-2
official evaluations.

Targets **olmOCR-bench** because:
- Public HuggingFace dataset with 7,010 deterministic unit tests
- Explicitly cited by DeepSeek-OCR-2 authors
- Clear pass/fail semantics — no heavy CDM/TEDS/LaTeXML dependencies
- Covers 7 challenging document types across 1,403 PDF pages

---

## Setup

```bash
# Step 0 (one-time): download olmOCR-bench including PDFs (~2 GB via Git LFS)
pip install huggingface_hub
hf download --repo-type dataset \
    allenai/olmOCR-bench --local-dir ./olmOCR-bench
# This places bench_data/  (7 JSONL files + pdfs/ directory) under ./olmOCR-bench/

# Required: benchmark dependencies (pymupdf is in sglang[test]; aiohttp/tqdm are in core)
pip install "sglang[test]"
# OR install PDF rendering manually (choose one):
#   pip install pymupdf          # recommended (faster, pure Python wheel)
#   pip install pdf2image        # needs poppler: sudo apt install poppler-utils

# Start the sglang server (matches run.sh in this repo)
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-OCR-2 \
    --host 127.0.0.1 --port 30000
```

> **Why the download step?**
> The olmOCR-bench PDF files are stored in Git LFS on HuggingFace.
> `datasets.load_dataset()` cannot retrieve LFS-backed binary files, so the
> benchmark reads the JSONL test files and PDFs directly from a local clone of
> the repository.

---

## Usage

```bash
# Full benchmark — all 7 splits (~7,010 tests)
python -m benchmark.ocr.bench_sglang \
    --port 30000 \
    --model deepseek-ai/DeepSeek-OCR-2 \
    --split all \
    --concurrency 8 \
    --output-dir ./ocr_bench_results

# Single split
python -m benchmark.ocr.bench_sglang --port 30000 --split arxiv_math --concurrency 16

# Quick smoke-test (50 samples from one split)
python -m benchmark.ocr.bench_sglang --port 30000 --split old_scans --max-samples 50

# Use "Free OCR" prompt instead of markdown conversion
python -m benchmark.ocr.bench_sglang --port 30000 --split all --prompt-mode free_ocr

# Save raw model outputs for inspection
python -m benchmark.ocr.bench_sglang --port 30000 --split multi_column --save-raw-outputs

```

---

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | `30000` | sglang server port |
| `--host` | `127.0.0.1` | sglang server host |
| `--model` | `deepseek-ai/DeepSeek-OCR-2` | Model ID (must match running server) |
| `--split` | `all` | Split name or `all` |
| `--concurrency` | `8` | Concurrent requests to server |
| `--output-dir` | `./ocr_bench_results` | Directory for result JSON files |
| `--max-samples` | `-1` | Limit samples per split (-1 = all) |
| `--prompt-mode` | `markdown` | `markdown` or `free_ocr` |
| `--request-timeout` | `300` | Per-request timeout (seconds) |
| `--render-dpi` | `150` | DPI for PDF → PNG rendering |
| `--save-raw-outputs` | `False` | Include raw OCR text in JSON output |

---

## Test Classes (olmOCR-bench)

| Test Type | Description | Matching strategy |
|-----------|-------------|-------------------|
| `text_presence` | 1–3 sentence text must appear in OCR output | Exact or fuzzy; optional position constraint (first/last N chars) |
| `text_absence` | Header/footer/page-number text must NOT appear | Fuzzy; case-insensitive |
| `natural_reading_order` | Two text spans must appear in the correct order | Soft/fuzzy positional matching |
| `table_accuracy` | Cell value with correct neighbor relationship | Markdown + HTML table parsing |
| `math_formula_accuracy` | LaTeX key-token symbols present in math regions | Symbol-token matching (≥70% threshold) |

> **Note on math**: The official olmOCR-bench uses KaTeX rendering + Playwright for
> bounding-box symbol matching. This benchmark uses a symbol-token proxy (no browser
> dependency). Scores on `arxiv_math` and `old_scans_math` may therefore differ from
> the official leaderboard.

---

## Dataset Splits

| Split | Documents | Tests | Document type |
|-------|-----------|-------|---------------|
| `arxiv_math` | 522 | 2,927 | arXiv math papers |
| `old_scans_math` | 36 | 458 | Scanned math textbooks (Internet Archive) |
| `table_tests` | 188 | 1,020 | Documents with tables |
| `old_scans` | 98 | 526 | Historical / typewritten documents (Library of Congress) |
| `headers_footers` | 266 | 753 | Documents with headers/footers to exclude |
| `multi_column` | 231 | 884 | Multi-column layouts |
| `long_tiny_text` | 62 | 442 | Dense small-print pages |

---

## Reference Scores

Column order matches the [olmOCR README](https://github.com/allenai/olmocr): AR = arxiv_math, OSM = old_scans_math, TA = table_tests, OS = old_scans, HF = headers_footers, MC = multi_column, LTT = long_tiny_text, Base = baseline.

| Model | AR | OSM | TA | OS | HF | MC | LTT | Base | **Overall** |
|-------|:--:|:---:|:--:|:--:|:--:|:--:|:---:|:----:|:-----------:|
| DeepSeek-OCR v1 | 77.2 | 73.6 | 80.2 | 33.3 | 96.1 | 66.4 | 79.4 | 99.8 | **75.7** |
| **DeepSeek-OCR-2** | **82.0** | **72.0** | **77.4** | — | — | — | — | — | **76.3** |
| olmOCR v0.4.0 | 83.0 | 82.3 | 84.9 | 47.7 | 96.1 | 83.7 | 81.9 | 99.7 | **82.4** |
| PaddleOCR-VL\* | 85.7 | 71.0 | 84.1 | 37.8 | 97.0 | 79.9 | 85.7 | 98.5 | **80.0** |
| Mistral OCR API | 77.2 | 67.5 | 60.6 | 29.3 | 93.6 | 71.3 | 77.1 | 99.4 | **72.0** |
| Marker 1.10.1 | 83.8 | 66.8 | 72.9 | 33.5 | 86.6 | 80.0 | 85.7 | 99.3 | **76.1** |
| MinerU 2.5.4\* | 76.6 | 54.6 | 84.9 | 33.7 | 96.6 | 78.2 | 83.5 | 93.7 | **75.2** |

\* = scores reported by model authors, not reproduced by olmOCR team.

DeepSeek-OCR-2 per-split scores for OS/HF/MC/LTT are not officially reported; only the three highlighted splits and overall appear on the [HuggingFace model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2).

> **Note on math scores**: This benchmark uses token-overlap matching (≥70% threshold) rather than the official KaTeX rendering + Playwright bounding-box comparison. Scores on `arxiv_math` and `old_scans_math` will therefore differ from the official leaderboard.

Sources: [olmOCR README](https://github.com/allenai/olmocr), [DeepSeek-OCR-2 HF card](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2).

---

## Output Files

Results are written to `--output-dir`:

```
ocr_bench_results/
├── arxiv_math.json       # per-split detailed results
├── old_scans.json
├── ...
└── summary.json          # aggregated across all evaluated splits
```

Each split JSON contains:
- `overall_score`: % tests passed
- `by_type`: per-test-type pass rate
- `total_tests`, `total_passed`, `error_samples`
- Per-sample `test_results` with `type`, `passed`, optional `error`

---

## Files

| File | Description |
|------|-------------|
| `bench_sglang.py` | Main benchmark runner — loads dataset, sends requests, aggregates |
| `eval_utils.py` | Test evaluators, Normalized Edit Distance metric, aggregation helpers |
| `generate_report.py` | Generates self-contained HTML reports with MathJax from result JSONs |
| `README.md` | This file |
