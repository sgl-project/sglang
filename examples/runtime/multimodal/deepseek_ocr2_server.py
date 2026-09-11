"""
Serve DeepSeek-OCR-2 (and DeepSeek-OCR) with SGLang, and run pages through it.

# Start the server. Pick one of the two context settings:

# (A) Official setting: total input+output stays within the model's 8192-token
#     window, which is what the official vLLM recipe does and what makes the
#     numbers comparable to it. No env var is needed: the scheduler clamps each
#     page's output budget down to `8192 - expanded_input` (see below).
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-OCR-2 --enable-multimodal \
    --context-length 8192 --allow-auto-truncate --enable-custom-logit-processor

# (B) Full output budget on long pages. Any --context-length above the derived
#     8192 needs this env var (9000 needs it exactly as much as 16384 does), and
#     pages whose input+output actually exceed 8192 then run past the trained
#     window, so their scores are not comparable to the official ones.
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-OCR-2 --enable-multimodal \
    --context-length 16384 --enable-custom-logit-processor

# Ascend NPU (add the Ascend flags):
#   --device npu --mm-attention-backend ascend_attn --attention-backend ascend \
#   --page-size 128 --disable-cuda-graph --skip-server-warmup

Then either run one document page through it:

    python examples/runtime/multimodal/deepseek_ocr2_server.py --image page.png

or batch a whole image directory into per-page markdown files (each named
<image-stem>.md), the layout the official OmniDocBench scorer consumes:

    python examples/runtime/multimodal/deepseek_ocr2_server.py \
        --image-dir /path/to/pages --output /path/to/pred_md

Why --max-new-tokens defaults to 8192: a 768px page expands to ~1100 image tokens
(up to 6 local crops of 144 plus the 256-token global view). Under setting (A)
(`--context-length 8192`) run the server with `--allow-auto-truncate`: sglang then
clamps each request's output budget to the remaining context
(`context_len - expanded_input`), which is the official "total <= 8192" semantics
and lets the full 8192 ceiling be requested. Without `--allow-auto-truncate` a
request whose `max_new_tokens + expanded_input` exceeds the context length is
rejected (400); lower the ceiling to fit if you need to run without the flag.

The DeepSeek-OCR-2 processor geometry (768px local crops) is applied
automatically by the server for OCR-2; DeepSeek-OCR stays at 640px.
"""

import argparse
import os
import re
import sys

import requests

from sglang.srt.sampling.custom_logit_processor import (
    DeepseekOCRNoRepeatNGramLogitProcessor,
)

PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
ROUTE = "/generate"
_REF_DET_RE = re.compile(r"(<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>)", re.DOTALL)
_FORMULA_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)


def postprocess(text: str) -> str:
    """Official-style cleanup: strip grounding boxes, tidy formula LaTeX.

    The official DeepSeek-OCR-2 harness stores markdown with the
    <|ref|>...<|/det|> grounding spans removed and formula-internal
    \\quad(...) annotations stripped. Mirroring it lets the batch output be
    scored directly by the OmniDocBench pdf_validation harness.
    """
    text = _REF_DET_RE.sub("", text)
    text = text.replace("\n\n\n\n", "\n\n").replace("\n\n\n", "\n\n")

    def process(m):
        return r"\[" + re.sub(r"\\quad\s*\([^)]*\)", "", m.group(1)).strip() + r"\]"

    return _FORMULA_RE.sub(process, text)


def md_name(image_path: str) -> str:
    base = os.path.basename(image_path)
    for ext in (".jpg", ".jpeg", ".png"):
        if base.lower().endswith(ext):
            return base[: -len(ext)] + ".md"
    return os.path.splitext(base)[0] + ".md"


def build_payload(image, max_new_tokens):
    return {
        "text": PROMPT,
        "image_data": image,
        "sampling_params": {
            "temperature": 0.0,  # greedy; sglang maps temp 0 -> top_k 1
            "max_new_tokens": max_new_tokens,
            # keep <|ref|>/<|det|>/<|grounding|> literal so the grounding spans
            # survive for postprocess()
            "skip_special_tokens": False,
            "custom_params": {
                # Official DeepSeek-OCR recipe values.
                "ngram_size": 40,
                "window_size": 90,
                "whitelist_token_ids": [128821, 128822],  # <td> </td>
            },
        },
        "custom_logit_processor": DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),
    }


def run_one(url, image, max_new_tokens):
    r = requests.post(url, json=build_payload(image, max_new_tokens), timeout=3600)
    r.raise_for_status()
    out = r.json()
    return out["text"] if isinstance(out, dict) else str(out)


def main():
    ap = argparse.ArgumentParser(
        description="Run DeepSeek-OCR-2 over one page or a directory of pages.",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="single document page image path/URL")
    source.add_argument("--image-dir", help="dir of page images to batch (jpg/png)")
    ap.add_argument(
        "--output", help="dir for per-page .md predictions (required with --image-dir)"
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="output ceiling; with --allow-auto-truncate the server clamps it to the remaining context",
    )
    ap.add_argument(
        "--raw",
        action="store_true",
        help="emit the raw model output instead of running postprocess() on it",
    )
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}{ROUTE}"

    if args.image:
        text = run_one(url, args.image, args.max_new_tokens)
        print(text if args.raw else postprocess(text))
        return 0

    if not args.output:
        ap.error("--output is required with --image-dir")

    images = sorted(
        p
        for p in os.listdir(args.image_dir)
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not images:
        ap.error(f"no jpg/jpeg/png images found in {args.image_dir}")
    os.makedirs(args.output, exist_ok=True)

    failures = []
    for idx, name in enumerate(images, start=1):
        try:
            text = run_one(url, os.path.join(args.image_dir, name), args.max_new_tokens)
        except Exception as exc:  # one bad page must not discard the whole run
            failures.append((name, exc))
            print(f"[{idx}/{len(images)}] FAILED {name}: {exc}", file=sys.stderr)
            continue
        with open(os.path.join(args.output, md_name(name)), "w", encoding="utf-8") as f:
            f.write(text if args.raw else postprocess(text))
        print(f"[{idx}/{len(images)}] wrote {md_name(name)}")

    print(f"done: {len(images) - len(failures)}/{len(images)} pages -> {args.output}")
    if failures:
        print(
            f"failed pages ({len(failures)}): "
            + ", ".join(name for name, _ in failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
