"""
Serve DeepSeek-OCR-2 (and DeepSeek-OCR) with SGLang.

# Start the server (CUDA):
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

Why the env var and --context-length: DeepSeek-OCR-2 pages expand to ~1100 image
tokens at the official 768px local-crop geometry, and long pages generate up to
8192 output tokens. sglang derives a context limit from the model (8192) and
rejects a request whose input+output budget exceeds it at input time; the env
var lets --context-length raise that limit (official vLLM caps total at 8192).

The DeepSeek-OCR-2 processor geometry (768px local crops) is applied
automatically by the server for OCR-2; DeepSeek-OCR stays at 640px.
"""

import argparse
import os
import re

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--image", help="single document page image path/URL")
    ap.add_argument("--image-dir", help="dir of page images to batch (jpg/png)")
    ap.add_argument("--output", help="dir for batch per-page .md predictions")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}{ROUTE}"

    if args.image:
        print(run_one(url, args.image, args.max_new_tokens))
        return

    images = sorted(
        p
        for p in os.listdir(args.image_dir)
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    os.makedirs(args.output, exist_ok=True)
    for name in images:
        text = run_one(url, os.path.join(args.image_dir, name), args.max_new_tokens)
        with open(os.path.join(args.output, md_name(name)), "w", encoding="utf-8") as f:
            f.write(postprocess(text))
        print(f"wrote {md_name(name)}")
    print(f"done: {len(images)} pages -> {args.output}")


if __name__ == "__main__":
    main()
