"""
Serve DeepSeek-OCR-2 (and DeepSeek-OCR) with SGLang.

# Start the server (CUDA):
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-OCR-2 --enable-multimodal \
    --context-length 16384 --enable-custom-logit-processor

# Ascend NPU (add the Ascend flags):
#   --device npu --mm-attention-backend ascend_attn --attention-backend ascend \
#   --page-size 128 --disable-cuda-graph --skip-server-warmup

# Then run a document page through it:
python examples/runtime/multimodal/deepseek_ocr2_server.py --image page.png

Why the env var and --context-length: DeepSeek-OCR-2 pages expand to ~1100 image
tokens at the official 768px local-crop geometry, and long pages generate up to
8192 output tokens. sglang derives a context limit from the model (8192) and
rejects a request whose input+output budget exceeds it at input time; the env
var lets --context-length raise that limit (official vLLM caps total at 8192).

The DeepSeek-OCR-2 processor geometry (768px local crops) is applied
automatically by the server for OCR-2; DeepSeek-OCR stays at 640px.
"""

import argparse
import json

import requests

from sglang.srt.sampling.custom_logit_processor import (
    DeepseekOCRNoRepeatNGramLogitProcessor,
)

PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
ROUTE = "/generate"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--image", required=True, help="path/URL of a document page image")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    payload = {
        "text": PROMPT,
        "image_data": args.image,
        "sampling_params": {
            "temperature": 0.0,  # greedy; sglang maps temp 0 -> top_k 1
            "max_new_tokens": args.max_new_tokens,
            # keep <|ref|>/<|det|>/<|grounding|> literal so the grounding spans
            # survive for downstream stripping
            "skip_special_tokens": False,
            "custom_params": {
                "ngram_size": 40,
                "window_size": 90,
                "whitelist_token_ids": [128821, 128822],  # <td> </td>
            },
        },
        "custom_logit_processor": DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),
    }
    resp = requests.post(f"http://{args.host}:{args.port}{ROUTE}", json=payload, timeout=600)
    resp.raise_for_status()
    out = resp.json()
    print(out["text"] if isinstance(out, dict) else out)


if __name__ == "__main__":
    main()
