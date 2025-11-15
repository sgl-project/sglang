#!/usr/bin/env python3
"""Runs vLLM's HF processor to produce MultiModalInputs for parity tests."""
from __future__ import annotations

import base64
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_pillow_stub() -> None:
    if importlib.util.find_spec("PIL") is not None:
        return
    stub_dir = Path(__file__).resolve().parents[2] / "src" / "multimodal" / "pillow_stub"
    if not stub_dir.exists():
        raise RuntimeError("Pillow stub directory missing: " + str(stub_dir))
    if str(stub_dir) not in sys.path:
        sys.path.insert(0, str(stub_dir))


if sys.version_info < (3, 10):
    print("PYTHON_TOO_OLD", file=sys.stderr)
    sys.exit(2)

_ensure_pillow_stub()

try:
    import PIL  # noqa: F401
    importlib.metadata.version("Pillow")
except Exception as exc:  # pragma: no cover
    print(f"FAILED_PIL {exc}", file=sys.stderr)
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".claude" / "vllm"))

from vllm.config.model import ModelConfig  # pylint: disable=wrong-import-position
from vllm.multimodal import MULTIMODAL_REGISTRY  # pylint: disable=wrong-import-position
from vllm.transformers_utils.tokenizer import (  # pylint: disable=wrong-import-position
    cached_tokenizer_from_config,
)


def _decode_image(payload: dict[str, Any]):
    data = base64.b64decode(payload["data"])
    width = int(payload["width"])
    height = int(payload["height"])
    channels = int(payload.get("channels", 3))
    array = np.frombuffer(data, dtype=np.uint8)
    array = array.reshape((height, width, channels))
    from PIL import Image  # type: ignore  # pylint: disable=import-outside-toplevel

    return Image.fromarray(array)


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    payload = json.load(sys.stdin)
    model_id = payload["model_id"]
    prompt = payload["prompt"]
    image_entry = payload["image"]

    config = ModelConfig(model=model_id, trust_remote_code=True)
    tokenizer = cached_tokenizer_from_config(config)
    processor = MULTIMODAL_REGISTRY.create_processor(config, tokenizer=tokenizer)

    mm_data = {"image": [_decode_image(image_entry)]}
    result = processor.apply(
        prompt,
        mm_data,
        hf_processor_mm_kwargs={},
        tokenization_kwargs={},
        mm_uuids=None,
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
