"""Python shim that mirrors vLLM's HuggingFace processor invocation."""

from __future__ import annotations

import base64
import json
import os
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor


def _ensure_imports() -> None:
    missing: list[str] = []
    try:
        import transformers  # noqa: F401  # pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover - defensive guard
        missing.append("transformers")

    try:
        import PIL  # noqa: F401  # pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover
        missing.append("Pillow")

    if missing:
        raise RuntimeError(
            "Missing Python packages for multimodal processing: "
            + ", ".join(missing)
            + ". Install them via `pip install -r python/requirements-mm.txt`."
        )


@lru_cache(maxsize=8)
def _load_processor(model_id: str, trust_remote_code: bool = True):
    _ensure_imports()
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)


def _decode_image(entry: dict[str, Any]) -> np.ndarray:
    data = base64.b64decode(entry["data"])
    width = int(entry["width"])
    height = int(entry["height"])
    channels = int(entry.get("channels", 3))
    mode = (entry.get("mode") or "rgb").lower()

    array = np.frombuffer(data, dtype=np.uint8)
    expected = width * height * channels
    if array.size != expected:
        raise ValueError(
            f"image payload size mismatch: expected {expected}, got {array.size}"
        )
    array = array.reshape((height, width, channels))

    if mode == "bgr":
        array = array[..., ::-1]
    elif mode != "rgb":
        raise ValueError(f"unsupported image mode: {mode}")

    return array


def _tensor_values(name: str, tensor: torch.Tensor) -> list[dict[str, Any]]:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.ndim == 0:
        tensor = tensor.view(1)

    values = []
    for sample in tensor:
        payload = sample.numpy().tobytes()
        values.append(
            {
                "kind": "tensor",
                "shape": list(sample.shape),
                "dtype": str(sample.dtype).replace("torch.", ""),
                "data": base64.b64encode(payload).decode("ascii"),
            }
        )

    if not values:
        raise ValueError(f"empty tensor output for field '{name}'")

    return values


def _json_value(value: Any) -> dict[str, Any]:
    return {"kind": "json", "value": value}


def _extract_mm_kwargs(outputs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    mm_kwargs: dict[str, list[dict[str, Any]]] = {}
    for name, value in outputs.items():
        if name == "input_ids":
            continue

        if isinstance(value, torch.Tensor):
            mm_kwargs[name] = _tensor_values(name, value)
        elif isinstance(value, (list, tuple, dict, str, int, float, bool)) or value is None:
            mm_kwargs[name] = [_json_value(value)]
        else:
            raise TypeError(f"unsupported hf processor output type for '{name}': {type(value)}")
    return mm_kwargs


def _decode_images(mm_data: dict[str, Any]) -> list[Image.Image]:
    images = []
    for item in mm_data.get("image", []):
        array = _decode_image(item)
        images.append(Image.fromarray(array).convert("RGB"))
    return images


def process_mm(
    model_id: str,
    prompt: str,
    mm_data: dict,
    processor_kwargs: dict | None = None,
    tokenization_kwargs: dict | None = None,
    mm_uuids: dict | None = None,
) -> str:
    """Call the HuggingFace processor and return a JSON blob."""

    processor_kwargs = processor_kwargs or {}
    tokenization_kwargs = tokenization_kwargs or {}

    processor = _load_processor(model_id, processor_kwargs.pop("trust_remote_code", True))

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    images = _decode_images(mm_data)
    processor_outputs = processor(
        text=prompt,
        images=images if images else None,
        return_tensors="pt",
        **processor_kwargs,
        **tokenization_kwargs,
    )

    prompt_ids = processor_outputs["input_ids"].tolist()[0]
    mm_kwargs = _extract_mm_kwargs(processor_outputs)

    response = {
        "prompt_token_ids": prompt_ids,
        "mm_kwargs": mm_kwargs,
        "mm_hashes": mm_uuids or {},
        "mm_placeholders": {},
    }

    return json.dumps(response)
